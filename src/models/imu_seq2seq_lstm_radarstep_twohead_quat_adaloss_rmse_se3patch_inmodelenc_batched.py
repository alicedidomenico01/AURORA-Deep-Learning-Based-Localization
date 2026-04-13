#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from src.fusion.fusion_spatial_a2do_v2 import A2DOHierarchicalFilter2Modalities

from src.dataloaders.radar4d_cam_pose_dataloader_raw import SEQ_META

from src.dataloaders.radar4d_cam_pose_dataloader_raw import SeqConfigCam, Radar4DPlusCamDatasetPairsRAW
from src.dataloaders.radar4d_cam_pose_dataloader_raw_batched import make_loaders_radar_cam_pose_raw_batched

from src.fusion.arcfm_monoscale import AdaptiveRadarCameraFusionMono

from src.utils.weather_scalar_logger import WeatherScalarLogger

from src.dataloaders.radar4d_cam_imu_pose_dataloader_raw_batched import (
    make_loaders_radar_cam_imu_pose_raw_batched,
    SeqConfigCamImu,
    pad_collate_radar_cam_imu_raw_batched,
    Radar4DCamImuDatasetPairsRAW,
)


logger = WeatherScalarLogger()
global_step = 0

import cv2

import sys
if '/mnt/data' not in sys.path:
    sys.path.append('/mnt/data')


from src.dataloaders.radar4d_pose_dataloader_raw import (
    SeqConfig, make_loaders_radar_pose_raw, pad_collate_radar_raw
)

from src.dataloaders.radar4d_pose_dataloader_raw_batched import (
    make_loaders_radar_pose_raw_batched, pad_collate_radar_raw_batched
)

# ===== encoder PN++ invariato =====
from src.encoders.radar4d_encoder_pn2 import Radar4DEncoderPN2, as_BCN, as_BNC
from src.backbones.camera_backbone_resnet18 import ImageBackboneResNet18

# ============================
# Hyperparam di default (identici ai tuoi)
# ============================
BATCH_SIZE = 8
EPOCHS = 25
LR = 1e-4
HIDDEN_DIM = 512
NUM_LAYERS = 2
SEED = 0

# ===== A2DO / Gumbel schedule =====
A2DO_WARMUP_EPOCHS = 5        # epoche in cui maschera "soft" fissa
A2DO_WARMUP_KEEP_PROB = 0.5   # keep prob durante warmup (0.5 è standard)

A2DO_TAU0 = 1.0               # tau iniziale 
A2DO_TAU_MIN = 0.3            # tau finale 
A2DO_TAU_DECAY = 0.97         # decay per epoca (esponenziale)
A2DO_HARD_TRAIN = False       # IMPORTANT: training soft
A2DO_HARD_EVAL  = False        # eval soft



# Loss 


class AdaptiveLoss(nn.Module):  #modulo Pytorch può essere usato in training loop
    def __init__(self,
                 wq_init: float = -1.0, wt_init: float = 0.0,
                 wq_min: float = -10.0, wq_max: float = 2.0,
                 wt_min: float = -6.0,  wt_max: float = 1.5,
                 huber_delta_deg: float = 1.0):
        super().__init__()    #chiama il costruttore
        self.w_q = nn.Parameter(torch.tensor(wq_init, dtype=torch.float32))  #parametro ottimizzabile con backpropagation
        self.w_t = nn.Parameter(torch.tensor(wt_init, dtype=torch.float32))
        self.wq_min, self.wq_max = float(wq_min), float(wq_max)
        self.wt_min, self.wt_max = float(wt_min), float(wt_max)
        self.delta = float(huber_delta_deg) * math.pi / 180.0   #conversione in radianti

    def forward(self, pred_q, pred_t, gt_q, gt_t):   #definisce come calcolare la loss
        pred_q = F.normalize(pred_q, p=2, dim=-1)
        gt_q   = F.normalize(gt_q,   p=2, dim=-1)
        dot = torch.sum(pred_q * gt_q, dim=-1, keepdim=True)  #risoluzione ambiguità q -q
        pred_q = torch.where(dot < 0.0, -pred_q, pred_q)

        dot_abs = torch.clamp(torch.sum(pred_q * gt_q, dim=-1).abs(), 0.0, 1.0 - 1e-7)  #calcola errore angolare quaternioni
        angle = 2.0 * torch.acos(dot_abs)           # radianti

        abs_a = torch.abs(angle)
        delta = torch.tensor(self.delta, device=angle.device, dtype=angle.dtype)
        loss_q = torch.where(abs_a <= delta, 0.5 * (abs_a * abs_a) / delta, abs_a - 0.5 * delta).mean()  #Huber Loss

        loss_t = F.smooth_l1_loss(pred_t, gt_t)

        w_q = torch.clamp(self.w_q, min=self.wq_min, max=self.wq_max)
        w_t = torch.clamp(self.w_t, min=self.wt_min, max=self.wt_max)
        return loss_q * torch.exp(-w_q) + w_q + loss_t * torch.exp(-w_t) + w_t

def per_sample_pose_losses(pred_q, pred_t, gt_q, gt_t, huber_delta_rad: float):
    """
    Ritorna (l_q, l_t) per-sample, shape [B].
    Coerente con AdaptiveLoss:
      - quat: angolo = 2*acos(|dot|) e poi Huber(angle; delta)
      - trans: smooth_l1 per componente, poi mean sulle 3 componenti
    """
    # --- quat normalize + sign continuity (come AdaptiveLoss) ---
    pred_q = F.normalize(pred_q, p=2, dim=-1)
    gt_q   = F.normalize(gt_q,   p=2, dim=-1)

    dot = torch.sum(pred_q * gt_q, dim=-1, keepdim=True)
    pred_q = torch.where(dot < 0.0, -pred_q, pred_q)

    dot_abs = torch.clamp(torch.sum(pred_q * gt_q, dim=-1).abs(), 0.0, 1.0 - 1e-7)
    angle = 2.0 * torch.acos(dot_abs)  # [B] in radianti

    # --- Huber per-sample su angle ---
    abs_a = torch.abs(angle)
    delta = torch.tensor(huber_delta_rad, device=angle.device, dtype=angle.dtype)
    l_q = torch.where(
        abs_a <= delta,
        0.5 * (abs_a * abs_a) / delta,
        abs_a - 0.5 * delta
    )  # [B]  output rimane per sample

    # --- SmoothL1 per-sample su t ---
    l_t_vec = F.smooth_l1_loss(pred_t, gt_t, reduction="none")  # [B,3]
    l_t = l_t_vec.mean(dim=1)  # [B] (mean sulle 3 componenti)

    return l_q, l_t

def knn_interpolate_xyz_feat(xyz_src, feat_src, xyz_dst, k=3):
        """
        xyz_src:  (B, Ns, 3) coarse
        feat_src: (B, Ns, C)
        xyz_dst:  (B, Nd, 3)  fine
        ritorna:  (B, Nd, C) - feature interpolate sui vicini più prossimi. è la funzione per fare coarse to fine
        """
        B, Ns, _ = xyz_src.shape
        _, Nd, _ = xyz_dst.shape
        C = feat_src.size(2)
        device = xyz_src.device

        # distanze euclidee tra ogni punto fine e tutti i punti coarse: (B, Nd, Ns)
        dist = torch.cdist(xyz_dst, xyz_src)  # richiede PyTorch >= 1.1

        # prendiamo i k vicini più vicini: indici e distanze
        knn_dist, knn_idx = torch.topk(dist, k, dim=-1, largest=False)  # (B, Nd, k)

        # pesi ~ 1/d, normalizzati più vicono peso maggiore
        inv = 1.0 / (knn_dist + 1e-8)               # (B, Nd, k)
        weight = inv / inv.sum(-1, keepdim=True)    # (B, Nd, k)

        # raccogliamo le feature dei vicini: feat_src[batch, idx]
        batch_idx = torch.arange(B, device=device).view(B, 1, 1)  # (B,1,1)
        # feat_knn: (B, Nd, k, C)
        feat_knn = feat_src[batch_idx, knn_idx]

        # interpolazione pesata sui k vicini: (B, Nd, C)
        feat_dst = (feat_knn * weight.unsqueeze(-1)).sum(dim=2)

        return feat_dst
class EnvHead(nn.Module):
    """
    Input:  g = [global_pc ; global_img]  (B, in_dim)

    Output:
      - s_w in (0,1): severità meteo (supervisionata)
      - s_l in (0,1): severità illuminazione (supervisionata)
      - w   (B, emb_dim): embedding per FiLM (weather_emb), dipendente da [s_w, s_l, h]
    """
    def __init__(self, in_dim: int, emb_dim: int = 32, h_dim: int = 128):
        super().__init__()

        # feature ricca "h" derivata da g
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, h_dim),
            nn.ReLU(inplace=True),
        )

        self.head_s_w = nn.Linear(h_dim, 1)
        self.head_s_l = nn.Linear(h_dim, 1)

        # w dipende da [s_w, s_l, h]
        self.mlp_w = nn.Sequential(
            nn.Linear(h_dim + 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, emb_dim),
        )
        self.head_q_cam = nn.Sequential(
            nn.Linear(h_dim + 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, g: torch.Tensor):
        h = self.trunk(g)                      # (B,h_dim)
        s_w = torch.sigmoid(self.head_s_w(h))  # (B,1)
        s_l = torch.sigmoid(self.head_s_l(h))  # (B,1)
        w_in = torch.cat([s_w, s_l, h], dim=1)  # (B,h_dim+2)
        w = self.mlp_w(w_in)                   # (B,emb_dim)

        q_cam = torch.sigmoid(self.head_q_cam(w_in))  # (B,1)

        
        return s_w, s_l, w, q_cam



def _set_a2do_hard(model, hard: bool):
    """
    Tenta di settare hard_gumbel su TUTTI i sottomoduli che lo espongono.
    Funziona anche se cambia il path degli attributi, finché esiste 'hard_gumbel'. Non entra con soft
    """
    for m in model.modules():
        if hasattr(m, "hard_gumbel"):
            try:
                m.hard_gumbel = bool(hard)
            except Exception:
                pass

# ============================
# Modello (forward compatibile legacy + batched)
# ============================
class Radar4DEncLSTM(nn.Module):
    """
    Forward input:
      - LEGACY: pairs_batch = list di B elementi; ognuno è lista di L coppie numpy
      - BATCHED: pairs_batch = dict con Xt,Ft,Xt1,Ft1,validN_t,pair_b,pair_t
                  opzionale: cam (dict) con I_t, I_t1, K, T_cam_from_imu, img_size
      - lengths: tensor (B,)
    Output:
      - pred_q (B,4), pred_t (B,3)
    """
    def __init__(self, enc_out_ch=256, hidden=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=0.1,
                 enc_kwargs=None, use_weather: bool = True):
        super().__init__()
        self.use_weather = bool(use_weather)

        # ============================================================
        # IMU encoder (stessa logica dell'IMU-only: LSTM + packed seq)
        # ============================================================
        self.imu_in_dim = None          # lo settiamo al primo forward (lazy)
        self.imu_enc_hidden = 128
        self.imu_enc_layers = 2
        self.imu_dropout = 0.1

        self.imu_encoder = None         # creato lazy al primo batch IMU
        self.last_imu_feat = None       # parcheggio per fusione futura

        ekw = dict(in_ch=3, radii_norm=(0.01,0.02,0.05), nsamples=(8,16,32),
                   width=64, cv_radius_norm=0.05, cv_nsample=32, out_ch=enc_out_ch)
        if enc_kwargs:
            ekw.update(enc_kwargs or {})
        self.encoder = Radar4DEncoderPN2(**ekw)
        self.enc_width = ekw.get("width", 64)

        # Backbone camera (ResNet18)
        self.img_backbone = ImageBackboneResNet18(
            out_channels=self.enc_width,
            use_stride16=True,
            freeze_at_start=True
        )

        # Dim weather embedding
        # Dim weather embedding (rimane definita, ma può essere disabilitata via flag)
        self.weather_emb_dim = 32  # puoi cambiare se vuoi
        self.last_weather_s_t = None    # (B,) per frame t
        self.last_weather_s_t1 = None   # (B,) per frame t+1 (se ti serve)
        self.last_s_weather_t = None
        self.last_s_weather_t1 = None
        self.last_s_illum_t = None
        self.last_s_illum_t1 = None

        # Se use_weather=False, passiamo None alle fusion: FiLM e gating si spengono automaticamente
        weather_dim_for_fusion = self.weather_emb_dim if self.use_weather else None

        FusionMono = AdaptiveRadarCameraFusionMono

        # Livello 1 (Più Fine): Stride 8 (Corrisponde a fmap_l2)
        self.fusion1 = FusionMono(
            pc_feat_dim=self.enc_width, img_feat_dim=self.enc_width, out_dim=self.enc_width,
            d_model=128, n_heads=4, n_samples=8, stride=8,
            weather_emb_dim=weather_dim_for_fusion
        )

        # Livello 2: Stride 16 (Corrisponde a fmap_l3)
        self.fusion2 = FusionMono(
            pc_feat_dim=self.enc_width, img_feat_dim=self.enc_width, out_dim=self.enc_width,
            d_model=128, n_heads=4, n_samples=8, stride=16,
            weather_emb_dim=weather_dim_for_fusion
        )

        # Livello 3: Stride 32 (Corrisponde a fmap_l4)
        self.fusion3 = FusionMono(
            pc_feat_dim=self.enc_width, img_feat_dim=self.enc_width, out_dim=self.enc_width,
            d_model=128, n_heads=4, n_samples=8, stride=32,
            weather_emb_dim=weather_dim_for_fusion
        )

        # Livello 4 (Più Coarse): Stride 32 (Corrisponde a fmap_l4, riutilizzo)
        self.fusion4 = FusionMono(
            pc_feat_dim=self.enc_width, img_feat_dim=self.enc_width, out_dim=self.enc_width,
            d_model=128, n_heads=4, n_samples=8, stride=32,
            weather_emb_dim=weather_dim_for_fusion
        )

        # Riferimenti ai moduli di fusione per il forward (ordinati dal Lvl 1 al Lvl 4)
        self._fusions = [self.fusion1, self.fusion2, self.fusion3, self.fusion4]

        # Testa per scalar+embedding del meteo: creata SOLO se use_weather=True
        # Input: [global_pc (C) ; global_img (C)] => 2*C
        if self.use_weather:
            self.env_head = EnvHead(
                in_dim=2 * self.enc_width,
                emb_dim=self.weather_emb_dim,
            )
        else:
            self.env_head = None




        
        # LSTM "probe" solo per stimare h_prev0
        self.lstm_probe = nn.LSTM(
            input_size=self.encoder.out_ch,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        

        # LSTM finale (quella che usata per la predizione)
        self.lstm = nn.LSTM(
            input_size=self.encoder.out_ch,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.hprev_proj = nn.Linear(self.encoder.out_ch, hidden)

        self.head_t = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 3))
        self.head_q = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 4))
        # -----------------------------
        # Heteroscedastic aleatoric uncertainty heads (Kendall & Gal)
        # Predicono log-variance per-sample: s_t(x), s_q(x)
        # -----------------------------
        self.head_logvar_t = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.head_logvar_q = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        

        # cache (per training/monitoring)
        self.last_logvar_t = None
        self.last_logvar_q = None

        self._dbg_cam_feat_count = 0
        self._dbg_cam_vis_done = False
        C = self.encoder.out_ch
        H = self.lstm.hidden_size
        Himu = self.imu_enc_hidden

        self.a2do_fuse = A2DOHierarchicalFilter2Modalities(
            dim_mod1=C,
            dim_mod2=Himu,
            hidden_dim=H,
            out_dim=C,           
            num_heads=4,
            tau_temporal=1.0,
            tau_spatial=1.0,
            hard_gumbel=False,
            attn_dropout=0.0,
        )


    
    def _ensure_imu_encoder(self, D, device):
        D = int(D)
        if self.imu_encoder is None:
            self.imu_in_dim = D
            self.imu_encoder = nn.LSTM(
                input_size=self.imu_in_dim,
                hidden_size=self.imu_enc_hidden,
                num_layers=self.imu_enc_layers,
                batch_first=True,
                dropout=self.imu_dropout,
                bidirectional=False
            ).to(device)
        else:
            # Se esiste già, assicurati che stia sullo stesso device del training
            pdev = next(self.imu_encoder.parameters()).device
            if pdev != device:
                self.imu_encoder = self.imu_encoder.to(device)


    def _encode_imu(self, imu, device):
        """
        imu dict da collate:
        imu["X"]       (B, L, D)  già normalizzato dal dataloader
        imu["lengths"] (B,)
        ritorna feat (B, H)
        """
        X = imu["X"].to(device=device, dtype=torch.float32, non_blocking=True)
        lengths = imu["lengths"].to(device=device, non_blocking=True)

        self._ensure_imu_encoder(X.shape[-1], device)

        Xp = nn.utils.rnn.pack_padded_sequence(
            X, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.imu_encoder(Xp)  # h: (layers, B, H)

        feat = h[-1]  # (B, H)
        return feat

    def _encode_imu_seq(self, imu, device):  #crea sequenza imu da fondere con radar-camera
        X = imu["X"].to(device=device, dtype=torch.float32, non_blocking=True)      # (B,L,Li,D)
        lengths = imu["lengths"].to(device=device, non_blocking=True)              # (B,L)

        B, L, Li, D = X.shape
        self._ensure_imu_encoder(D, device)

        Xf = X.reshape(B*L, Li, D)
        lf = lengths.reshape(B*L)

        # pack richiede almeno 1
        lf = torch.clamp(lf, min=1)

        Xp = nn.utils.rnn.pack_padded_sequence(
            Xf, lf.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.imu_encoder(Xp)   # (layers, B*L, H)
        feat = h[-1]                      # (B*L, H)
        feat = feat.reshape(B, L, -1)     # (B,L,H)
        return feat


    # ---------------- LEGACY (lista di coppie numpy) ----------------
    def _encode_pair_mean(self, xyz_t, feat_t, xyz_t1, feat_t1, device):
        xt  = torch.from_numpy(xyz_t).unsqueeze(0).to(device)
        ft  = torch.from_numpy(feat_t).unsqueeze(0).to(device)
        xt1 = torch.from_numpy(xyz_t1).unsqueeze(0).to(device)
        ft1 = torch.from_numpy(feat_t1).unsqueeze(0).to(device)
        # legacy: solo radar
        E = self.encoder(xt, ft, xt1, ft1)   # (1,N,C)
        v = E.mean(dim=1).squeeze(0)         # (C,)
        return v


    # ---------------- PATH batched con fusione camera ----------------
        # ---------------- PATH batched con fusione camera ----------------
    def _encode_batched_with_cam(self, Xt, Ft, Xt1, Ft1, cam, device):
        """
        Xt,Ft,Xt1,Ft1: tensori (P, N*, ...)
        cam: dict dal dataloader radar4d_cam_pose_dataloader_raw_batched
        """
        # Assicuriamoci che i tensori siano contigui
        Xt   = Xt.contiguous()
        Ft   = Ft.contiguous()
        Xt1  = Xt1.contiguous()
        Ft1  = Ft1.contiguous()

        # 1) Piramidi radar t e t+1 (4 livelli) prima della fusione con la camera
        xyzs_t,  feats_t  = self.encoder.encode_pyramid(Xt,  Ft,  which="t")
        xyzs_t1, feats_t1 = self.encoder.encode_pyramid(Xt1, Ft1, which="t1")
        # xyzs_t  = [xyz1, xyz2, xyz3, xyz4]
        # feats_t = [f1,   f2,   f3,   f4]
        


        # 2) Prepara immagini
        I_t  = cam["I_t"]
        I_t1 = cam["I_t1"]
        if isinstance(I_t, list):
            shapes_t  = [im.shape for im in I_t]
            shapes_t1 = [im.shape for im in I_t1]
            assert all(s == shapes_t[0] for s in shapes_t),  "Immagini I_t con size diverse nel batch"
            assert all(s == shapes_t1[0] for s in shapes_t1), "Immagini I_t1 con size diverse nel batch"
            I_t  = torch.stack(I_t,  dim=0)
            I_t1 = torch.stack(I_t1, dim=0)
        I_t  = I_t.to(device=device, dtype=torch.float32, non_blocking=True)   # (P,3,H,W)
        I_t1 = I_t1.to(device=device, dtype=torch.float32, non_blocking=True)

        K   = cam["K"].to(device=device, dtype=torch.float32)                  # (P,3,3) intrinseci camera
        Tci = cam["T_cam_from_imu"].to(device=device, dtype=torch.float32)     # (P,4,4) trasformazione coordinate per proiettare i punti
        img_size = cam["img_size"]                                            # di solito tensor (P,2) o tuple (H,W)

        # 3) Feature camera multiscala
        fmap_t_dict  = self.img_backbone(I_t)    # {"l2": f2_t, "l3": f3_t, "l4": f4_t}
        fmap_t1_dict = self.img_backbone(I_t1)

        # 3.b) Global pooling per il weather per frame t e t+1
        #    usiamo il livello radar più fine (lvl1) e la fmap camera più fine ("l2")
        f1_t   = feats_t[0]               # (P,N1,C)
        f1_t1  = feats_t1[0]
        global_pc_t  = f1_t.mean(dim=1)                      # (P, C)
        global_pc_t1 = f1_t1.mean(dim=1)                     # (P, C)
        global_img_t  = fmap_t_dict["l2"].mean(dim=(2, 3))   # (P, C)
        global_img_t1 = fmap_t1_dict["l2"].mean(dim=(2, 3))  # (P, C)

        # Concatenazione radar+camera
        g_t  = torch.cat([global_pc_t,  global_img_t],  dim=-1)   # (P, 2C)
        g_t1 = torch.cat([global_pc_t1, global_img_t1], dim=-1)   # (P, 2C)

        # 3.c) Env scalars (meteo + illuminazione) + embedding per i due frame
        # 3.c) Env scalars + embedding (solo se abilitato)
        if self.use_weather:
            sW_t, sL_t, w_t, qcam_t      = self.env_head(g_t)
            sW_t1, sL_t1, w_t1, qcam_t1  = self.env_head(g_t1)

            # q_cam logging / loss
            self.last_qcam_log  = qcam_t.detach().view(-1).cpu()  # (P,)
            self.last_qcam_pred = qcam_t.view(-1)                 # (P,) con gradiente

            # scalari per loss (NO detach)
            self.last_s_weather_pred = sW_t.view(-1)
            self.last_s_illum_pred   = sL_t.view(-1)
            self.last_s_weather_predt1 = sW_t1.view(-1)
            self.last_s_illum_predt1   = sL_t1.view(-1)

            # scalari per logging (detach)
            self.last_s_weather_log  = sW_t.detach().view(-1).cpu()
            self.last_s_illum_log    = sL_t.detach().view(-1).cpu()
            self.last_s_weather_logt1 = sW_t1.detach().view(-1).cpu()
            self.last_s_illum_logt1   = sL_t1.detach().view(-1).cpu()
        else:
            # niente EnvHead, niente FiLM, niente q_cam
            w_t = None
            w_t1 = None
            qcam_t = None
            qcam_t1 = None

            # reset campi usati altrove (evita log vecchi / branch strani)
            self.last_qcam_log = None
            self.last_qcam_pred = None
            self.last_s_weather_pred = None
            self.last_s_illum_pred = None
            self.last_s_weather_predt1 = None
            self.last_s_illum_predt1 = None
            self.last_s_weather_log = None
            self.last_s_illum_log = None
            self.last_s_weather_logt1 = None
            self.last_s_illum_logt1 = None




        # 4) A-RCFM multiscala
        feats_t_fused = []
        feats_t1_fused = []
        
        # Riferimenti alle fmap ordinati per Livello Radar (0=Lvl 1, 3=Lvl 4)
        fmap_refs    = [fmap_t_dict["l2"], fmap_t_dict["l3"], fmap_t_dict["l4"], fmap_t_dict["l4"]]
        fmap_refs_t1 = [fmap_t1_dict["l2"], fmap_t1_dict["l3"], fmap_t1_dict["l4"], fmap_t1_dict["l4"]]
        

        for l_idx in range(4): # 0, 1, 2, 3 (che corrisponde a Livello 1, 2, 3, 4)
            xyz_t, f_t     = xyzs_t[l_idx], feats_t[l_idx]
            xyz_t1, f_t1   = xyzs_t1[l_idx], feats_t1[l_idx]
            fmap_t_ref     = fmap_refs[l_idx]
            fmap_t1_ref    = fmap_refs_t1[l_idx]
            fusion_module  = self._fusions[l_idx] # <-- Chiama self.fusion1, self.fusion2, ...

            # Fusione per t
            f_t_fused = fusion_module(
                xyz_t, f_t, fmap_t_ref, K, Tci, img_size,
                weather_emb=w_t, weather_gate=qcam_t,
            )
            feats_t_fused.append(f_t_fused)

            # Fusione per t+1
            f_t1_fused = fusion_module(
                xyz_t1, f_t1, fmap_t1_ref, K, Tci, img_size,
                weather_emb=w_t1, weather_gate=qcam_t1,
            )
            feats_t1_fused.append(f_t1_fused)

        # 5) Cost volume multi-livello usando le feature fuse
        cvs = self.encoder.cost_pyramid(
            xyzs_t,  feats_t_fused,
            xyzs_t1, feats_t1_fused,
        )
        cv1, cv2, cv3, cv4 = cvs

        # 6) Coarse-to-fine 4 -> 1 (replica della logica nel forward dell'encoder),
        #    ma usando le feature fuse radar+camera invece delle sole feature radar.

        # livello 4 (più coarse)
        x4 = torch.cat([feats_t_fused[3], cv4], dim=-1)      # (P,N4,C+Cv)
        x4 = self.encoder.proj4(as_BCN(x4))
        x4 = as_BNC(x4)         # concatena feature fuse e cost volume                              (P,N4,width) 
        

        # lvl4 -> lvl3
        # lvl4 -> lvl3
        x4_up3 = knn_interpolate_xyz_feat(xyzs_t[3], x4, xyzs_t[2])   # (B, N3, width) upsampling livello 4

        

        x3 = torch.cat([feats_t_fused[2], cv3, x4_up3], dim=-1)
        x3 = self.encoder.proj3(as_BCN(x3))
        x3 = as_BNC(x3)        # concatena feature fuse livello 3, cost volume livello 3, upsample livello 4 poi proiezione

        # lvl3 -> lvl2
        x3_up2 = knn_interpolate_xyz_feat(xyzs_t[2], x3, xyzs_t[1])   # (B, N2, width)
        x2 = torch.cat([feats_t_fused[1], cv2, x3_up2], dim=-1)
        x2 = self.encoder.proj2(as_BCN(x2))
        x2 = as_BNC(x2)

        # lvl2 -> lvl1
        x2_up1 = knn_interpolate_xyz_feat(xyzs_t[1], x2, xyzs_t[0])   # (B, N1, width)
        x1 = torch.cat([feats_t_fused[0], cv1, x2_up1], dim=-1)
        x1 = self.encoder.proj1(as_BCN(x1))
        x1 = as_BNC(x1)


        # 7) Head finale sul livello 1
        E = self.encoder.out_head(as_BCN(x1))               # (P,out_ch,N1)
        E = as_BNC(E)                                       # (P,N1,out_ch)
        return E


    def forward(self, pairs_batch, lengths, cam=None, imu=None, a2do_tau_temporal: float = 1.0, a2do_tau_spatial: float = 1.0, a2do_warmup_keep_prob=None):
        """
        Se cam è None -> solo radar (forward identico a prima).
        Se cam è un dict (come da dataloader cam_batched) -> usa A-RCFM.
        """
        device = next(self.parameters()).device
        # ---- IMU branch (solo encoding, fusione dopo) ----
        self.last_imu_feat = None
        if imu is not None:
            self.last_imu_feat = self._encode_imu_seq(imu, device)  # (B, Himu)
            imu_feat_seq = self.last_imu_feat


        B = int(lengths.size(0))
        Lmax = int(lengths.max().item())
        C = self.encoder.out_ch
        X_pad = torch.zeros(B, Lmax, C, device=device, dtype=torch.float32)

        if isinstance(pairs_batch, list):
            # ======= LEGACY PATH (lista) =======
            for b in range(B):
                pairs = pairs_batch[b]
                for t in range(len(pairs)):
                    xyz_t, feat_t, xyz_t1, feat_t1 = pairs[t]
                    v = self._encode_pair_mean(xyz_t, feat_t, xyz_t1, feat_t1, device)
                    X_pad[b, t] = v
        else:
            # ======= BATCHED PATH (dict) =======
            Xt   = pairs_batch["Xt"].to(device, non_blocking=True)
            Ft   = pairs_batch["Ft"].to(device, non_blocking=True)
            Xt1  = pairs_batch["Xt1"].to(device, non_blocking=True)
            Ft1  = pairs_batch["Ft1"].to(device, non_blocking=True)
            validN_t = pairs_batch["validN_t"].to(device, non_blocking=True)
            pair_b   = pairs_batch["pair_b"].to(device, non_blocking=True)
            pair_t   = pairs_batch["pair_t"].to(device, non_blocking=True)

            # Se ho la camera uso A-RCFM, altrimenti encoder puro radar
            if cam is not None:
                E = self._encode_batched_with_cam(Xt, Ft, Xt1, Ft1, cam, device)
            else:
                E = self.encoder(Xt, Ft, Xt1, Ft1)   # (P, Nmax, C)

            Nmax = E.size(1)
            sum_valid = E.sum(dim=1)                         # (P,C)
            Ni = torch.full((E.size(0),1), float(Nmax), device=device, dtype=E.dtype)
            V = sum_valid / Ni #un vettore per coppia
                                            # (P,C)

            for i in range(V.size(0)):  #ricostruire sequenza temporale per batch
                b = int(pair_b[i].item()); t = int(pair_t[i].item())
                X_pad[b, t] = V[i]

         # ---- prima della LSTM: costruisci mask padding subito ----
        key_padding_mask = torch.arange(Lmax, device=lengths.device)[None, :] >= lengths[:, None]  # [B,L] serve per fare ignorare i pad ad A2DO

        # ---- 1) PASS PROVVISORIO: ottieni h_prev dal decoder "grezzo" ----
        packed0 = torch.nn.utils.rnn.pack_padded_sequence(X_pad, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out0, _ = self.lstm_probe(packed0)
        out0, _ = torch.nn.utils.rnn.pad_packed_sequence(out0, batch_first=True)   # h_seq0: [B,L,H]

        h_prev0 = torch.zeros_like(out0)  #serve per stato precedente A2DO
        h_prev0[:, 1:, :] = out0[:, :-1, :]

        # maschera per sicurezza (zero sui pad)
        out0   = out0.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        h_prev0 = h_prev0.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        # ---- 2) A2DO: fonde radar+cam (X_pad) con imu_feat_seq ----
        # imu_feat_seq deve essere [B,L,Himu]
        a2do_out = self.a2do_fuse(
            x_mod1=X_pad,
            x_mod2=imu_feat_seq,
            h_prev=h_prev0,
            key_padding_mask=key_padding_mask,
            tau_temporal=a2do_tau_temporal,
            tau_spatial=a2do_tau_spatial,
            warmup_keep_prob=a2do_warmup_keep_prob,
        )
        X_fused = a2do_out["y"]   # [B,L,C] perché out_dim=C :contentReference[oaicite:4]{index=4}
        # ---- NOVELTY: keep A2DO outputs (policy etc.) for decision-process regularization ----
        self.last_a2do_out = a2do_out
        # ---- 3) PASS FINALE: LSTM  su feature fuse ----
        packed = torch.nn.utils.rnn.pack_padded_sequence(X_fused, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        h_seq = out
        h_prev = torch.zeros_like(h_seq)
        h_prev[:, 1:, :] = h_seq[:, :-1, :]

        h_seq  = h_seq.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        h_prev = h_prev.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        last_idx = (lengths - 1).clamp(min=0).view(-1,1,1).expand(-1,1,h_seq.size(2))
        last = h_seq.gather(1, last_idx).squeeze(1)

        pred_t = self.head_t(last)  # predizione posa
        pred_q = self.head_q(last)

        # --- predicted log-variances (shape: [B]) ---
        logvar_t = self.head_logvar_t(last).squeeze(-1)
        logvar_q = self.head_logvar_q(last).squeeze(-1)  

        # clamp per stabilità numerica (evita exp troppo grandi/piccoli)
        # clamp per stabilità numerica (evita exp troppo grandi/piccoli)
        
        # logvar = log(sigma^2)
        logvar_t = torch.clamp(logvar_t, min=-12.0, max=3)   
        logvar_q = torch.clamp(logvar_q, min=-12.0, max=3)  


        self.last_logvar_t = logvar_t
        self.last_logvar_q = logvar_q

        return pred_q, pred_t, h_prev, key_padding_mask



# ============================
# Metriche/plot/EVAL 
# ============================
def quat_normalize_np(q): #normalizza quaternione
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0,0,0,1], dtype=np.float64)
    return q / n

def quat_mul_np(q1, q2): #moltiplica quaternioni
    w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
    w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    nq = np.array([x,y,z,w], dtype=np.float64)
    return nq / (np.linalg.norm(nq) + 1e-12)

def quat_to_mat_np(q): #converte quaternione in matrice di rotazione
    x,y,z,w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=np.float64)

def compose_trajectory_quat_se3patch(dposes_local):
    T = np.zeros((len(dposes_local)+1, 3), dtype=np.float64)
    Q = np.zeros((len(dposes_local)+1, 4), dtype=np.float64)
    Q[0] = np.array([0.0,0.0,0.0,1.0], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    for i, dp in enumerate(dposes_local):
        dt_loc = dp[:3].astype(np.float64)
        dq = dp[3:].astype(np.float64)
        if np.dot(dq, Q[i]) < 0.0:
            dq = -dq
        dq = quat_normalize_np(dq)
        T[i+1] = T[i] + (R @ dt_loc)
        Q[i+1] = quat_mul_np(Q[i], dq)
        R = quat_to_mat_np(Q[i+1])
    traj = np.concatenate([T, Q], axis=1)
    return traj

def ate3d_noalign(xyz_pred, xyz_gt):
    n = min(len(xyz_pred), len(xyz_gt))
    e = xyz_pred[:n] - xyz_gt[:n]
    return np.sqrt((e*e).sum(axis=1).mean())

def ate2d_noalign(xyz_pred, xyz_gt):
    n = min(len(xyz_pred), len(xyz_gt))
    e = xyz_pred[:n,:2] - xyz_gt[:n,:2]
    return np.sqrt((e*e).sum(axis=1).mean())

def umeyama_alignment(xyz_src, xyz_dst, with_scale=True):
    """
    Stima la trasformazione di similarità (Sim(3)) che allinea xyz_src -> xyz_dst
    usando l'algoritmo di Umeyama.
    xyz_src, xyz_dst: array (N,3)
    """
    xyz_src = np.asarray(xyz_src, dtype=np.float64)
    xyz_dst = np.asarray(xyz_dst, dtype=np.float64)
    n = xyz_src.shape[0]
    assert xyz_dst.shape[0] == n and xyz_src.shape[1] == 3 and xyz_dst.shape[1] == 3

    mu_src = xyz_src.mean(axis=0)
    mu_dst = xyz_dst.mean(axis=0)
    X = xyz_src - mu_src
    Y = xyz_dst - mu_dst

    # covarianza (target * source^T)
    Sigma = (Y.T @ X) / float(n)

    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt
    if with_scale:
        var_src = (X * X).sum() / float(n)
        s = (D * np.diag(S)).sum() / float(var_src)
    else:
        s = 1.0

    t = mu_dst - s * (R @ mu_src)
    return s, R, t


def ate3d_aligned_sim3(xyz_pred, xyz_gt, with_scale=True):
    """
    ATE 3D allineato Sim(3) (Umeyama), come in KITTI/TUM:
      - stima Sim(3) che porta pred -> gt
      - applica l'allineamento
      - calcola RMSE in 3D
    """
    n = min(len(xyz_pred), len(xyz_gt))
    if n < 2:
        return float("nan")
    X = np.asarray(xyz_pred[:n], dtype=np.float64)
    Y = np.asarray(xyz_gt[:n], dtype=np.float64)

    s, R, t = umeyama_alignment(X, Y, with_scale=with_scale)
    X_aligned = (s * (R @ X.T)).T + t  # (N,3)
    e = X_aligned - Y
    rmse = np.sqrt((e * e).sum(axis=1).mean())
    return float(rmse)


def yaw_from_quat(qw, qx, qy, qz):
    siny_cosp = 2*(qw*qz + qx*qy)
    cosy_cosp = 1 - 2*(qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)

def kitti_drift(xyz_pred, yaw_pred, xyz_gt, yaw_gt, seg=100.0):
    dT_list, dRdeg_list = [], []
    if len(xyz_pred) < 2 or len(xyz_gt) < 2: return dT_list, dRdeg_list
    s = np.zeros(len(xyz_gt), dtype=np.float64)
    for i in range(1,len(xyz_gt)):
        s[i] = s[i-1] + np.linalg.norm(xyz_gt[i]-xyz_gt[i-1])
    L = s[-1]
    if L < seg: return dT_list, dRdeg_list
    idx = 0
    while True:
        start_s = idx*seg
        end_s   = (idx+1)*seg
        if end_s > L: break
        i0 = np.searchsorted(s, start_s)
        i1 = np.searchsorted(s, end_s)
        dT = np.linalg.norm((xyz_pred[i1]-xyz_pred[i0]) - (xyz_gt[i1]-xyz_gt[i0]))
        dYaw = (yaw_pred[i1]-yaw_pred[i0]) - (yaw_gt[i1]-yaw_gt[i0])
        dT_list.append(dT); dRdeg_list.append(abs(dYaw)*180.0/math.pi)
        idx += 1
    return dT_list, dRdeg_list

def whole_sequence_drift(xyz_pred, yaw_pred, xyz_gt, yaw_gt):
    """
    Compute translation and rotation drift over the ENTIRE trajectory
    (not fixed-length segments).

    Returns:
      trans_drift_pct       : float  –  (||pos_error|| / path_length) * 100  [%]
      rot_drift_deg_per_m   : float  –  |yaw_error| / path_length            [deg/m]
      path_length           : float  –  total GT path length [m]
    """
    N = len(xyz_gt)
    if N < 2:
        return None, None, 0.0

    # Path length (ground truth)
    path_length = 0.0
    for i in range(1, N):
        path_length += float(np.linalg.norm(xyz_gt[i] - xyz_gt[i - 1]))
    if path_length < 1e-6:
        return None, None, 0.0

    # translation drift ---
    # Compare incremental motions (i-1 -> i) between pred and gt,
    
    trans_err_sum = 0.0
    for i in range(1, N):
        d_pred = xyz_pred[i] - xyz_pred[i - 1]
        d_gt   = xyz_gt[i]   - xyz_gt[i - 1]
        trans_err_sum += float(np.linalg.norm(d_pred - d_gt))

    # Normalize by total GT path length 
    trans_drift_pct = trans_err_sum / path_length * 100 

    # Rotation drift
    rot_err_sum = 0.0
    for i in range(1, N):
        dyaw_pred = yaw_pred[i] - yaw_pred[i - 1]
        dyaw_gt   = yaw_gt[i]   - yaw_gt[i - 1]

        # wrap angle difference to [-pi, pi]
        d_err = dyaw_pred - dyaw_gt
        d_err = (d_err + math.pi) % (2 * math.pi) - math.pi

        rot_err_sum += abs(d_err)

    # convert to degrees
    rot_err_deg = rot_err_sum * 180.0 / math.pi

    # normalize by total path length
    rot_drift_deg_per_m = rot_err_deg / path_length

    return trans_drift_pct, rot_drift_deg_per_m, path_length

# ============================
# RPE stile VoD / KITTI (segmenti 20..160 m, m/m e deg/m)
# ============================

def traj_to_poses4x4(traj):
    """
    traj: array (N,7) con [tx,ty,tz,qx,qy,qz,qw]
    ritorna: array (N,4,4) di matrici SE(3)
    """
    N = traj.shape[0]
    poses = np.zeros((N, 4, 4), dtype=np.float64)
    poses[:] = np.eye(4, dtype=np.float64)[None, :, :]
    for i in range(N):
        tx, ty, tz, qx, qy, qz, qw = traj[i]
        R = quat_to_mat_np(np.array([qx, qy, qz, qw], dtype=np.float64))
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
        poses[i] = T
    return poses


def compute_rpe_vod_style(
    traj_pred,
    traj_gt,
    seg_lengths=(20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0),
):
    """
    RPE stile KITTI/VoD (comparabile ai paper):
      - trans: RMSE( ||t_err|| / L )  [m/m]
      - rot:   RMSE( ang_err_deg / L ) [deg/m]
    Calcolato su tutti i segmenti (i0 -> i1) tali che la distanza lungo GT sia ~L.
    """
    assert traj_pred.shape == traj_gt.shape, "traj_pred e traj_gt devono avere stessa shape"
    N = traj_pred.shape[0]
    if N < 2:
        return {
            "per_length": {},
            "overall_mean_trans_rmse_m_per_m": None,
            "overall_mean_rot_rmse_deg_per_m": None,
        }

    poses_pred = traj_to_poses4x4(traj_pred)
    poses_gt   = traj_to_poses4x4(traj_gt)

    # lunghezza cumulativa lungo GT
    xyz_gt = traj_gt[:, :3]
    s = np.zeros(N, dtype=np.float64)
    for i in range(1, N):
        s[i] = s[i-1] + np.linalg.norm(xyz_gt[i] - xyz_gt[i-1])
    Ltot = float(s[-1])

    results_per_L = {}
    all_t_norm = []
    all_r_norm = []

    for seg_len in seg_lengths:
        seg_len = float(seg_len)
        if Ltot < seg_len:
            continue

        t_norm_list = []  # m/m
        r_norm_list = []  # deg/m

        for i0 in range(0, N-1):
            end_s = s[i0] + seg_len
            i1 = np.searchsorted(s, end_s)
            if i1 >= N:
                break

            T_rel_gt   = np.linalg.inv(poses_gt[i0])   @ poses_gt[i1]
            T_rel_pred = np.linalg.inv(poses_pred[i0]) @ poses_pred[i1]

            T_err = np.linalg.inv(T_rel_pred) @ T_rel_gt
            t_err = T_err[:3, 3]
            R_err = T_err[:3, :3]

            # angolo di rotazione da matrice
            tr = np.trace(R_err)
            tr = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
            ang_rad = math.acos(tr)

            trans_error_m = float(np.linalg.norm(t_err))              # [m]
            rot_error_deg = float(ang_rad * 180.0 / math.pi)          # [deg]

            # normalizza PER-SINGOLO segmento (questa è la chiave!)
            t_norm_list.append(trans_error_m / seg_len)               # [m/m]
            r_norm_list.append(rot_error_deg / seg_len)               # [deg/m]

        if len(t_norm_list) == 0:
            continue

        t_norm = np.asarray(t_norm_list, dtype=np.float64)
        r_norm = np.asarray(r_norm_list, dtype=np.float64)

        trans_rmse = float(np.sqrt(np.mean(t_norm * t_norm)))         # [m/m]
        rot_rmse   = float(np.sqrt(np.mean(r_norm * r_norm)))         # [deg/m]

        results_per_L[str(int(seg_len))] = {
            "num_segments": int(len(t_norm)),
            "trans_rmse_m_per_m": trans_rmse,
            "rot_rmse_deg_per_m": rot_rmse,
        }

        all_t_norm.append(t_norm)
        all_r_norm.append(r_norm)

    if len(all_t_norm) == 0:
        overall_t = None
        overall_r = None
    else:
        all_t_norm = np.concatenate(all_t_norm, axis=0)
        all_r_norm = np.concatenate(all_r_norm, axis=0)
        overall_t = float(np.sqrt(np.mean(all_t_norm * all_t_norm)))
        overall_r = float(np.sqrt(np.mean(all_r_norm * all_r_norm)))

    return {
        "per_length": results_per_L,
        "overall_mean_trans_rmse_m_per_m": overall_t,
        "overall_mean_rot_rmse_deg_per_m": overall_r,
    }



def quaternion_angular_error_rad(q1, q2):
    q1 = F.normalize(q1, p=2, dim=-1)
    q2 = F.normalize(q2, p=2, dim=-1)
    dot = torch.sum(q1*q2, dim=-1).abs().clamp(0.0, 1.0-1e-7)
    return 2.0 * torch.acos(dot)

def enforce_quat_continuity(q):
    out = [q[0:1]]
    for i in range(1, q.size(0)):
        qi = q[i:i+1]
        prev = out[-1]
        if torch.sum(qi*prev) < 0:
            qi = -qi
        out.append(qi)
    return torch.cat(out, dim=0)
def policy_regularizers(pi, key_padding_mask=None, entropy_weight=0.0, switch_weight=0.0):
    """
    pi: [B,L,K] probability distribution (policy), K=2 for {drop, keep}
    key_padding_mask: [B,L] True on PAD positions
    Returns: (loss, dict_debug)
    """
    B, L, K = pi.shape
    device = pi.device
    dtype  = pi.dtype

    # valid mask: 1 on real timesteps, 0 on pad
    valid = torch.ones((B, L), device=device, dtype=dtype)
    if key_padding_mask is not None:
        valid = (1.0 - key_padding_mask.to(dtype))  # True pad -> 0

    # entropy H(pi)
    ent = -(pi * (pi.clamp_min(1e-8).log())).sum(dim=-1)  # [B,L]
    ent = (ent * valid).sum() / (valid.sum() + 1e-8)
    loss_ent = -ent  # minimize(-H) => maximize H

    # switching cost: ||pi_t - pi_{t-1}||_1
    dpi = (pi[:, 1:, :] - pi[:, :-1, :]).abs().sum(dim=-1)  # [B,L-1]
    valid_pair = valid[:, 1:] * valid[:, :-1]
    loss_switch = (dpi * valid_pair).sum() / (valid_pair.sum() + 1e-8)

    loss = entropy_weight * loss_ent + switch_weight * loss_switch
    dbg = {"entropy": ent.detach(), "switch": loss_switch.detach()}
    return loss, dbg

# ============================
# TRAIN 
# ============================
def train_and_validate(train_loader, val_loader, out_dir, seqs_cfg_cam, wq_init=-2.5, wt_init=0.0, device='cuda', no_weather: bool = False, use_amp: bool = False, resume_path: str = ""):
    global global_step
    model = Radar4DEncLSTM(use_weather=(not no_weather))
    device = torch.device(device if torch.cuda.is_available() and device!='cpu' else 'cpu')
    crit = AdaptiveLoss(
    wq_init=wq_init, wt_init=wt_init,
    wq_min=-10.0, wq_max=2.0,
    wt_min=-4.0,  wt_max=1.5,
    huber_delta_deg=1.0
    ).to(device)
    model.to(device); crit.to(device)

    opt = torch.optim.Adam([*model.parameters(), *crit.parameters()],
                        lr=LR, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)



    os.makedirs(out_dir, exist_ok=True)
    best = 1e9
    train_loss_hist, val_loss_hist = [], []
    train_rmse_t_hist, val_rmse_t_hist = [], []
    train_rmse_rdeg_hist, val_rmse_rdeg_hist = [], []
    train_loss_sW_hist = []
    train_loss_sL_hist = []
    val_loss_sW_hist = []
    val_loss_sL_hist = []
    
    printed_dbg_train = False
    printed_dbg_val   = False

    # RESUME TRAIN (solo se richiesto)
    # ============================
    start_ep = 1
    if resume_path:
        assert os.path.isfile(resume_path), f"resume_path non esiste: {resume_path}"
        print(f"[RESUME] Carico checkpoint train da: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        sd = ckpt["state_dict"]

        # IMPORTANTISSIMO (come fai già in eval_camimu):
        # se nel checkpoint ci sono pesi imu_encoder.*, istanzia imu_encoder prima del load_state_dict
        if "imu_encoder.weight_ih_l0" in sd:
            D = sd["imu_encoder.weight_ih_l0"].shape[1]  # input dim IMU dal checkpoint
            model._ensure_imu_encoder(D, device=torch.device("cpu"))

        model.load_state_dict(sd, strict=True)

        # ripristina i pesi della AdaptiveLoss (checkpoint attuale li salva così)
        with torch.no_grad():
            crit.w_q.copy_(torch.tensor(float(ckpt.get("loss_w_q", crit.w_q.item())), dtype=crit.w_q.dtype))
            crit.w_t.copy_(torch.tensor(float(ckpt.get("loss_w_t", crit.w_t.item())), dtype=crit.w_t.dtype))

         # --- 3) ripristina optimizer / scaler / global_step (NUOVO) ---
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if use_amp and ("scaler" in ckpt):
            scaler.load_state_dict(ckpt["scaler"])
        global_step = int(ckpt.get("global_step", 0))

        # epoca da cui ripartire
        start_ep = int(ckpt.get("epoch", 0)) + 1

        print(f"[RESUME] Riparto da epoch={start_ep}. (optimizer e AMP scaler ripartono fresh)")

    
    # Uncertainty lambda schedule (SEPARATE WEIGHTS!)
    # ============================
    # CRITICAL: Translation and rotation need DIFFERENT weights due to scale mismatch
    LAMBDA_TRANS_MAX = 0.01        # Weight for translation uncertainty
    LAMBDA_ROT_MAX = 0.01           # Weight for rotation uncertainty 
    
    UNC_DELAY_EPOCHS = 0.0         # No delay, start warmup immediately
    UNC_WARMUP_EPOCHS = 2.0        # Warmup over 2 epochs (recommended)

    steps_per_epoch = max(1, len(train_loader))
    UNC_DELAY_STEPS = int(UNC_DELAY_EPOCHS * steps_per_epoch)
    UNC_WARMUP_STEPS = max(1, int(UNC_WARMUP_EPOCHS * steps_per_epoch))

    def lambda_trans_unc(step: int) -> float:
        """Linear warmup for translation: 0 -> LAMBDA_TRANS_MAX."""
        if step < UNC_DELAY_STEPS:
            return 0.0
        x = (step - UNC_DELAY_STEPS) / float(UNC_WARMUP_STEPS)
        x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
        # Start from 10% and ramp up
        return (0.1 + 0.9 * x) * LAMBDA_TRANS_MAX
    
    def lambda_rot_unc(step: int) -> float:
        """Linear warmup for rotation: 0 -> LAMBDA_ROT_MAX."""
        if step < UNC_DELAY_STEPS:
            return 0.0
        x = (step - UNC_DELAY_STEPS) / float(UNC_WARMUP_STEPS)
        x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
        # Start from 10% and ramp up
        return (0.1 + 0.9 * x) * LAMBDA_ROT_MAX


    def _sid_to_seqnames(sid):
        """
        sid: tipicamente Tensor (B,) con indici [0..len(seqs_cfg_cam)-1]
            oppure lista di stringhe già pronte.
        ritorna: list[str] length B
        """
        if isinstance(sid, torch.Tensor):
            sid_list = sid.detach().cpu().tolist()
        else:
            sid_list = list(sid)

        # Se sono già stringhe tipo "01", ok
        if len(sid_list) > 0 and isinstance(sid_list[0], str):
            return sid_list

        # Altrimenti sono ID -> mappa su seqs_cfg_cam[idx].name
        return [str(seqs_cfg_cam[int(i)].name) for i in sid_list]


    for ep in range(start_ep, EPOCHS+1):

        # ===== compute tau + warmup =====
        if ep <= A2DO_WARMUP_EPOCHS:
            # durante warmup: decisione morbida e costante
            a2do_tau = A2DO_TAU0
            a2do_warmup_keep_prob = A2DO_WARMUP_KEEP_PROB
        else:
            # dopo warmup: annealing di tau
            # decay parte da ep=1, ma qui lo facciamo partire dopo warmup
            k = ep - A2DO_WARMUP_EPOCHS
            a2do_tau = max(A2DO_TAU_MIN, A2DO_TAU0 * (A2DO_TAU_DECAY ** (k - 1)))
            a2do_warmup_keep_prob = None

        # training: sempre soft
        _set_a2do_hard(model, A2DO_HARD_TRAIN)

        model.train(); tr_sum=0.0; ntr=0
        tr_sum_sW = 0.0
        tr_sum_sL = 0.0
        ntr_env = 0   # denominatore per aux-loss (conteggia P)
        tr_sum_sq_t=0.0; tr_sum_sq_r=0.0; tr_cnt=0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {ep}/{EPOCHS} [train]", ncols=100)):
            # batch può essere:
            #  - (pairs_batch, y, sid, lengths)          -> solo radar
            #  - (pairs_batch, cam, y, sid, lengths)     -> radar + camera
            # batch può essere:
            #  - (pairs_batch, y, sid, lengths)                -> solo radar
            #  - (pairs_batch, cam, y, sid, lengths)           -> radar + camera
            #  - (pairs_batch, imu, y, sid, lengths)           -> radar + imu (no cam)
            #  - (pairs_batch, cam, imu, y, sid, lengths)      -> radar + cam + imu
            if step == 0:
                print("DEBUG TRAIN BATCH LEN =", len(batch))
                if len(batch) == 6:
                    print("OK: batch = (radar, cam, imu, y, sid, lengths)")
                else:
                    print("WARNING: batch NON contiene IMU")

            if len(batch) == 4:
                pairs_batch, y, sid, lengths = batch
                cam = None
                imu = None
            else:
                # camera path: ora deve includere imu
                pairs_batch, cam, imu, y, sid, lengths = batch

            y = y.to(device); lengths = lengths.to(device)
            gt_t, gt_q = y[:, :3], y[:, 3:7]



            if not printed_dbg_train:
                with torch.no_grad():
                    gt_t_norm = torch.norm(gt_t, dim=1)
                    frac_zero_t = (gt_t_norm < 1e-6).float().mean().item()
                    qI = torch.tensor([0,0,0,1.0], device=gt_q.device, dtype=gt_q.dtype).view(1,4)
                    dot_q = torch.sum(F.normalize(gt_q, p=2, dim=-1) * qI, dim=-1).abs()
                    frac_q_identity = (dot_q > 0.999999).float().mean().item()
                print(f"[DBG train ep{ep}] mean|gt_t|={gt_t_norm.mean().item():.3e}  "
                    f"frac(|gt_t|≈0)={frac_zero_t:.3f}  frac(q≈I)={frac_q_identity:.3f}")
                printed_dbg_train = True

            
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_q, pred_t, h_prev, key_padding_mask = model(pairs_batch, lengths, cam=cam, imu=imu)
                loss = crit(pred_q, pred_t, gt_q, gt_t)

                # ---------- ADDITIVE UNCERTAINTY LOSS (Kendall & Gal) ----------
                # L_tot = L_base + lambda_unc * mean_i [exp(-s_q)*l_q + s_q + exp(-s_t)*l_t + s_t]
                # ---------- ADDITIVE UNCERTAINTY LOSS (Kendall & Gal) ----------
                # L_tot = L_base + lambda_unc * mean_i [exp(-s_q)*l_q + s_q + exp(-s_t)*l_t + s_t]
                # ---------- ADDITIVE UNCERTAINTY LOSS (SEPARATE WEIGHTS!) ----------
                # L_tot = L_base + lambda_t * mean_i[exp(-s_t)*l_t + s_t] + lambda_q * mean_i[exp(-s_q)*l_q + s_q]
                LAMBDA_T = lambda_trans_unc(global_step)
                LAMBDA_Q = lambda_rot_unc(global_step)

                if (hasattr(model, "last_logvar_q") and (model.last_logvar_q is not None) and
                    hasattr(model, "last_logvar_t") and (model.last_logvar_t is not None)):

                    l_q_i, l_t_i = per_sample_pose_losses(
                        pred_q, pred_t, gt_q, gt_t,
                        huber_delta_rad=crit.delta
                    )  # [B], [B]

                    s_q = model.last_logvar_q
                    s_t = model.last_logvar_t

                    # Compute uncertainty losses SEPARATELY
                    loss_unc_t = (torch.exp(-s_t) * l_t_i + s_t).mean()
                    loss_unc_q = (torch.exp(-s_q) * l_q_i + s_q).mean()
                    
                    # Apply SEPARATE weights
                    loss = loss + LAMBDA_T * loss_unc_t + LAMBDA_Q * loss_unc_q


                # ---------- WEATHER LOSS (solo se disponibile) ----------
                if (cam is not None) and (model.use_weather):
                    sW_gt = cam["weather_sev"].to(device=device, dtype=torch.float32)
                    sL_gt = cam["illum_sev"].to(device=device, dtype=torch.float32)

                    sW_pred = model.last_s_weather_pred.to(device=device, dtype=torch.float32)
                    sL_pred = model.last_s_illum_pred.to(device=device, dtype=torch.float32)

                    loss_sW = F.smooth_l1_loss(sW_pred, sW_gt)
                    loss_sL = F.smooth_l1_loss(sL_pred, sL_gt)

                    # logging 
                    seq_names_batch = _sid_to_seqnames(sid)
                    logger.log_pairs(
                        split="train",
                        epoch=ep,
                        step=global_step,
                        s_weather_pred=model.last_s_weather_log,
                        s_illum_pred=model.last_s_illum_log,
                        q_cam_pred=model.last_qcam_log,
                        s_weather_gt=cam["weather_sev"].detach().cpu(),
                        s_illum_gt=cam["illum_sev"].detach().cpu(),
                        pair_b=pairs_batch["pair_b"].detach().cpu(),
                        pair_t=pairs_batch["pair_t"].detach().cpu(),
                        seq_names=seq_names_batch,
                    )

                    loss = loss + 0.2 * loss_sW + 0.2 * loss_sL

                    # aggiorna statistiche env SOLO dentro if
                    P = int(cam["weather_sev"].shape[0])
                    tr_sum_sW += float(loss_sW.item()) * P
                    tr_sum_sL += float(loss_sL.item()) * P
                    ntr_env += P

                # ---------- NOVELTY LOSS (SEMPRE) ----------
                LAMBDA_ENT = 1e-4
                LAMBDA_SW  = 1e-3

                loss_dec = 0.0
                if hasattr(model, "last_a2do_out") and (model.last_a2do_out is not None):
                    a2 = model.last_a2do_out
                    if ("pi_modalities" in a2) and (a2["pi_modalities"] is not None):
                        for pi in a2["pi_modalities"]:  # each [B,L,2]
                            reg, _dbg = policy_regularizers(
                                pi,
                                key_padding_mask=key_padding_mask,
                                entropy_weight=LAMBDA_ENT,
                                switch_weight=LAMBDA_SW
                            )
                            loss_dec = loss_dec + reg

                loss = loss + loss_dec

            # backward + step in AMP
            scaler.scale(loss).backward()

            # clip corretto con AMP
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            scaler.step(opt)
            scaler.update()

            # === LOG DELLO SCALARE s_t PER QUESTO BATCH (TRAIN) ===
            global_step += 1
            

            tr_sum += float(loss.item())*y.size(0); ntr += y.size(0) #accumulando statistiche
            t_diff = pred_t - gt_t
            tr_sum_sq_t += float(torch.sum(torch.sum(t_diff*t_diff, dim=1)).item())
            ang = quaternion_angular_error_rad(pred_q, gt_q)
            tr_sum_sq_r += float(torch.sum(ang*ang).item()); tr_cnt += y.size(0)

        tr_loss = tr_sum/max(1,ntr) 
        tr_loss_sW = tr_sum_sW / max(1, ntr_env)
        tr_loss_sL = tr_sum_sL / max(1, ntr_env)
        train_loss_sW_hist.append(tr_loss_sW)
        train_loss_sL_hist.append(tr_loss_sL)
        train_loss_hist.append(tr_loss)
        train_rmse_t_hist.append(math.sqrt(tr_sum_sq_t/max(1,tr_cnt)))
        train_rmse_rdeg_hist.append(math.sqrt(tr_sum_sq_r/max(1,tr_cnt))*(180.0/math.pi))

        model.eval(); va_sum=0.0; nva=0
        va_sum_sW = 0.0
        va_sum_sL = 0.0
        nva_env = 0
        va_sum_sq_t=0.0; va_sum_sq_r=0.0; va_cnt=0
        # eval
        _set_a2do_hard(model, A2DO_HARD_EVAL)

        # in eval tenere lo stesso tau dell'epoca
        a2do_tau_eval = max(A2DO_TAU_MIN, a2do_tau)
        # =======================
        # Minimal uncertainty logging (VAL)
        # =======================
        UNC_SAVE_MAX = 2000          # max punti salvati per epoca (poco overhead)
        UNC_POINTS_PER_BATCH = 32    # quanti sample estrarre dal batch (random)
        UNC_BATCH_STRIDE = 5         # logga 1 batch ogni N (riduce overhead)

        unc_logvar_t = []
        unc_logvar_q = []
        unc_trans_err = []
        unc_rot_err_deg = []


        with torch.no_grad():
            for batch_idx, batch in enumerate(
                    tqdm(val_loader, desc=f"Epoch {ep}/{EPOCHS} [val]", ncols=100, leave=False)
                ):
                # batch può essere:
                #  - (pairs_batch, y, sid, lengths)                -> solo radar
                #  - (pairs_batch, cam, y, sid, lengths)           -> radar + camera
                #  - (pairs_batch, imu, y, sid, lengths)           -> radar + imu (no cam)
                #  - (pairs_batch, cam, imu, y, sid, lengths)      -> radar + cam + imu
                if len(batch) == 4:
                    pairs_batch, y, sid, lengths = batch
                    cam = None
                    imu = None
                else:
                    # camera path: ora deve includere imu
                    pairs_batch, cam, imu, y, sid, lengths = batch

                y = y.to(device); lengths = lengths.to(device)
                gt_t, gt_q = y[:, :3], y[:, 3:7]


                # ===== DEBUG LABELS (VAL) =====
                if not printed_dbg_val:
                    gt_t_norm = torch.norm(gt_t, dim=1)
                    frac_zero_t = (gt_t_norm < 1e-6).float().mean().item()
                    qI = torch.tensor([0,0,0,1.0], device=gt_q.device, dtype=gt_q.dtype).view(1,4)
                    dot_q = torch.sum(F.normalize(gt_q, p=2, dim=-1) * qI, dim=-1).abs()
                    frac_q_identity = (dot_q > 0.999999).float().mean().item()
                    print(f"[DBG val   ep{ep}] mean|gt_t|={gt_t_norm.mean().item():.3e}  "
                        f"frac(|gt_t|≈0)={frac_zero_t:.3f}  frac(q≈I)={frac_q_identity:.3f}")
                    printed_dbg_val = True

                pred_q, pred_t, h_prev, key_padding_mask = model(pairs_batch, lengths, cam=cam, imu=imu, a2do_tau_temporal=a2do_tau_eval, a2do_tau_spatial=a2do_tau_eval, a2do_warmup_keep_prob= None)  # <-- QUI
                # =======================
                # Collect minimal uncertainty-vs-error pairs (VAL)
                # =======================
                if (batch_idx % UNC_BATCH_STRIDE == 0) and (len(unc_logvar_t) < UNC_SAVE_MAX):
                    mref = model.module if hasattr(model, "module") else model

                    if (hasattr(mref, "last_logvar_t") and (mref.last_logvar_t is not None) and
                        hasattr(mref, "last_logvar_q") and (mref.last_logvar_q is not None)):

                        # predicted log-variances (detach, no grad)
                        s_t = mref.last_logvar_t.detach()
                        s_q = mref.last_logvar_q.detach()

                        # true errors (interpretabili)
                        trans_err = torch.norm((pred_t - gt_t).detach(), dim=1)  # [B] in metri
                        rot_err_deg = (quaternion_angular_error_rad(pred_q.detach(), gt_q) * (180.0 / math.pi))  # [B] in gradi

                        B = int(s_t.shape[0])
                        remaining = UNC_SAVE_MAX - len(unc_logvar_t)
                        k = min(UNC_POINTS_PER_BATCH, remaining, B)

                        # sample random indices (pochi punti -> poco overhead)
                        idx = torch.randperm(B, device=s_t.device)[:k]

                        unc_logvar_t.extend(s_t[idx].cpu().numpy().tolist())
                        unc_logvar_q.extend(s_q[idx].cpu().numpy().tolist())
                        unc_trans_err.extend(trans_err[idx].cpu().numpy().tolist())
                        unc_rot_err_deg.extend(rot_err_deg[idx].cpu().numpy().tolist())

                loss = crit(pred_q, pred_t, gt_q, gt_t)

                # ---------- ADDITIVE UNCERTAINTY LOSS (VAL) ----------
                loss = crit(pred_q, pred_t, gt_q, gt_t)

                # ---------- ADDITIVE UNCERTAINTY LOSS (VAL - SEPARATE WEIGHTS!) ----------
                LAMBDA_T = lambda_trans_unc(global_step)
                LAMBDA_Q = lambda_rot_unc(global_step)

                if (hasattr(model, "last_logvar_q") and (model.last_logvar_q is not None) and
                    hasattr(model, "last_logvar_t") and (model.last_logvar_t is not None)):

                    l_q_i, l_t_i = per_sample_pose_losses(
                        pred_q, pred_t, gt_q, gt_t,
                        huber_delta_rad=crit.delta
                    )

                    s_q = model.last_logvar_q
                    s_t = model.last_logvar_t

                    # Compute uncertainty losses SEPARATELY
                    loss_unc_t = (torch.exp(-s_t) * l_t_i + s_t).mean()
                    loss_unc_q = (torch.exp(-s_q) * l_q_i + s_q).mean()
                    
                    # Apply SEPARATE weights
                    loss = loss + LAMBDA_T * loss_unc_t + LAMBDA_Q * loss_unc_q

                if (cam is not None) and (model.use_weather):
                    sW_gt = cam["weather_sev"].to(device=device, dtype=torch.float32)  # (P,)
                    sL_gt = cam["illum_sev"].to(device=device, dtype=torch.float32)    # (P,)

                    sW_pred = model.last_s_weather_pred.to(device=device, dtype=torch.float32)  # (P,)
                    sL_pred = model.last_s_illum_pred.to(device=device, dtype=torch.float32)    # (P,)

                    loss_sW = F.smooth_l1_loss(sW_pred, sW_gt)
                    loss_sL = F.smooth_l1_loss(sL_pred, sL_gt)
                    # --- logging scalari ambiente (VAL) ---
                    seq_names_batch = _sid_to_seqnames(sid)

                    logger.log_pairs(
                        split="val",
                        epoch=ep,
                        step=batch_idx,
                        s_weather_pred=model.last_s_weather_log,
                        s_illum_pred=model.last_s_illum_log,
                        s_weather_gt=cam["weather_sev"].detach().cpu(),
                        s_illum_gt=cam["illum_sev"].detach().cpu(),
                        pair_b=pairs_batch["pair_b"].detach().cpu(),
                        pair_t=pairs_batch["pair_t"].detach().cpu(),
                        seq_names=seq_names_batch,   # <-- QUI
                    )


                
                    P = int(cam["weather_sev"].shape[0])
                    va_sum_sW += float(loss_sW.item()) * P
                    va_sum_sL += float(loss_sL.item()) * P
                    nva_env += P


                    loss = loss + 0.2 * loss_sW + 0.2 * loss_sL

               

                va_sum += float(loss.item())*y.size(0); nva += y.size(0)
                t_diff = pred_t - gt_t
                va_sum_sq_t += float(torch.sum(torch.sum(t_diff*t_diff, dim=1)).item())
                ang = quaternion_angular_error_rad(pred_q, gt_q)
                va_sum_sq_r += float(torch.sum(ang*ang).item()); va_cnt += y.size(0)

        va_loss = va_sum/max(1,nva); val_loss_hist.append(va_loss)
        va_loss_sW = va_sum_sW / max(1, nva_env)
        va_loss_sL = va_sum_sL / max(1, nva_env)


        val_loss_sW_hist.append(va_loss_sW)
        val_loss_sL_hist.append(va_loss_sL)
        val_rmse_t_hist.append(math.sqrt(va_sum_sq_t/max(1,va_cnt)))
        val_rmse_rdeg_hist.append(math.sqrt(va_sum_sq_r/max(1,va_cnt))*(180.0/math.pi))

        # =======================
        # Save minimal uncertainty evidence (VAL)
        # =======================
        if len(unc_logvar_t) > 0:
            unc_path = os.path.join(out_dir, f"uncertainty_val_ep{ep:03d}.npz")
            np.savez(
                unc_path,
                logvar_t=np.asarray(unc_logvar_t, dtype=np.float32),
                logvar_q=np.asarray(unc_logvar_q, dtype=np.float32),
                trans_err_m=np.asarray(unc_trans_err, dtype=np.float32),
                rot_err_deg=np.asarray(unc_rot_err_deg, dtype=np.float32),
            )


        print(f"[E{ep:02d}] train {tr_loss:.6f} | val {va_loss:.6f}  "
              f"(RMSE_T {train_rmse_t_hist[-1]:.3f}/{val_rmse_t_hist[-1]:.3f} m, "
              f"RMSE_R {train_rmse_rdeg_hist[-1]:.3f}/{val_rmse_rdeg_hist[-1]:.3f} deg)")

        ckpt = {
            "epoch": ep,
            "state_dict": model.state_dict(),
            "loss_w_q": float(crit.w_q.detach().cpu()),
            "loss_w_t": float(crit.w_t.detach().cpu()),
            "opt": opt.state_dict(),
            "global_step": global_step,
        }
        if use_amp:
            ckpt["scaler"] = scaler.state_dict()
        torch.save(ckpt, os.path.join(out_dir, "last.pt"))
        if va_loss < best:
            best = va_loss
            torch.save(ckpt, os.path.join(out_dir, "radar4d_lstm_best.pt"))

    with open(os.path.join(out_dir, "loss_history_se3patch.json"), "w") as f:
        json.dump({
            "train_loss": train_loss_hist,
            "val_loss": val_loss_hist,
            "train_rmse_t_m": train_rmse_t_hist,
            "val_rmse_t_m": val_rmse_t_hist,
            "train_rmse_r_deg": train_rmse_rdeg_hist,
            "val_rmse_r_deg": val_rmse_rdeg_hist,
            "train_loss_weather_sev": train_loss_sW_hist,
            "val_loss_weather_sev":   val_loss_sW_hist,

            "train_loss_illum_sev": train_loss_sL_hist,
            "val_loss_illum_sev":   val_loss_sL_hist,
        }, f, indent=2)

    xs = list(range(1, len(train_loss_hist)+1))
    plt.figure(); plt.plot(xs, train_loss_hist, label="Train"); plt.plot(xs, val_loss_hist, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss per Epoca")
    plt.grid(True, linestyle="--", alpha=0.4); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_se3patch.png"), dpi=200); plt.close()
    
    if model.use_weather:
        plt.figure()
        plt.plot(xs, train_loss_sW_hist, label="Train weather_sev")
        plt.plot(xs, val_loss_sW_hist, label="Val weather_sev")
        plt.xlabel("Epoch")
        plt.ylabel("SmoothL1 loss")
        plt.title("Weather severity supervision loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_weather_sev.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.plot(xs, train_loss_sL_hist, label="Train illum_sev")
        plt.plot(xs, val_loss_sL_hist, label="Val illum_sev")
        plt.xlabel("Epoch")
        plt.ylabel("SmoothL1 loss")
        plt.title("Illumination severity supervision loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "loss_illum_sev.png"), dpi=200)
        plt.close()


    plt.figure(figsize=(8, 4))
    plt.plot(xs, train_rmse_t_hist, label="Training", marker="o", markersize=3, linewidth=1.5)
    plt.plot(xs, val_rmse_t_hist,   label="Validation", marker="o", markersize=3, linewidth=1.5)

    plt.title("Translation Error (RMSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Error [m]")

    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rmse_translation_se3patch.png"), dpi=200)
    plt.close()


    plt.figure(figsize=(8, 4))
    plt.plot(xs, train_rmse_rdeg_hist, label="Training", marker="o", markersize=3, linewidth=1.5)
    plt.plot(xs, val_rmse_rdeg_hist,   label="Validation", marker="o", markersize=3, linewidth=1.5)

    plt.title("Mean Angular Error")
    plt.xlabel("Epoch")
    plt.ylabel("Error [deg]")  # <-- se vuoi in rad, vedi nota sotto

    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rmse_rotation_deg_se3patch.png"), dpi=200)
    plt.close()

    if model.use_weather:
       logger.save(os.path.join(out_dir, "env_scalars_all.csv"))
    return best

# ============================
# Eval per-split/per-sequenza (come prima)
# ============================
def evaluate_split_per_scene(seqs_cfg, checkpoint, out_root, split_key="val", device='cuda'):
    ds, train_loader, val_loader, test_loader, per_scene = make_loaders_radar_pose_raw(
        seqs_cfg, batch_size=BATCH_SIZE, seed=SEED, num_workers=0
    )
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loaders[split_key]

    model = Radar4DEncLSTM()
    ckpt = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(ckpt["state_dict"], strict=True)
    device = torch.device(device if torch.cuda.is_available() and device!='cpu' else 'cpu')
    model.to(device); model.eval()
    _set_a2do_hard(model, False)


    out_root = os.path.join(out_root, f"eval_{split_key}_se3patch")
    os.makedirs(out_root, exist_ok=True)

    all_metrics = {}
    ate3d_list, ate2d_list = [], []
    drift_t_list, drift_r_list = [], []

    # aggrega per sequenza
    # --- nuovo loop per-scena ordinato ---
    for s, idxs_scene in per_scene[split_key].items():
        seqname = seqs_cfg[s].name
        print(f"[eval {split_key} pred {seqname}]")

        subset = torch.utils.data.Subset(ds, idxs_scene)
        loader_scene = torch.utils.data.DataLoader(
            subset,
            batch_size=1,           # o BATCH_SIZE, ma 1 è più semplice/robusto per l'ordine
            shuffle=False,
            num_workers=0,
            collate_fn=pad_collate_radar_raw   # <-- QUESTO È IL PUNTO CHIAVE
)

        preds_list, gts_list = [], []

        with torch.no_grad():
            for pairs_batch, y, sid, lengths in tqdm(loader_scene, desc=f"[{seqname}]"):
                y = y.to(device)
                lengths = lengths.to(device)

                gt_t, gt_q = y[:, :3], y[:, 3:7]
                pred_q, pred_t, h_prev, key_padding_mask = model(pairs_batch, lengths)

                y7_pred = torch.cat([pred_t, pred_q], dim=1).cpu().numpy()
                y7_gt = y.cpu().numpy()

                preds_list.append(y7_pred)
                gts_list.append(y7_gt)

        if len(preds_list) == 0:
            continue

        dposes_pred = np.concatenate(preds_list, axis=0)
        dposes_gt = np.concatenate(gts_list, axis=0)

        # --- correzione continuità quaternioni ---
        dq = torch.from_numpy(dposes_pred[:, 3:7])
        dq = enforce_quat_continuity(dq)
        dposes_pred[:, 3:7] = dq.numpy()

        # --- ricostruisci traiettoria ---
        traj_pred = compose_trajectory_quat_se3patch(dposes_pred)
        traj_gt = compose_trajectory_quat_se3patch(dposes_gt)
        xyz_pred, xyz_gt = traj_pred[:, :3], traj_gt[:, :3]

        dq = torch.from_numpy(dposes_pred[:, 3:7])
        dq = enforce_quat_continuity(dq)
        dposes_pred[:,3:7] = dq.numpy()

        traj_pred = compose_trajectory_quat_se3patch(dposes_pred)
        traj_gt   = compose_trajectory_quat_se3patch(dposes_gt)
        xyz_pred, xyz_gt = traj_pred[:, :3], traj_gt[:, :3]

        ate3d_raw = ate3d_noalign(xyz_pred, xyz_gt)
        ate2d_raw = ate2d_noalign(xyz_pred, xyz_gt)

        yaw_pred = np.array([yaw_from_quat(qw, qx, qy, qz) for (tx,ty,tz,qx,qy,qz,qw) in traj_pred])
        yaw_gt   = np.array([yaw_from_quat(qw, qx, qy, qz) for (tx,ty,tz,qx,qy,qz,qw) in traj_gt])
        T_errs_100m, R_errs_deg_100m = kitti_drift(xyz_pred, yaw_pred, xyz_gt, yaw_gt)
        T_errs_per_m     = [float(v)/100.0 for v in T_errs_100m]
        R_errs_deg_per_m = [float(v)/100.0 for v in R_errs_deg_100m]

        seq_out = os.path.join(out_root, seqname); os.makedirs(seq_out, exist_ok=True)
        np.savez(os.path.join(seq_out, "traj_pred_gt.npz"), traj_pred=traj_pred, traj_gt=traj_gt)
        per_seq = {
            "ATE3D_raw_rmse_m": float(ate3d_raw),
            "ATE2D_raw_rmse_m": float(ate2d_raw),
            "T_errs_100m": T_errs_100m,
            "R_errs_deg_100m": R_errs_deg_100m,
            "T_errs_percent_per_m": T_errs_per_m,
            "R_errs_deg_per_m": R_errs_deg_per_m
        }
        with open(os.path.join(seq_out, "metrics.json"), "w") as f:
            json.dump(per_seq, f, indent=2)

        all_metrics[seqname] = per_seq
        ate3d_list.append(float(ate3d_raw)); ate2d_list.append(float(ate2d_raw))
        drift_t_list.extend(T_errs_per_m); drift_r_list.extend(R_errs_deg_per_m)

        # ===== titolo da SEQ_META =====
        weather_str, illum_str = SEQ_META.get(seqname, ("Clear", "Day"))
        title = f"Predicted Trajectory vs Ground Truth - {seqname} - {illum_str}/{weather_str}"

        # ===== plot =====
        plt.figure()

        # GT → blu elettrico (linea piena)
        plt.plot(
            xyz_gt[:, 0], xyz_gt[:, 1],
            label="Ground Truth",
            color="#00B0FF",
            linestyle="-",
            linewidth=2.5,
        )

        # Pred → rosso tratteggiato
        plt.plot(
            xyz_pred[:, 0], xyz_pred[:, 1],
            label="Predicted",
            color="red",
            linestyle="--",
            linewidth=2.5,
        )

        plt.axis("equal")

        # ===== assi con unità =====
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.title(title)

        plt.tight_layout()
        plt.savefig(os.path.join(seq_out, "traj_xy.png"), dpi=200)
        plt.close()


    split_summary = {
        "mean_ATE3D_raw_rmse_m": float(np.mean(ate3d_list)) if ate3d_list else None,
        "mean_ATE2D_raw_rmse_m": float(np.mean(ate2d_list)) if ate2d_list else None,
        "mean_translation_drift_percent_per_m": float(np.mean(drift_t_list)) if drift_t_list else None,
        "mean_rotation_drift_deg_per_m": float(np.mean(drift_r_list)) if drift_r_list else None
    }
    all_metrics["_split_summary"] = split_summary
    with open(os.path.join(out_root, "metrics_all.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    return all_metrics

def evaluate_split_per_scene_camimu(seqs_cfg_camimu, checkpoint, out_root, split_key="val", device='cuda', no_weather: bool = False):
    """
    Eval per-split/per-sequenza usando Radar+Camera+IMU (A-RCFM attivo + IMU disponibile).
    METRICHE e PLOT identici a evaluate_split_per_scene_cam:
      - ATE3D raw, ATE2D raw, ATE3D Sim3 aligned
      - KITTI drift (T_errs, R_errs)
      - RPE stile VoD
      - plot traj XY (traj_xy.png)
      - salvataggi metrics.json, metrics_all.json, rpe_vodstyle_all.json, traj_pred_gt.npz
    """
    # 1) Dataset Radar+Camera+IMU
    ds = Radar4DCamImuDatasetPairsRAW(seqs_cfg_camimu)

    # 2) Mapping sid -> indici globali (come nella versione cam)
    sid_to_indices = {sid: [] for sid in range(len(ds.rc_dataset.seqs))}
    for gidx, s in enumerate(ds.rc_dataset.samples):
        sid, _ = s
        sid_to_indices.setdefault(sid, []).append(gidx)



    perc = (0.70, 0.15, 0.15)
    idx_train, idx_val, idx_test = [], [], []
    per_scene_idx = {"train": {}, "val": {}, "test": {}}

    for sid in range(len(ds.rc_dataset.seqs)):
        idx_all = sid_to_indices[sid]
        n = len(idx_all)
        if n == 0:
            per_scene_idx["train"][sid] = []
            per_scene_idx["val"][sid]   = []
            per_scene_idx["test"][sid]  = []
            continue

        # mantieni ordine temporale
        ntr = int(perc[0] * n)
        nva = int(perc[1] * n)

        tr = idx_all[:ntr]
        va = idx_all[ntr:ntr + nva]
        te = idx_all[ntr + nva:]

        idx_train.extend(tr); per_scene_idx["train"][sid] = tr
        idx_val.extend(va);   per_scene_idx["val"][sid]   = va
        idx_test.extend(te);  per_scene_idx["test"][sid]  = te

    # 3) Sanity check indici
    N = len(ds) - 1
    def _sanitize(arr):
        return [i for i in arr if (0 <= i <= N)]
    idx_train = _sanitize(idx_train)
    idx_val   = _sanitize(idx_val)
    idx_test  = _sanitize(idx_test)

    split_map = {"train": idx_train, "val": idx_val, "test": idx_test}
    if split_key not in split_map:
        raise ValueError(f"split_key deve essere train/val/test, trovato: {split_key}")

    # 4) Carica modello
    model = Radar4DEncLSTM(use_weather=(not no_weather))
    ckpt = torch.load(checkpoint, map_location='cpu')
    sd = ckpt["state_dict"]

    # --- IMPORTANTISSIMO: istanzia imu_encoder PRIMA del load_state_dict ---
    if "imu_encoder.weight_ih_l0" in sd:
        D = sd["imu_encoder.weight_ih_l0"].shape[1]   # input dim IMU dal checkpoint
        model._ensure_imu_encoder(D, device=torch.device("cpu"))

    # ora combaciano le chiavi e puoi usare strict=True
    model.load_state_dict(sd, strict=True)

    device = torch.device(device if torch.cuda.is_available() and device != 'cpu' else 'cpu')
    model.to(device); model.eval()
    _set_a2do_hard(model, False)



    out_root = os.path.join(out_root, f"eval_{split_key}_se3patch")
    os.makedirs(out_root, exist_ok=True)

    all_metrics = {}
    ate3d_list, ate2d_list = [], []
    ate3d_sim3_list = []
    drift_t_list, drift_r_list = [], []
    rpe_all = {}

    # Whole-sequence drift accumulators (for length-weighted mean)
    ws_trans_drift_list = []   # one float per sequence
    ws_rot_drift_list   = []   # one float per sequence
    ws_path_lengths     = []   # GT path length per sequence [m]

    # 5) Loop per-sequenza per lo split scelto
    for sid, idxs_scene in per_scene_idx[split_key].items():
        if not idxs_scene:
            continue

        seqname = ds.rc_dataset.seqs[sid].name
        print(f"[eval_camimu {split_key} pred {seqname}]")

        subset = torch.utils.data.Subset(ds, idxs_scene)
        loader_scene = torch.utils.data.DataLoader(
            subset,
            batch_size=1,           # preserva ordine temporale
            shuffle=False,
            num_workers=0,
            collate_fn=pad_collate_radar_cam_imu_raw_batched
        )

        preds_list, gts_list = [], []

        with torch.no_grad():
            for pairs_batch, cam, imu, y, sid_batch, lengths in tqdm(loader_scene, desc=f"[{seqname}]"):
                y = y.to(device)
                lengths = lengths.to(device)

                gt_t, gt_q = y[:, :3], y[:, 3:7]
                pred_q, pred_t, h_prev, key_padding_mask = model(pairs_batch, lengths, cam=cam, imu=imu)

                y7_pred = torch.cat([pred_t, pred_q], dim=1).cpu().numpy()
                y7_gt   = y.cpu().numpy()

                preds_list.append(y7_pred)
                gts_list.append(y7_gt)

        if len(preds_list) == 0:
            continue

        dposes_pred = np.concatenate(preds_list, axis=0)
        dposes_gt   = np.concatenate(gts_list,   axis=0)

        # --- correzione continuità quaternioni ---
        dq = torch.from_numpy(dposes_pred[:, 3:7])
        dq = enforce_quat_continuity(dq)
        dposes_pred[:, 3:7] = dq.numpy()

        traj_pred = compose_trajectory_quat_se3patch(dposes_pred)
        traj_gt   = compose_trajectory_quat_se3patch(dposes_gt)
        xyz_pred, xyz_gt = traj_pred[:, :3], traj_gt[:, :3]

        # --- ATE raw + ATE allineato Sim(3) ---
        ate3d_raw  = ate3d_noalign(xyz_pred, xyz_gt)
        ate2d_raw  = ate2d_noalign(xyz_pred, xyz_gt)
        ate3d_sim3 = ate3d_aligned_sim3(xyz_pred, xyz_gt, with_scale=True)

        # --- RPE stile VoD / KITTI (segmenti 20..160m) ---
        rpe_vod = compute_rpe_vod_style(traj_pred, traj_gt)

        # --- drift KITTI ---
        yaw_pred = np.array([yaw_from_quat(qw, qx, qy, qz) for (tx,ty,tz,qx,qy,qz,qw) in traj_pred])
        yaw_gt   = np.array([yaw_from_quat(qw, qx, qy, qz) for (tx,ty,tz,qx,qy,qz,qw) in traj_gt])
        T_errs_100m, R_errs_deg_100m = kitti_drift(xyz_pred, yaw_pred, xyz_gt, yaw_gt)
        T_errs_per_m     = [float(v)/100.0 for v in T_errs_100m]
        R_errs_deg_per_m = [float(v)/100.0 for v in R_errs_deg_100m]

        # --- whole-sequence drift ---
        ws_t_drift, ws_r_drift, ws_pathlen = whole_sequence_drift(
            xyz_pred, yaw_pred, xyz_gt, yaw_gt
        )

        # --- salva per-sequenza ---
        seq_out = os.path.join(out_root, seqname)
        os.makedirs(seq_out, exist_ok=True)

        np.savez(os.path.join(seq_out, "traj_pred_gt.npz"), traj_pred=traj_pred, traj_gt=traj_gt)

        per_seq = {
            "ATE3D_raw_rmse_m": float(ate3d_raw),
            "ATE2D_raw_rmse_m": float(ate2d_raw),
            "ATE3D_sim3_aligned_rmse_m": float(ate3d_sim3),
            "T_errs_100m": T_errs_100m,
            "R_errs_deg_100m": R_errs_deg_100m,
            "T_errs_percent_per_m": T_errs_per_m,
            "R_errs_deg_per_m": R_errs_deg_per_m,
            "RPE_VoD": rpe_vod,
            "whole_seq_trans_drift_pct": ws_t_drift,
            "whole_seq_rot_drift_deg_per_m": ws_r_drift,
            "whole_seq_path_length_m": ws_pathlen,
        }
        with open(os.path.join(seq_out, "metrics.json"), "w") as f:
            json.dump(per_seq, f, indent=2)

        all_metrics[seqname] = per_seq
        ate3d_list.append(float(ate3d_raw))
        ate2d_list.append(float(ate2d_raw))
        ate3d_sim3_list.append(float(ate3d_sim3))
        drift_t_list.extend(T_errs_per_m)
        drift_r_list.extend(R_errs_deg_per_m)
        # Accumulate whole-sequence drift (skip sequences too short)
        if ws_t_drift is not None:
            ws_trans_drift_list.append(ws_t_drift)
            ws_rot_drift_list.append(ws_r_drift)
            ws_path_lengths.append(ws_pathlen)

        # --- PLOT identico (traj XY) ---
        # ===== titolo da SEQ_META =====
        weather_str, illum_str = SEQ_META.get(seqname, ("Clear", "Day"))
        title = f"Predicted Trajectory vs Ground Truth - {seqname} - {illum_str}/{weather_str}"

        # ===== plot =====
        plt.figure()

        # GT → blu elettrico (linea piena)
        plt.plot(
            xyz_gt[:, 0], xyz_gt[:, 1],
            label="Ground Truth",
            color="#00B0FF",
            linestyle="-",
            linewidth=1.5,
        )

        # Pred → rosso tratteggiato
        plt.plot(
            xyz_pred[:, 0], xyz_pred[:, 1],
            label="Predicted",
            color="red",
            linestyle="--",
            linewidth=1.5,
        )

        plt.axis("equal")

        # ===== assi con unità =====
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")

        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.title(title)

        plt.tight_layout()
        plt.savefig(os.path.join(seq_out, "traj_xy.png"), dpi=200)
        plt.close()


    # --- summary split ---
    split_summary = {
        "mean_ATE3D_raw_rmse_m": float(np.mean(ate3d_list)) if ate3d_list else None,
        "mean_ATE2D_raw_rmse_m": float(np.mean(ate2d_list)) if ate2d_list else None,
        "mean_ATE3D_sim3_aligned_rmse_m": float(np.mean(ate3d_sim3_list)) if ate3d_sim3_list else None,
        "mean_translation_drift_percent_per_m": float(np.mean(drift_t_list)) if drift_t_list else None,
        "mean_rotation_drift_deg_per_m": float(np.mean(drift_r_list)) if drift_r_list else None,
    }

    # --- whole-sequence drift: length-weighted mean across sequences ---
    if ws_trans_drift_list:
        ws_weights = np.array(ws_path_lengths, dtype=np.float64)
        ws_weights /= ws_weights.sum()
        ws_mean_trans = float(np.dot(ws_weights, ws_trans_drift_list))
        ws_mean_rot   = float(np.dot(ws_weights, ws_rot_drift_list))
        ws_simple_trans = float(np.mean(ws_trans_drift_list))
        ws_simple_rot   = float(np.mean(ws_rot_drift_list))
    else:
        ws_mean_trans = ws_mean_rot = None
        ws_simple_trans = ws_simple_rot = None

    split_summary["whole_seq_trans_drift_pct_weighted_mean"]     = ws_mean_trans
    split_summary["whole_seq_rot_drift_deg_per_m_weighted_mean"] = ws_mean_rot
    split_summary["whole_seq_trans_drift_pct_simple_mean"]       = ws_simple_trans
    split_summary["whole_seq_rot_drift_deg_per_m_simple_mean"]   = ws_simple_rot

    all_metrics["_split_summary"] = split_summary

    with open(os.path.join(out_root, "metrics_all.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # file separato RPE stile VoD per tutte le sequenze
    rpe_all = {seq: all_metrics[seq]["RPE_VoD"] for seq in all_metrics if seq != "_split_summary"}
    with open(os.path.join(out_root, "rpe_vodstyle_all.json"), "w") as f:
        json.dump(rpe_all, f, indent=2)

    return all_metrics


# ============================
# CLI (identica, ma default GT e calib)
# ============================
def parse_seqs_arg(s):
    return [p.strip() for p in s.split(",") if p.strip()]

def main():
    ap = argparse.ArgumentParser("Radar4D Seq2Seq LSTM — encoder in-model + quaternion + adaptive loss (SE3 patch)")
    ap.add_argument("--mode", choices=["train","eval"], required=True)
    ap.add_argument("--seqs", required=True, help='es: "01,02,03"')
    ap.add_argument("--output_root", default="outputs/imu_seq2seq_lstm_radarstep_twohead_quat_adaloss_rmse_se3patch_inmodelenc")
    ap.add_argument("--checkpoint", default="", help="usato in --mode eval")
    ap.add_argument("--eval_split", choices=["train","val","test","both"], default="val")
    ap.add_argument("--wq_init", type=float, default=-2.5)
    ap.add_argument("--wt_init", type=float, default=0.0)
    ap.add_argument("--root_dir", type=str, default="/media/arrubuntu20/HDD/Hercules")
    ap.add_argument("--pcd_rel", type=str, default="Continental/continental_pcd")
    ap.add_argument("--img_rel", type=str, default="stereo_left")
    ap.add_argument("--resume_train", default="", help="Path a checkpoint .pt (es. .../last.pt) per riprendere il TRAIN")
    # GT: local_inspva.txt nella stessa cartella PR_GT/ della sequenza
    ap.add_argument("--gt_rel",  type=str, default="PR_GT/local_inspva.txt")
    # la calib è nella cartella "Calibration" della sequenza
    ap.add_argument("--calib_rel", type=str, default="Calibration")
    ap.add_argument("--imu_rel", type=str, default="sensor_data/xsens_imu.csv",
                    help="Path relativo al file IMU dentro ogni sequenza")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--bs", type=int, default=BATCH_SIZE)
    ap.add_argument("--nw", type=int, default=8)
    ap.add_argument("--amp", action="store_true", help="Use mixed precision AMP")
    ap.add_argument("--no_weather", action="store_true",
                help="Disabilita EnvHead + scalari meteo/illum + FiLM + q_cam gating + loss/logging ambiente")

    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True

    seq_ids = parse_seqs_arg(args.seqs)

    # Config per eval (solo radar, come prima)
    seqs_cfg = [SeqConfig(
                    name=sid,
                    pcd_dir=os.path.join(args.root_dir, sid, args.pcd_rel),
                    gt_path=os.path.join(args.root_dir, sid, args.gt_rel),
                    calib_dir=os.path.join(args.root_dir, sid, args.calib_rel)
                ) for sid in seq_ids]

    # Config per train/val con camera (fusione A-RCFM)
    seqs_cfg_cam = [SeqConfigCam(
                        name=sid,
                        pcd_dir=os.path.join(args.root_dir, sid, args.pcd_rel),
                        img_dir=os.path.join(args.root_dir, sid, args.img_rel),
                        gt_path=os.path.join(args.root_dir, sid, args.gt_rel),
                        calib_dir=os.path.join(args.root_dir, sid, args.calib_rel)
                    ) for sid in seq_ids]

    # Config per train/val con camera + imu
    seqs_cfg_camimu = [SeqConfigCamImu(
                        name=sid,
                        pcd_dir=os.path.join(args.root_dir, sid, args.pcd_rel),
                        img_dir=os.path.join(args.root_dir, sid, args.img_rel),
                        gt_path=os.path.join(args.root_dir, sid, args.gt_rel),
                        calib_dir=os.path.join(args.root_dir, sid, args.calib_rel),
                        imu_path=os.path.join(args.root_dir, sid, args.imu_rel),
                        seq_len=4,
                        window_stride=2,
                    ) for sid in seq_ids]

    os.makedirs(args.output_root, exist_ok=True)


    if args.mode == "train":
        # ====== TRAIN/VAL CON FUSIONE RADAR+CAMERA ======
        # 1) Crea i loader camera-only SOLO per avere lo split
        train_loader, val_loader, test_loader, train_ds = make_loaders_radar_cam_imu_pose_raw_batched(
            seqs_cfg_camimu,
            batch_size=args.bs,
            num_workers=args.nw,
            shuffle_train=True,
            perc=(0.70,0.15,0.15),
            imu_norm_stats=None,  
        )
        print("DEBUG len(train_loader) =", len(train_loader))
        print("DEBUG len(train_dataset) =", len(train_loader.dataset))


        best = train_and_validate(train_loader, val_loader, args.output_root, seqs_cfg_cam=seqs_cfg_cam,
                                  wq_init=args.wq_init, wt_init=args.wt_init, device=args.device, no_weather=args.no_weather, use_amp=args.amp, resume_path=args.resume_train)
        print("Training done. Best val loss:", best)
    else:
        # ====== EVAL CON FUSIONE RADAR+CAMERA ======
        if not args.checkpoint:
            guess = os.path.join(args.output_root, "radar4d_lstm_best.pt")
            if os.path.exists(guess):
                args.checkpoint = guess
            else:
                raise ValueError("In eval serve --checkpoint")
        if args.eval_split == "both":
            res_val  = evaluate_split_per_scene_camimu(seqs_cfg_camimu, args.checkpoint, args.output_root, split_key="val",  device=args.device, no_weather=args.no_weather)
            res_test = evaluate_split_per_scene_camimu(seqs_cfg_camimu, args.checkpoint, args.output_root, split_key="test", device=args.device, no_weather=args.no_weather)
            with open(os.path.join(args.output_root, "eval_summary_val_test_se3patch_cam.json"), "w") as f:
                json.dump({"val":res_val, "test":res_test}, f, indent=2)
            print("[SUMMARY] salvato eval_summary_val_test_se3patch_cam.json")
        else:
            evaluate_split_per_scene_camimu(seqs_cfg_camimu, args.checkpoint, args.output_root, split_key=args.eval_split, device=args.device, no_weather=args.no_weather)



if __name__ == "__main__":
    main()
