#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# IMPORT dal dataloader versione IMU-step (nuovo file)
from imu_data_imustep_quat_localpatch import make_loaders_quat_localpatch, pad_collate_quat_localpatch

# ============================
# Hyperparam (come prima)
# ============================
BATCH_SIZE = 16
EPOCHS = 60
LR = 1e-4
HIDDEN_DIM = 256
NUM_LAYERS = 3
SEED = 0
CAP_TRAIN_WINDOWS_PER_SCENE = None

# ============================
# Helper per quaternioni (xyzw)
# ============================
def quat_norm_xyzw(q):  # q [...,4] in ordine [x,y,z,w]
    return F.normalize(q, dim=-1)

def to_wxyz(q_xyzw):
    x,y,z,w = q_xyzw.unbind(-1)
    return torch.stack([w,x,y,z], dim=-1)

def to_xyzw(q_wxyz):
    w,x,y,z = q_wxyz.unbind(-1)
    return torch.stack([x,y,z,w], dim=-1)

def quat_dot_xyzw(a, b):
    return torch.sum(a*b, dim=-1, keepdim=True)

def align_quat_to_gt(q_pred_xyzw, q_gt_xyzw):
    q_pred = quat_norm_xyzw(q_pred_xyzw)
    q_gt   = quat_norm_xyzw(q_gt_xyzw)
    dot = quat_dot_xyzw(q_pred, q_gt)
    q_pred_aligned = torch.where(dot < 0, -q_pred, q_pred)
    dot_abs = torch.clamp(dot.abs(), 0.0, 1.0)
    return q_pred_aligned, dot_abs

def geodesic_angle_from_dot_abs(dot_abs):
    dot_abs = torch.clamp(dot_abs, 0.0, 1.0 - 1e-7)
    return 2.0 * torch.acos(dot_abs)  # rad

def geodesic_loss_deg(q_pred_xyzw, q_gt_xyzw, reduction="mean"):
    q_pred_aligned, dot_abs = align_quat_to_gt(q_pred_xyzw, q_gt_xyzw)
    angle = geodesic_angle_from_dot_abs(dot_abs)  # rad
    loss = angle**2
    return loss.mean() if reduction=="mean" else loss

def enforce_quat_continuity(q_seq_xyzw):
    q = quat_norm_xyzw(q_seq_xyzw)
    if q.dim() == 2:
        q = q.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False
    B, T, _ = q.shape
    out = q.clone()
    for t in range(1, T):
        dot = torch.sum(out[:, t-1, :] * out[:, t, :], dim=-1, keepdim=True)
        flip = (dot < 0.0)
        out[:, t, :] = torch.where(flip, -out[:, t, :], out[:, t, :])
    if squeeze_back:
        out = out.squeeze(0)
    return out

# ============================
# LOSS adattiva (identica)
# ============================
def quaternion_angular_error_rad(pred_q, gt_q):
    pred_q_norm = F.normalize(pred_q, p=2, dim=-1)
    gt_q_norm = F.normalize(gt_q, p=2, dim=-1)
    dot_product = torch.abs(torch.sum(pred_q_norm * gt_q_norm, dim=-1))
    dot_product_clamped = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
    angle_rad = 2 * torch.acos(dot_product_clamped)
    return angle_rad

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveLoss(nn.Module):
    def __init__(self,
                 wq_init: float = -1.0, wt_init: float = 0.0,     # init asimmetrici
                 wq_min: float = -10.0, wq_max: float = 2.0,       # rot: più libertà
                 wt_min: float = -4.0, wt_max: float = 1.5,       # trasl: meno libertà
                 huber_delta_deg: float = 1.0):                   # soglia Huber (1°)
        super().__init__()
        self.w_q = nn.Parameter(torch.tensor(wq_init, dtype=torch.float32))
        self.w_t = nn.Parameter(torch.tensor(wt_init, dtype=torch.float32))
        self.wq_min, self.wq_max = float(wq_min), float(wq_max)
        self.wt_min, self.wt_max = float(wt_min), float(wt_max)
        self.delta = float(huber_delta_deg) * math.pi / 180.0     # in radianti

    def forward(self, pred_q, pred_t, gt_q, gt_t):
        # --- Rotazione: normalizza e allinea il segno (q ~ -q)
        pred_q = F.normalize(pred_q, p=2, dim=-1)
        gt_q   = F.normalize(gt_q,   p=2, dim=-1)
        dot = torch.sum(pred_q * gt_q, dim=-1, keepdim=True)
        pred_q = torch.where(dot < 0.0, -pred_q, pred_q)

        dot_abs = torch.clamp(torch.sum(pred_q * gt_q, dim=-1).abs(), 0.0, 1.0 - 1e-7)
        angle = 2.0 * torch.acos(dot_abs)    # radianti

        # --- Huber sull'angolo (robusta ma sensibile vicino a 0)
        abs_a = torch.abs(angle)
        delta = torch.tensor(self.delta, device=angle.device, dtype=angle.dtype)
        # forma standard: piecewise quadratica/lineare
        loss_q = torch.where(
            abs_a <= delta,
            0.5 * (abs_a * abs_a) / delta,   # zona quadratica
            abs_a - 0.5 * delta              # zona lineare
        ).mean()

        # --- Traslazione (SmoothL1 come prima)
        loss_t = F.smooth_l1_loss(pred_t, gt_t)

        # --- Pesi asimmetrici (clamp diversi per R e T)
        w_q = torch.clamp(self.w_q, min=self.wq_min, max=self.wq_max)
        w_t = torch.clamp(self.w_t, min=self.wt_min, max=self.wt_max)

        total = loss_q * torch.exp(-w_q) + w_q + loss_t * torch.exp(-w_t) + w_t
        return total

# Modello two-head (xyzw) — identico
# ============================
class IMUPoseSeq2SeqTwoHeadQuat(nn.Module):
    def __init__(self, in_dim, enc_hidden=128, dec_hidden=128,
                 enc_layers=2, dec_layers=2, dropout=0.4):
        super().__init__()
        self.encoder = nn.LSTM(input_size=in_dim, hidden_size=enc_hidden,
                               num_layers=enc_layers, batch_first=True,
                               dropout=dropout, bidirectional=False)
        self.h_adapt = nn.Linear(enc_hidden, dec_hidden) if enc_hidden != dec_hidden else None
        self.decoder = nn.LSTM(input_size=enc_hidden, hidden_size=dec_hidden,
                               num_layers=dec_layers, batch_first=True,
                               dropout=dropout, bidirectional=False)
        self.trunk = nn.Sequential(
            nn.Linear(dec_hidden, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.head_t = nn.Linear(64, 3)  # Δt_local (frame IMU)
        self.head_r = nn.Linear(64, 4)  # dq (x,y,z,w)

    def forward(self, x, lengths=None):
        B = x.size(0)
        if lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h, c) = self.encoder(x_packed)   # hidden già all'ultimo timestep reale
            enc_hidden_dim = self.encoder.hidden_size
        else:
            enc_out, (h, c) = self.encoder(x)
            enc_hidden_dim = enc_out.size(-1)

        # opzionale adattamento dimensione hidden
        if self.h_adapt is not None:
            h = torch.stack([self.h_adapt(h[l]) for l in range(h.size(0))], dim=0)
            c = torch.stack([self.h_adapt(c[l]) for l in range(c.size(0))], dim=0)

        go = torch.zeros(B, 1, enc_hidden_dim, device=x.device, dtype=x.dtype)
        dec_out, _ = self.decoder(go, (h, c))
        z = self.trunk(dec_out[:, -1, :])
        t = self.head_t(z)
        q = self.head_r(z)  # (B,4) xyzw
        y7 = torch.cat([t, q], dim=-1)
        return y7, t, q

# ============================
# Metriche & util — SE(3) patch (identico)
# ============================
def quat_normalize_np(q):
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0,0,0,1], dtype=np.float64)
    return q / n

def quat_mul_np(q1, q2):
    x1,y1,z1,w1 = q1
    x2,y2,z2,w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return quat_normalize_np(np.array([x,y,z,w], dtype=np.float64))

def quat_to_mat_np(q):
    x,y,z,w = quat_normalize_np(q)
    xx,yy,zz = x*x, y*y, z*z
    xy,xz,yz = x*y, x*z, y*z
    wx,wy,wz = w*x, w*y, w*z
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
    out = np.concatenate([T, Q], axis=1)
    return out

def ate3d_noalign(pred_xyz, gt_xyz):
    n = min(len(pred_xyz), len(gt_xyz))
    err = np.linalg.norm(pred_xyz[:n] - gt_xyz[:n], axis=1)
    return float(np.sqrt(np.mean(err**2)))

def ate2d_noalign(pred_xyz, gt_xyz):
    n = min(len(pred_xyz), len(gt_xyz))
    pe = pred_xyz[:n, :2]; ge = gt_xyz[:n, :2]
    err = np.linalg.norm(pe - ge, axis=1)
    return float(np.sqrt(np.mean(err**2)))

def yaw_from_quat(qw,qx,qy,qz):
    return math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

def kitti_drift(pred_xyz, pred_yaw, gt_xyz, gt_yaw, seg_lengths=(10,50,100,200,300,400,500,600,700,800), step=10):
    T_errs, R_errs = [], []
    for L in seg_lengths:
        rts, rrs = [], []
        for start in range(0, len(gt_xyz)-1, step):
            dist=0.0; k=start
            while k < len(gt_xyz)-1 and dist < L:
                dist += np.linalg.norm(gt_xyz[k+1]-gt_xyz[k]); k+=1
            if dist < L*0.9:
               continue
            gt_rel = gt_xyz[k]-gt_xyz[start]
            pr_rel = pred_xyz[k]-pred_xyz[start]
            t_err = np.linalg.norm(gt_rel - pr_rel)/max(L,1e-6)*100.0
            ry = (pred_yaw[k]-pred_yaw[start]) - (gt_yaw[k]-gt_yaw[start])
            r_err = abs(ry)/(L/100.0)*(180.0/math.pi)
            rts.append(t_err); rrs.append(r_err)
        if rts:
            T_errs.append(float(np.mean(rts)))
            R_errs.append(float(np.mean(rrs)))
    return T_errs, R_errs

# ============================
# Train + Validate — invariato (etichette IMU frame)
# ============================
def train_and_validate(seqs, output_root, wq_init=-2.5, wt_init=0.0):
    torch.manual_seed(SEED); np.random.seed(SEED)

    ds, train_loader, val_loader, _, per_scene_idx = make_loaders_quat_localpatch(
        seqs=seqs, batch_size=BATCH_SIZE,
        cap_train_windows_per_scene=CAP_TRAIN_WINDOWS_PER_SCENE,
        seed=SEED, perc=(0.70,0.15,0.15), num_workers=0
    )


    in_dim = ds.X_norm[0].shape[1]
    model = IMUPoseSeq2SeqTwoHeadQuat(in_dim=in_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # dopo model.to(device)
    crit = AdaptiveLoss(
        wq_init=-1.0, wt_init=0.0,      # init bilanciati ma con enfasi leggera su R
        wq_min=-10.0, wq_max=2.0,        # rotazione: può pesare molto (anti-bias)
        wt_min=-4.0, wt_max=1.5,        # traslazione: peso limitato (già va bene)
        huber_delta_deg=1.0
    ).to(device)

    optim = torch.optim.Adam(
        [*model.parameters(), *crit.parameters()],
        lr=LR, weight_decay=1e-4
    )


    best = 1e9
    train_loss_hist = []
    val_loss_hist = []
    train_rmse_t_hist,   val_rmse_t_hist   = [], []
    train_rmse_rdeg_hist, val_rmse_rdeg_hist = [], []

    os.makedirs(output_root, exist_ok=True)

    for ep in range(EPOCHS):
        model.train(); tr_loss=0.0; ntr=0
        tr_sum_sq_t = 0.0; tr_sum_sq_r = 0.0; tr_count = 0

        for xb,yb,_,lengths in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            _, t_pred, q_pred = model(xb,lengths)
            t_gt, q_gt = yb[:, :3], yb[:, 3:]
            l = crit(q_pred, t_pred, q_gt, t_gt)
            l.backward(); optim.step()
            tr_loss += float(l.item()) * xb.size(0); ntr += xb.size(0)

            t_diff = t_pred - t_gt
            tr_sum_sq_t += float(torch.sum(torch.sum(t_diff*t_diff, dim=1)).item())
            angles_rad = quaternion_angular_error_rad(q_pred, q_gt)
            tr_sum_sq_r += float(torch.sum(angles_rad * angles_rad).item())
            tr_count += xb.size(0)

        tr_m = tr_loss / max(1,ntr)
        train_loss_hist.append(tr_m)
        train_rmse_t = math.sqrt(tr_sum_sq_t / max(1, tr_count))
        train_rmse_rdg = math.sqrt(tr_sum_sq_r / max(1, tr_count)) * (180.0 / math.pi)
        train_rmse_t_hist.append(train_rmse_t)
        train_rmse_rdeg_hist.append(train_rmse_rdg)

        # ===== VALIDATION =====
        model.eval(); va_loss=0.0; nva=0
        va_sum_sq_t = 0.0; va_sum_sq_r = 0.0; va_count = 0
        with torch.no_grad():
            for xb,yb,_,lengths in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                _, t_pred, q_pred = model(xb,lengths)
                t_gt, q_gt = yb[:, :3], yb[:, 3:]
                l = crit(q_pred, t_pred, q_gt, t_gt)
                va_loss += float(l.item()) * xb.size(0); nva += xb.size(0)

                t_diff = t_pred - t_gt
                va_sum_sq_t += float(torch.sum(torch.sum(t_diff*t_diff, dim=1)).item())
                angles_rad = quaternion_angular_error_rad(q_pred, q_gt)
                va_sum_sq_r += float(torch.sum(angles_rad * angles_rad).item())
                va_count += xb.size(0)

        va_m = va_loss / max(1,nva)
        val_loss_hist.append(va_m)
        val_rmse_t   = math.sqrt(va_sum_sq_t / max(1, va_count))
        val_rmse_rdg = math.sqrt(va_sum_sq_r / max(1, va_count)) * (180.0 / math.pi)
        val_rmse_t_hist.append(val_rmse_t)
        val_rmse_rdeg_hist.append(val_rmse_rdg)

        print(f"[E{ep+1:02d}] train loss {tr_m:.6f}  val loss {va_m:.6f}  (w_q={float(crit.w_q.item()):.3f}, w_t={float(crit.w_t.item()):.3f})")
        print(f"       RMSE_T(train/val) = {train_rmse_t:.4f} m / {val_rmse_t:.4f} m"
              f" | RMSE_R(train/val) = {train_rmse_rdg:.3f}° / {val_rmse_rdg:.3f}°")

        if va_m < best:
            best = va_m
            torch.save({
                "model": model.state_dict(),
                "in_dim": in_dim,
                "cfg": {"HIDDEN_DIM":HIDDEN_DIM,"NUM_LAYERS":NUM_LAYERS,
                        "wq_init": float(wq_init), "wt_init": float(wt_init)},
                "per_scene_idx": per_scene_idx,
                "norm": {
                    "mean": ds.mean.tolist() if ds.mean is not None else None,
                    "std":  ds.std.tolist()  if ds.std  is not None else None
                },
                "seq_ids": ds.seq_ids
            }, os.path.join(output_root, "imu_seq2seq_lstm_imustep_twohead_quat_adaloss_rmse_se3patch_best.pt"))

    print("Training done. Best val loss:", best)

    with open(os.path.join(output_root, "loss_history_se3patch.json"), "w") as f:
        json.dump({
            "train_loss": train_loss_hist,
            "val_loss": val_loss_hist,
            "train_rmse_t_m": train_rmse_t_hist,
            "val_rmse_t_m": val_rmse_t_hist,
            "train_rmse_r_deg": train_rmse_rdeg_hist,
            "val_rmse_r_deg": val_rmse_rdeg_hist
        }, f, indent=2)

    plt.figure()
    plt.plot(range(1, len(train_loss_hist)+1), train_loss_hist, label="Train loss")
    plt.plot(range(1, len(val_loss_hist)+1), val_loss_hist, label="Val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("IMU Seq2Seq — Adaptive quaternion loss (SE3 patch) — IMU frame")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_root, "loss_curve_se3patch.png"), dpi=200); plt.close()

    plt.figure()
    plt.plot(range(1, len(train_rmse_t_hist)+1), train_rmse_t_hist, label="Train RMSE T (m)")
    plt.plot(range(1, len(val_rmse_t_hist)+1),   val_rmse_t_hist,   label="Val RMSE T (m)")
    plt.xlabel("Epoch"); plt.ylabel("RMSE T [m]")
    plt.title("RMSE Traslazione per Epoca (SE3 patch) — IMU frame")
    plt.grid(True, linestyle="--", alpha=0.4); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_root, "rmse_translation_se3patch.png"), dpi=200); plt.close()

    plt.figure()
    plt.plot(range(1, len(train_rmse_rdeg_hist)+1), train_rmse_rdeg_hist, label="Train RMSE R (deg)")
    plt.plot(range(1, len(val_rmse_rdeg_hist)+1),   val_rmse_rdeg_hist,   label="Val RMSE R (deg)")
    plt.xlabel("Epoch"); plt.ylabel("RMSE R [deg]")
    plt.title("RMSE Rotazione per Epoca (SE3 patch) — IMU frame")
    plt.grid(True, linestyle="--", alpha=0.4); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_root, "rmse_rotation_deg_se3patch.png"), dpi=200); plt.close()

    return float(best)

# ============================
# EVAL per scene con traiettoria (identico; IMU frame)
# ============================
def evaluate_split_per_scene_se3patch(seqs, checkpoint, output_root, split="val", plot_frame="imu"):
    # >>>>> NUOVO HEADER (sostituisce quello che ricrea loader/split) <<<<<
    import numpy as np
    import torch, os, json
    from torch.utils.data import DataLoader, Subset
    from imu_data_imustep_quat_localpatch import (
        MultiSeqIMUStepsQuatLocalPatch, pad_collate_quat_localpatch
    )

    # 1) Carico il checkpoint e ottengo per_scene_idx + norm salvati a train
    ckpt = torch.load(checkpoint, map_location='cpu')
    per_scene_idx = ckpt.get("per_scene_idx", None)
    if per_scene_idx is None:
        raise ValueError("Checkpoint privo di 'per_scene_idx'. Riesegui il training salvando lo split.")

    # 2) Creo il dataset 'grezzo' per le seq richieste
    ds = MultiSeqIMUStepsQuatLocalPatch(seqs)

    # 3) Applico la stessa normalizzazione del train se presente nel ckpt
    if "norm" in ckpt and ckpt["norm"] is not None and ckpt["norm"].get("mean") is not None:
        mean = np.array(ckpt["norm"]["mean"], dtype=np.float32)
        std  = np.array(ckpt["norm"]["std"],  dtype=np.float32)
        # metodo del tuo dataset che reimposta mean/std senza ricalcolarle
        ds.regenerate_norm(mean, std)
    else:
        print("[WARN] checkpoint senza normalizzazione salvata: userò quella ricalcolata dal dataset (meno riproducibile).")

    # 4) Inizializzo il modello dopo aver letto ckpt
    model = IMUPoseSeq2SeqTwoHeadQuat(in_dim=ckpt["in_dim"])
    model.load_state_dict(ckpt["model"]); model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)

    # 5) Path di output + mapping split
    os.makedirs(output_root, exist_ok=True)
    out_root = os.path.join(output_root, f"eval_{split}_se3patch"); os.makedirs(out_root, exist_ok=True)

    all_metrics = {}
    split_map = {"train": "train", "val": "val", "test": "test"}
    split_key = split_map[split]
    # >>>>> FINE NUOVO HEADER <<<<<


    ate3d_list, ate2d_list = [], []
    drift_t_list, drift_r_list = [], []

    for sid, seqname in enumerate(ds.seq_ids):
        idxs_scene = per_scene_idx[split_key].get(sid, [])
        if not idxs_scene:
            print(f"[{split}] scena {seqname}: nessun intervallo.")
            continue

        loader_scene = DataLoader(
            Subset(ds, idxs_scene),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
            collate_fn=pad_collate_quat_localpatch
        )

        pairs = []
        with torch.no_grad():
            for batch in loader_scene:
                if len(batch) == 4:
                    xb, yb, idx_batch,lengths = batch
                elif len(batch) == 3:
                    xb, yb, idx_batch = batch
                    lengths = None
                else:
                    xb, yb = batch
                    lengths = None
                    if 'accum_idx' not in locals():
                        accum_idx = 0
                    idx_batch = torch.arange(accum_idx, accum_idx + xb.shape[0])
                    accum_idx += xb.shape[0]

                xb = xb.to(device)
                if lengths is not None:
                    lengths = lengths.to(device)
                y7, _, _ = model(xb,lengths)
                y7 = y7.cpu().numpy()
                yb = yb.numpy()
                for i in range(len(idx_batch)):
                    pairs.append((int(idx_batch[i]), y7[i], yb[i]))

        if len(pairs) == 0:
            print(f"[{split}] scena {seqname}: nessun dato dopo il loader.")
            continue

        pairs.sort(key=lambda t: t[0])

        dposes_pred = np.stack([p[1] for p in pairs], axis=0)
        dposes_gt   = np.stack([p[2] for p in pairs], axis=0)
        idxs        = np.array([p[0] for p in pairs], dtype=int)

        dq = torch.from_numpy(dposes_pred[:, 3:7])
        if len(idxs) > 1:
            breaks = np.where(np.diff(idxs) != 1)[0] + 1
        else:
            breaks = np.array([], dtype=int)
        starts = [0] + breaks.tolist()
        ends   = breaks.tolist() + [len(idxs)]
        for s, e in zip(starts, ends):
            dq[s:e] = enforce_quat_continuity(dq[s:e])
        dposes_pred[:, 3:7] = dq.numpy()

        traj_pred = compose_trajectory_quat_se3patch(dposes_pred)
        traj_gt   = compose_trajectory_quat_se3patch(dposes_gt)
        xyz_pred, xyz_gt = traj_pred[:, :3], traj_gt[:, :3]
        # ---- PLOT TRAIETTORIA (come originale: vista dall'alto XY) ----
        # --- opzionale: ancoraggio al mondo per il PLOT ---
        xyz_pred_plot = xyz_pred
        xyz_gt_plot   = xyz_gt
        title_suffix  = "IMU frame"

        if plot_frame == "world":
            # Indice k nella GT del primo Δpose del blocco contiguo
            ks = [ds.samples[i][1] for i in idxs_scene]
            first_k = ks[0]

            # Posa assoluta iniziale (IMU→World) dalla GT
            T0 = ds.gt_T_list[sid][first_k]
            R0 = T0[:3, :3].astype(np.float64)
            t0 = T0[:3, 3].astype(np.float64)

            # Trasforma le traiettorie locali (IMU frame) nel mondo
            xyz_gt_plot   = (R0 @ xyz_gt.T).T   + t0
            xyz_pred_plot = (R0 @ xyz_pred.T).T + t0
            title_suffix  = f"world frame (anchored to GT@k={first_k})"
# ---------------------------------------------------

        seq_out = os.path.join(out_root, seqname)
        os.makedirs(seq_out, exist_ok=True)
        plt.figure(figsize=(6, 6))
        plt.plot(xyz_gt_plot[:, 0],   xyz_gt_plot[:, 1],   label="GT",   linewidth=2)
        plt.plot(xyz_pred_plot[:, 0], xyz_pred_plot[:, 1], label="Pred", linestyle="--")
        plt.scatter([xyz_gt_plot[0, 0]], [xyz_gt_plot[0, 1]], s=18, marker="o", label="Start", zorder=3)
        ax = plt.gca(); ax.set_aspect("equal", adjustable="box")
        plt.xlabel("X [m]"); plt.ylabel("Y [m]")
        plt.title(f"{seqname} — Trajectory ({title_suffix})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(loc="best")
        plt.tight_layout()

        # salvo con due nomi per massima compatibilità con il tuo flusso
        plt.savefig(os.path.join(seq_out, "traj.png"),   dpi=200)
        plt.savefig(os.path.join(seq_out, "traj_xy.png"), dpi=200)
        plt.close()
        # ---- fine plot ----


        ate3d_raw = ate3d_noalign(xyz_pred, xyz_gt)
        ate2d_raw = ate2d_noalign(xyz_pred, xyz_gt)

        yaw_pred = np.array(
            [yaw_from_quat(qw, qx, qy, qz) for (tx, ty, tz, qx, qy, qz, qw) in traj_pred],
            dtype=np.float64
        )
        yaw_gt = np.array(
            [yaw_from_quat(qw, qx, qy, qz) for (tx, ty, tz, qx, qy, qz, qw) in traj_gt],
            dtype=np.float64
        )
        T_errs_100m, R_errs_deg_100m = kitti_drift(xyz_pred, yaw_pred, xyz_gt, yaw_gt)
        T_errs_per_m     = [float(v) / 100.0 for v in T_errs_100m]
        R_errs_deg_per_m = [float(v) / 100.0 for v in R_errs_deg_100m]

        seq_out = os.path.join(out_root, seqname)
        os.makedirs(seq_out, exist_ok=True)
        np.savez(os.path.join(seq_out, "traj_pred_gt.npz"),
                 traj_pred=traj_pred, traj_gt=traj_gt)
        per_seq = {
            "ATE3D_raw_rmse_m": float(ate3d_raw),
            "ATE2D_raw_rmse_m": float(ate2d_raw),
            "translation_drift_percent_100-800m": [float(v) for v in T_errs_100m],
            "rotation_drift_deg_per_100m_100-800m": [float(v) for v in R_errs_deg_100m],
            "translation_drift_percent_per_m_100-800m": T_errs_per_m,
            "rotation_drift_deg_per_m_100-800m": R_errs_deg_per_m
        }
        with open(os.path.join(seq_out, "metrics.json"), "w") as f:
            json.dump(per_seq, f, indent=2)

        print(f"[EVAL {split}] {seqname}: ATE2D_raw = {ate2d_raw:.3f} m | ATE3D_raw = {ate3d_raw:.3f} m")

        all_metrics[seqname] = per_seq
        ate3d_list.append(float(ate3d_raw))
        ate2d_list.append(float(ate2d_raw))
        if len(T_errs_per_m) > 0:
            drift_t_list.extend(T_errs_per_m)
        if len(R_errs_deg_per_m) > 0:
            drift_r_list.extend(R_errs_deg_per_m)

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

# ============================
# CLI
# ============================
def parse_seqs_arg(seqs_arg):
    return [s.strip() for s in seqs_arg.split(",") if s.strip()]

def main():
    ap = argparse.ArgumentParser("IMU Seq2Seq LSTM — two-head + quaternion + adaptive loss (SE3 patch) — IMU frame")
    ap.add_argument("--mode", choices=["train","eval"], required=True)
    ap.add_argument("--seqs", required=True, help='es: "01,02,03"')
    ap.add_argument("--output_root", default="outputs/imu_seq2seq_lstm_imustep_twohead_quat_adaloss_rmse_se3patch")
    ap.add_argument("--checkpoint", default="", help="usato in --mode eval")
    ap.add_argument("--eval_split", choices=["train","val","test","both"], default="val")
    ap.add_argument("--cap_train_windows_per_scene", type=int, default=None)
    ap.add_argument("--wq_init", type=float, default=-2.5, help="init log-var rotazione")
    ap.add_argument("--wt_init", type=float, default=0.0, help="init log-var traslazione")
    ap.add_argument("--plot_frame", choices=["imu","world"], default="imu",
                help="Solo per il PLOT: 'imu' come ora, oppure 'world' ancorato alla GT assoluta del pezzo")

    args = ap.parse_args()

    global CAP_TRAIN_WINDOWS_PER_SCENE
    CAP_TRAIN_WINDOWS_PER_SCENE = args.cap_train_windows_per_scene

    seqs = parse_seqs_arg(args.seqs)
    if args.mode == "train":
        train_and_validate(seqs, args.output_root, wq_init=args.wq_init, wt_init=args.wt_init)
    else:
        if not args.checkpoint:
            raise ValueError("In eval serve --checkpoint")
        if args.eval_split == "both":
            res_val  = evaluate_split_per_scene_se3patch(seqs, args.checkpoint, args.output_root, split="val",plot_frame=args.plot_frame)
            res_test = evaluate_split_per_scene_se3patch(seqs, args.checkpoint, args.output_root, split="test",plot_frame=args.plot_frame)
            with open(os.path.join(args.output_root, "eval_summary_val_test_se3patch.json"), "w") as f:
                json.dump({"val":res_val, "test":res_test}, f, indent=2)
            print("[SUMMARY] salvato eval_summary_val_test_se3patch.json")
        else:
            evaluate_split_per_scene_se3patch(seqs, args.checkpoint, args.output_root, split=args.eval_split)

if __name__ == "__main__":
    main()
