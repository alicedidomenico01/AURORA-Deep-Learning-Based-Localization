# arcfm_monoscale.py
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveRadarCameraFusionMono(nn.Module):
    """
    A-RCFM monoscala (ispirato a 4DRVO-Net, sez. III-B) per fusione 4D Radar + Camera.

    Input:
      - xyz_imu:        (B,N,3)  punti radar nel frame IMU (metri / scala coerente con calibrazione)
      - feat_pc:        (B,N,Cp) feature per punto 4D radar (es. output encoder Radar4DEncoderPN2)
      - img_feat:       (B,Ci,Hf,Wf) feature map 2D dell'immagine (ResNet18)
      - K:              (B,3,3) intrinseche camera
      - T_cam_from_imu: (B,4,4) trasformazione SE(3) dal frame IMU al frame camera
      - img_size:       (B,2) [H_img, W_img] dimensioni originali dell'immagine
    Parametri:
      - pc_feat_dim:  Cp (dim delle feature per punto in ingresso)
      - img_feat_dim: Ci (dim dei canali della fmap immagine)
      - out_dim:      C_out (dim delle feature fuse in uscita; default = Cp)
      - d_model:      dim. spazio di attenzione (per Q,K,V)
      - n_heads:      numero di head per MultiheadAttention (d_model dev’essere divisibile per n_heads)
      - n_samples:    K (numero di sample deformabili per punto)
      - stride:       stride spaziale dell’encoder immagine (8 o 16)

    Output:
      - feat_fused: (B,N,out_dim) feature fuse F ( FPC + FI←P proiettato )
    """

    def __init__(
        self,
        pc_feat_dim: int,
        img_feat_dim: int,
        out_dim: Optional[int] = None,
        d_model: int = 128,
        n_heads: int = 4,
        n_samples: int = 8,
        stride: int = 8,
        weather_emb_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model deve essere divisibile per n_heads"
        assert n_samples > 0, "n_samples (K) deve essere > 0"
        assert stride in (8, 16, 32), "stride atteso 8 o 16 (come ImageBackboneResNet18)"

        self.pc_feat_dim = pc_feat_dim
        self.img_feat_dim = img_feat_dim
        self.out_dim = out_dim if out_dim is not None else pc_feat_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_samples = n_samples
        self.stride = stride

        # --- Proiezione radar feature -> query Q_P ---
        self.lin_q = nn.Linear(pc_feat_dim, d_model)

        # --- Offset e pesi per campionamento deformabile (Δp_k, A_k) ---
        # Δp_k: per ogni punto una coppia (dx, dy) per ciascuno dei K sample → 2*K dimensioni
        self.lin_offset = nn.Linear(d_model, 2 * n_samples)
        # A_k: K pesi (softmax) per combinazione pesata dei campioni immagine
        self.lin_alpha = nn.Linear(d_model, n_samples)

        # --- Proiezione feature immagine aggregate per ottenere -> Key,Value
        self.lin_k = nn.Linear(img_feat_dim, d_model)
        self.lin_v = nn.Linear(img_feat_dim, d_model)

        # --- Cross-attention Multi-head (batch_first=True) 
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )

        # --- MLP finale per fondere [FPC ; FI←P] -> F_out 
        #concateno radar e contributo camera e poi passo in un piccolo MPL
        self.mlp_fuse = nn.Sequential(
            nn.Linear(pc_feat_dim + d_model, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, self.out_dim),
        )
                # --- (OPZIONALE) FiLM controllata da weather embedding ---
        self.weather_emb_dim = weather_emb_dim
        if self.weather_emb_dim is not None:
            # gamma, beta: (B, Ci) -> broadcast su (B, Ci, Hf, Wf) scaling and shift
            self.gamma_mlp = nn.Sequential(
                nn.Linear(self.weather_emb_dim, self.img_feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.img_feat_dim, self.img_feat_dim),
            )
            self.beta_mlp = nn.Sequential(
                nn.Linear(self.weather_emb_dim, self.img_feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.img_feat_dim, self.img_feat_dim),
            )


    @staticmethod
    def _project_points_to_image(
        xyz_imu: torch.Tensor,
        K: torch.Tensor,
        T_cam_from_imu: torch.Tensor,
        img_size: torch.Tensor,
        stride: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Proietta punti dal frame IMU alla feature map dell'immagine.

        xyz_imu:        (B,N,3)
        K:              (B,3,3)
        T_cam_from_imu: (B,4,4)
        img_size:       (B,2) [H_img, W_img]
        stride:         int (8 o 16)

        Ritorna:
          - u_feat: (B,N) coordinate x sulla fmap (0..Wf-1)
          - v_feat: (B,N) coordinate y sulla fmap (0..Hf-1)
        """
        device = xyz_imu.device
        dtype = xyz_imu.dtype

        B, N, _ = xyz_imu.shape
        H_img = img_size[:, 0].to(device=device, dtype=dtype)  # (B,)
        W_img = img_size[:, 1].to(device=device, dtype=dtype)  # (B,)

        # Costruisci omogenee [x,y,z,1]
        ones = torch.ones(B, N, 1, device=device, dtype=dtype)
        xyz_h = torch.cat([xyz_imu, ones], dim=-1)  # (B,N,4)

        # Porta nel frame camera: P_cam = T_cam_from_imu * P_imu
        T = T_cam_from_imu.to(device=device, dtype=dtype)      # (B,4,4)
        K = K.to(device=device, dtype=dtype)                    # (B,3,3)

        # (B,N,4) @ (B,4,4)^T -> (B,N,4)
        xyz_cam = torch.matmul(xyz_h, T.transpose(1, 2))       # (B,N,4)
        xyz_cam = xyz_cam[..., :3]                             # (B,N,3)

        Xc = xyz_cam[..., 0]
        Yc = xyz_cam[..., 1]
        Zc = xyz_cam[..., 2]

        eps = 1e-6

        
        Z_front = -Zc

        # Evita divisioni per zero o valori negativi/zero
        Z_safe = torch.where(Z_front > eps,
                             Z_front,
                             torch.full_like(Z_front, eps))

        # Proiezione pinhole con profondità Z_front
        
        xyz_cam_for_proj = torch.stack([Xc, Yc, Z_front], dim=-1)  # (B,N,3)

        cam_pts = torch.matmul(xyz_cam_for_proj, K.transpose(1, 2))  # (B,N,3)
        u = cam_pts[..., 0] / Z_safe   # (B,N)
        v = cam_pts[..., 1] / Z_safe   # (B,N)


        # Scala alle dimensioni della feature map
        # Hf = H_img / stride, Wf = W_img / stride
        Hf = H_img / float(stride)  # (B,)
        Wf = W_img / float(stride)  # (B,)

        # Evita divide-by-zero nel caso limite
        Hf = torch.clamp(Hf, min=1.0)
        Wf = torch.clamp(Wf, min=1.0)

        # u,v sono in pixel dell'immagine originale, portali su [0, Wf-1] e [0, Hf-1]
        # attenzione alle broadcast: (B,N) * (B,1)
        u_feat = u * (Wf.view(B, 1) / W_img.view(B, 1))  # (B,N)
        v_feat = v * (Hf.view(B, 1) / H_img.view(B, 1))  # (B,N)

        return u_feat, v_feat

    def forward(
        self,
        xyz_imu: torch.Tensor,         # (B,N,3)
        feat_pc: torch.Tensor,         # (B,N,Cp)
        img_feat: torch.Tensor,        # (B,Ci,Hf,Wf)
        K: torch.Tensor,               # (B,3,3)
        T_cam_from_imu: torch.Tensor,  # (B,4,4)
        img_size: torch.Tensor,        # (B,2) [H_img, W_img]
        weather_emb: Optional[torch.Tensor] = None,
        weather_gate: Optional[torch.Tensor] = None,  # (B,1) o (B,)
    ) -> torch.Tensor:
        B, N, Cp = feat_pc.shape
        _, Ci, Hf, Wf = img_feat.shape
        assert Cp == self.pc_feat_dim, f"Cp={Cp} != pc_feat_dim={self.pc_feat_dim}"
        assert Ci == self.img_feat_dim, f"Ci={Ci} != img_feat_dim={self.img_feat_dim}"

        device = feat_pc.device
        dtype = feat_pc.dtype

                # (OPZIONALE) Modula la feature map immagine in base al weather embedding
        if (self.weather_emb_dim is not None) and (weather_emb is not None):
            # weather_emb: (B, D_w)
            gamma = self.gamma_mlp(weather_emb).view(B, Ci, 1, 1)  # (B,Ci,1,1)
            beta  = self.beta_mlp(weather_emb).view(B, Ci, 1, 1)   # (B,Ci,1,1)
            # FiLM: F'_img = (1 + gamma) * F_img + beta
            img_feat = img_feat * (1.0 + gamma) + beta


        # 1) Proiezione punti radar su fmap immagine (coordinate continue)
        u_feat, v_feat = self._project_points_to_image(
            xyz_imu=xyz_imu,
            K=K,
            T_cam_from_imu=T_cam_from_imu,
            img_size=img_size,
            stride=self.stride,
        )  # (B,N), (B,N)

        # 2) Costruisci query Q_P dalle feature per punto (F_PC)
        Qp = self.lin_q(feat_pc)  # (B,N,d_model)

        # 3) Offset e pesi per campionamento deformabile
        # Δp_k: (B,N,2K) -> (B,N,K,2)
        offsets = self.lin_offset(Qp).view(B, N, self.n_samples, 2)  # (dx,dy)
        # A_k: (B,N,K) con softmax su K
        alpha = self.lin_alpha(Qp).view(B, N, self.n_samples)
        alpha = F.softmax(alpha, dim=-1)  # (B,N,K)

        # 4) Costruisci coordinate (u_k, v_k) = p + Δp_k sulla fmap
        # u_feat, v_feat: (B,N) → (B,N,1) per broadcast
        u0 = u_feat.unsqueeze(-1)  # (B,N,1)
        v0 = v_feat.unsqueeze(-1)  # (B,N,1)

        # offsets[..., 0] = Δx, offsets[..., 1] = Δy
        u_samples = u0 + offsets[..., 0]  # (B,N,K)
        v_samples = v0 + offsets[..., 1]  # (B,N,K)

        # 5) Normalizza le coord. per grid_sample: [-1,1]
        # x_norm = 2*u/(Wf-1)-1 , y_norm = 2*v/(Hf-1)-1
        Wf_f = float(Wf)
        Hf_f = float(Hf)
        # Evita casi degeneri
        Wf_minus1 = max(Wf_f - 1.0, 1.0)
        Hf_minus1 = max(Hf_f - 1.0, 1.0)

        x_norm = 2.0 * (u_samples / Wf_minus1) - 1.0  # (B,N,K)
        y_norm = 2.0 * (v_samples / Hf_minus1) - 1.0  # (B,N,K)

        # Clamping per sicurezza
        x_norm = torch.clamp(x_norm, -1.5, 1.5)
        y_norm = torch.clamp(y_norm, -1.5, 1.5)

        # grid per grid_sample: (B,1,N*K,2)
        grid = torch.stack([x_norm, y_norm], dim=-1)  # (B,N,K,2)
        grid = grid.view(B, 1, N * self.n_samples, 2)  # (B,1,N*K,2)

        # 6) Sample della fmap immagine FI(p + Δp_k) via bilinear sampling
        # img_feat: (B,Ci,Hf,Wf)
        # out: (B,Ci,1,N*K) → reshape (B,N,K,Ci)
        sampled = F.grid_sample(
            img_feat,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # (B,Ci,1,N*K)

        sampled = sampled.view(B, Ci, N, self.n_samples)       # (B,Ci,N,K)
        sampled = sampled.permute(0, 2, 3, 1).contiguous()     # (B,N,K,Ci) = F_I^k

        # 7) Aggregazione pesata F̂_I^p = Σ_k A_k F_I^k
        alpha_exp = alpha.unsqueeze(-1)                         # (B,N,K,1)
        F_agg = (alpha_exp * sampled).sum(dim=2)                # (B,N,Ci)

        # 8) K,V dalle feature immagine aggregate
        K_I = self.lin_k(F_agg)                                 # (B,N,d_model)
        V_I = self.lin_v(F_agg)                                 # (B,N,d_model)

        # 9) Cross-attention: QP, KI, VI
        # nn.MultiheadAttention con batch_first=True vuole (B,N,E)
        attn_out, _ = self.cross_attn(
            query=Qp,   # (B,N,d_model)
            key=K_I,    # (B,N,d_model)
            value=V_I,  # (B,N,d_model)
        )                                                   # -> (B,N,d_model), contributo "da camera"

        # 9.b) Gating meteo sul contributo della camera
        if weather_gate is not None:
            # weather_gate atteso come (B,1) o (B,)
            g = weather_gate.view(-1, 1, 1)      # (B,1,1)
            # di sicurezza: forziamo a [0,1]
            g = torch.clamp(g, 0.0, 1.0)
            attn_out = g * attn_out              # se g=0 -> ignora la camera

        # 10) Fusione finale [FPC ; FI←P] -> F_out
        feat_cat = torch.cat([feat_pc, attn_out.to(dtype)], dim=-1)  # (B,N,Cp + d_model)
        feat_fused = self.mlp_fuse(feat_cat)                         # (B,N,out_dim)

        return feat_fused


