#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A2DO-style Adaptive Degradation Feature Filter (Temporal + Spatial)
=================================================================

This file implements the hierarchical feature filtering module described in
"A2DO: Adaptive Anti-Degradation Odometry with Deep Multi-Sensor ..." :

1) Temporal Feature Filter (Eq. 4-5 in the paper):
   - For each modality feature X_t (e.g., LiDAR / Visual / IMU), compute a
     2-class probability p_t ∈ R^2 (discard vs retain) using Multi-Head Attention
     with:
         Q = X_t,   K = h_{t-1},   V = h_{t-1}
   - Sample a discrete decision d_t ∈ {0,1} via Gumbel-Softmax (d=1 retain).
   - Apply coarse gating: d_t * X_t
   - Concatenate retained (gated) modality features to form F_t.

2) Spatial Feature Filter (Eq. 6 in the paper):
   - Apply self-attention with Q=K=V=F_t across feature *channels* to produce
     a retain/discard probability per channel P_c.
   - Sample a discrete channel decision D_c via Gumbel-Softmax.
   - Apply fine channel-wise gating: F_c = F_t ⊗ D_c.

Note:
- The paper uses three modalities (LiDAR, Visual, IMU). Here we provide a
  generic N-modality implementation and a convenient 2-modality wrapper
  (e.g., radar-camera fused + IMU).
- To be faithful to the paper, the Temporal Feature Filter requires the
  decoder hidden state from the *previous* step, h_{t-1}. In practice, you
  can pass a tensor h_prev of shape [B, L, H] where h_prev[:, t] = h_{t-1}.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_padding_mask_keep_zero(
    keep: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    keep: [B, L, ...] in {0,1} (or [0,1])
    key_padding_mask: [B, L] with True on padded positions.
    If padded => force keep=0.
    """
    if key_padding_mask is None:
        return keep
    # key_padding_mask True means PAD => set keep to 0
    pad = key_padding_mask.to(dtype=keep.dtype)
    while pad.ndim < keep.ndim:
        pad = pad.unsqueeze(-1)
    return keep * (1.0 - pad)


class TemporalFeatureFilterA2DO(nn.Module):
    """
    Temporal Feature Filter (coarse), faithful to Eq.(4)-(5) in A2DO.

    For each modality m:
      p_{m,t} = MHA(Q = x_{m,t}, K = h_{t-1}, V = h_{t-1})  -> 2 logits
      d_{m,t} ~ GumbelSoftmax(p_{m,t})                      -> {0,1} (retain=1)
      x'_{m,t} = d_{m,t} * x_{m,t}

    Then:
      F_t = concat_m x'_{m,t}

    Inputs
    ------
    xs: list of modality tensors, each [B, L, C_m]
    h_prev: [B, L, H] where h_prev[:, t] is h_{t-1}
    key_padding_mask: optional [B, L] True on PAD positions
    """

    def __init__(
        self,
        in_dims: List[int],
        hidden_dim: int,
        attn_dim: int = 128,
        num_heads: int = 4,
        tau: float = 1.0,
        hard_gumbel: bool = True,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"
        self.in_dims = list(in_dims)
        self.hidden_dim = int(hidden_dim)
        self.attn_dim = int(attn_dim)
        self.num_heads = int(num_heads)
        self.tau = float(tau)
        self.hard_gumbel = bool(hard_gumbel)

        # One query projection per modality (Q comes from x_{m,t})
        self.q_proj = nn.ModuleList([nn.Linear(d, attn_dim) for d in self.in_dims])

        # Shared key/value projection from h_{t-1}
        self.k_proj = nn.Linear(hidden_dim, attn_dim)
        self.v_proj = nn.Linear(hidden_dim, attn_dim)

        # One MHA per modality, exactly as Eq.(4): Q from modality, K/V from h_{t-1}
        self.mha = nn.ModuleList(
            [
                nn.MultiheadAttention(attn_dim, num_heads, dropout=attn_dropout, batch_first=True)
                for _ in self.in_dims
            ]
        )

        # Map attended output to 2-class logits (discard vs retain) per modality
        self.logit_head = nn.ModuleList([nn.Linear(attn_dim, 2) for _ in self.in_dims])

    @staticmethod
    def _sample_retain_from_logits(
        logits: torch.Tensor, tau: float, hard: bool
    ) -> torch.Tensor:
        """
        logits: [..., 2]
        Returns retain indicator in {0,1} (or soft in [0,1]) with convention:
          class 0 = discard, class 1 = retain
        """
        probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1) #apply GumbelSoftmax
        retain = probs[..., 1]  # retain
        return retain

    def forward(
        self,
        xs: List[torch.Tensor], #list feature tensor (one for modality)
        h_prev: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
        warmup_keep_prob: Optional[float] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Returns
        -------
        Ft: [B, L, sum(C_m)] concatenated gated features (Eq. 5)
        ds: list of retain indicators d_{m,t} with shape [B, L, 1]
        logits_list: list of 2-class logits per modality [B, L, 2]
        """
        assert isinstance(xs, list) and len(xs) == len(self.in_dims), "xs must match in_dims length"
        B, L = xs[0].shape[0], xs[0].shape[1]
        for x, d in zip(xs, self.in_dims):
            assert x.shape[:2] == (B, L), "All modalities must share [B,L]"
            assert x.shape[2] == d, f"Expected C={d}, got {x.shape[2]}"
        assert h_prev.shape[:2] == (B, L), "h_prev must be [B,L,H]"
        assert h_prev.shape[2] == self.hidden_dim, f"Expected H={self.hidden_dim}, got {h_prev.shape[2]}"

        t = self.tau if tau is None else float(tau)

        # Project K,V from h_{t-1}. Shape [B,L,attn_dim]
        K = self.k_proj(h_prev)
        V = self.v_proj(h_prev)

        gated_feats: List[torch.Tensor] = []
        ds: List[torch.Tensor] = []
        logits_list: List[torch.Tensor] = []

        for m, x in enumerate(xs):
            Q = self.q_proj[m](x)                 # [B,L,attn_dim]

            
            Q1 = Q.reshape(B * L, 1, self.attn_dim)   # [B*L,1,attn_dim]
            K1 = K.reshape(B * L, 1, self.attn_dim)   # [B*L,1,attn_dim]
            V1 = V.reshape(B * L, 1, self.attn_dim)   # [B*L,1,attn_dim]

            kpm1 = None
            if key_padding_mask is not None:
                # key_padding_mask: [B,L] -> [B*L,1]
                kpm1 = key_padding_mask.reshape(B * L, 1)

            attn_out1, _ = self.mha[m](Q1, K1, V1, key_padding_mask=kpm1)  # [B*L,1,attn_dim] multi-head attention
            attn_out = attn_out1.reshape(B, L, self.attn_dim)              # [B,L,attn_dim]

            logits = self.logit_head[m](attn_out)  # [B,L,2]
            if warmup_keep_prob is not None:
                # Warmup: maschera soft costante (più stabile del sampling)
                retain = torch.full(
                    (B, L),
                    float(warmup_keep_prob),
                    device=logits.device,
                    dtype=logits.dtype,
                )
            else:
                retain = self._sample_retain_from_logits(logits, tau=t, hard=self.hard_gumbel)  # [B,L]
            retain = retain.unsqueeze(-1)  # [B,L,1]

            retain = _apply_padding_mask_keep_zero(retain, key_padding_mask)
            x_gated = x * retain  # [B,L,C_m]

            gated_feats.append(x_gated)
            ds.append(retain)
            logits_list.append(logits)

        Ft = torch.cat(gated_feats, dim=-1)  # Eq.(5)
        return Ft, ds, logits_list


class SpatialFeatureFilterA2DO(nn.Module):
    """
    Spatial Feature Filter (fine), faithful to Eq.(6) in A2DO.

    Given Ft ∈ R^{B×L×C}, produce per-channel retain/discard decision using
    self-attention across channels.

    Implementation detail:
    - Treat channels as "tokens": sequence length = C
    - Embed each scalar channel value into attn_dim via a linear layer
    - Apply self-attention over channel tokens (Q=K=V=embedded)
    - Produce 2-class logits per channel, sample retain indicator via Gumbel-Softmax
    """

    def __init__(
        self,
        feat_dim: int,
        attn_dim: int = 128,
        num_heads: int = 4,
        tau: float = 1.0,
        hard_gumbel: bool = True,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"
        self.feat_dim = int(feat_dim)
        self.attn_dim = int(attn_dim)
        self.num_heads = int(num_heads)
        self.tau = float(tau)
        self.hard_gumbel = bool(hard_gumbel)

        # Embed scalar channel value -> attn_dim
        self.embed = nn.Linear(1, attn_dim)
        self.mha = nn.MultiheadAttention(attn_dim, num_heads, dropout=attn_dropout, batch_first=True)

        # Produce 2-class logits per channel token
        self.logit_head = nn.Linear(attn_dim, 2)

    @staticmethod
    def _sample_retain_from_logits(
        logits: torch.Tensor, tau: float, hard: bool
    ) -> torch.Tensor:
        probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        retain = probs[..., 1]  # convention: class 1 = retain
        return retain

    def forward(
        self,
        Ft: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
        warmup_keep_prob: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ft: [B, L, C]
        key_padding_mask: [B, L] True on PAD
        Returns:
          Fc: [B, L, C] fine-filtered features (Eq. 6)
          Dc: [B, L, C] retain mask per channel (0/1)
          logits_c: [B, L, C, 2] channel logits
        """
        B, L, C = Ft.shape
        assert C == self.feat_dim, f"Expected C={self.feat_dim}, got {C}"

        t = self.tau if tau is None else float(tau)

        # Reshape to treat channels as tokens:
        # For each (B,L) sample, we have a length-C sequence of scalar tokens.
        tokens = Ft.reshape(B * L, C, 1)                 # [B*L, C, 1]
        tokens = self.embed(tokens)                      # [B*L, C, attn_dim]
        attn_out, _ = self.mha(tokens, tokens, tokens)   # [B*L, C, attn_dim]

        logits_c = self.logit_head(attn_out)             # [B*L, C, 2]
        if warmup_keep_prob is not None:
            # Warmup: maschera soft costante su tutti i canali
            retain = torch.full(
                (B, L, C),
                float(warmup_keep_prob),
                device=logits_c.device,
                dtype=logits_c.dtype,
            )
        else:
            retain = self._sample_retain_from_logits(logits_c, tau=t, hard=self.hard_gumbel)  # [B*L, C]
            retain = retain.view(B, L, C)                    # [B,L,C]

        # If padded timesteps, force retain=0 on those timesteps (all channels)
        retain = _apply_padding_mask_keep_zero(retain, key_padding_mask)

        Fc = Ft * retain                                 # Eq.(6): Ft ⊗ Dc
        logits_c = logits_c.view(B, L, C, 2)
        return Fc, retain, logits_c


class A2DOHierarchicalFilter(nn.Module):
    """
    Full hierarchical filter = Temporal Feature Filter -> Spatial Feature Filter,
    matching the A2DO paper order.
    """

    def __init__(
        self,
        in_dims: List[int],
        hidden_dim: int,
        temporal_attn_dim: int = 128,
        spatial_attn_dim: int = 128,
        num_heads: int = 4,
        tau_temporal: float = 1.0,
        tau_spatial: float = 1.0,
        hard_gumbel: bool = True,
        attn_dropout: float = 0.0,
        out_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_dims = list(in_dims)
        self.hidden_dim = int(hidden_dim)

        self.temporal = TemporalFeatureFilterA2DO(
            in_dims=in_dims,
            hidden_dim=hidden_dim,
            attn_dim=temporal_attn_dim,
            num_heads=num_heads,
            tau=tau_temporal,
            hard_gumbel=hard_gumbel,
            attn_dropout=attn_dropout,
        )

        feat_dim = int(sum(in_dims))
        self.spatial = SpatialFeatureFilterA2DO(
            feat_dim=feat_dim,
            attn_dim=spatial_attn_dim,
            num_heads=num_heads,
            tau=tau_spatial,
            hard_gumbel=hard_gumbel,
            attn_dropout=attn_dropout,
        )

        self.out_dim = out_dim
        self.proj_out = nn.Linear(feat_dim, out_dim) if out_dim is not None else None

    def forward(
        self,
        xs: List[torch.Tensor],
        h_prev: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        tau_temporal: Optional[float] = None,
        tau_spatial: Optional[float] = None,
        warmup_keep_prob: Optional[float] = None
    ):
        """
        Returns a dict with all intermediate signals (useful for logging/analysis).
        """
        Ft, ds, logits_m = self.temporal(
            xs=xs,
            h_prev=h_prev,
            key_padding_mask=key_padding_mask,
            tau=tau_temporal,
            warmup_keep_prob=warmup_keep_prob,
        )
        Fc, Dc, logits_c = self.spatial(
            Ft=Ft,
            key_padding_mask=key_padding_mask,
            tau=tau_spatial,
            warmup_keep_prob=warmup_keep_prob,
        )

        y = self.proj_out(Fc) if self.proj_out is not None else Fc
        # ---- NOVELTY: policy distributions (pi) for latent decision process ----
        pi_m = [F.softmax(lg, dim=-1) for lg in logits_m]   # list of [B,L,2]
        pi_c = F.softmax(logits_c, dim=-1)                  # [B,L,C,2]


        return {
            "y": y,                 # [B,L,out_dim] if out_dim else [B,L,sumC]
            "Ft": Ft,               # temporally gated concatenation (Eq.5)
            "Fc": Fc,               # spatially gated features (Eq.6)
            "d_modalities": ds,     # list of [B,L,1] (retain per modality per time)
            "Dc_channels": Dc,      # [B,L,C] (retain per channel)
            "logits_modalities": logits_m,  # list of [B,L,2]
            "logits_channels": logits_c,    # [B,L,C,2]
            "pi_modalities": pi_m,   # list of [B,L,2]  (policy)
            "pi_channels": pi_c,     # [B,L,C,2]

        }


class A2DOHierarchicalFilter2Modalities(nn.Module):
    """
    Convenience wrapper for 2 modalities, e.g.:
      - modality 1: radar-camera fused feature sequence
      - modality 2: IMU feature sequence
    """

    def __init__(
        self,
        dim_mod1: int,
        dim_mod2: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        temporal_attn_dim: int = 128,
        spatial_attn_dim: int = 128,
        num_heads: int = 4,
        tau_temporal: float = 1.0,
        tau_spatial: float = 1.0,
        hard_gumbel: bool = True,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.filter = A2DOHierarchicalFilter(
            in_dims=[dim_mod1, dim_mod2],
            hidden_dim=hidden_dim,
            temporal_attn_dim=temporal_attn_dim,
            spatial_attn_dim=spatial_attn_dim,
            num_heads=num_heads,
            tau_temporal=tau_temporal,
            tau_spatial=tau_spatial,
            hard_gumbel=hard_gumbel,
            attn_dropout=attn_dropout,
            out_dim=out_dim,
        )

    def forward(
        self,
        x_mod1: torch.Tensor,
        x_mod2: torch.Tensor,
        h_prev: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        tau_temporal: Optional[float] = None,
        tau_spatial: Optional[float] = None,
        warmup_keep_prob: Optional[float] = None,
    ):
        return self.filter(
            xs=[x_mod1, x_mod2],
            h_prev=h_prev,
            key_padding_mask=key_padding_mask,
            tau_temporal=tau_temporal,
            tau_spatial=tau_spatial,
            warmup_keep_prob=warmup_keep_prob,
        )
