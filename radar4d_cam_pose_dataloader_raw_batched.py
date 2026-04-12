# -*- coding: utf-8 -*-
"""
radar4d_cam_pose_dataloader_raw_batched.py

Collate/loader per Radar+Camera:
- impacchetta punti/feature 
- immagini: stack (P,3,H,W) se tutte uguali, altrimenti lista
- K e T_cam_from_imu per pair

API:
  from radar4d_cam_pose_dataloader_raw import SeqConfigCam
  from radar4d_cam_pose_dataloader_raw_batched import make_loaders_radar_cam_pose_raw_batched
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import List, Tuple, Dict

from radar4d_cam_pose_dataloader_raw import (
    Radar4DPlusCamDatasetPairsRAW, SeqConfigCam
)

__all__ = ["pad_collate_radar_cam_raw_batched", "make_loaders_radar_cam_pose_raw_batched"]


def pad_collate_radar_cam_raw_batched(batch: List[Tuple]):
    """
    batch elem:
      (pairs:list[L] di (xyz_t, feat_t, xyz_t1, feat_t1), y(7), sid(int), L(int), extra:dict)

    Ritorna:
      pairs_batched: dict con Xt, Ft, Xt1, Ft1, validN_t, pair_b, pair_t
      cam: dict con I_t/I_t1, K, T_cam_from_imu, img_size
      y_out: (B,7)
      sid_o: (B,)
      lengths:(B,)
    """
    B = len(batch)
    pairs_list = [b[0] for b in batch]
    y_out = torch.stack([b[1] for b in batch], dim=0)
    sid_o = torch.tensor([b[2] for b in batch], dtype=torch.long)
    lengths = torch.tensor([b[3] for b in batch], dtype=torch.long)
    extras  = [b[4] for b in batch]

    pair_b, pair_t, Nt_list, Nt1_list = [], [], [], []
    Cin_t = None; Cin_t1 = None
    for b_idx, pairs in enumerate(pairs_list):
        for t_idx, (xyz_t, feat_t, xyz_t1, feat_t1) in enumerate(pairs):
            if Cin_t is None:  Cin_t  = int(feat_t.shape[1])
            if Cin_t1 is None: Cin_t1 = int(feat_t1.shape[1])
            pair_b.append(b_idx); pair_t.append(t_idx)
            Nt_list.append(xyz_t.shape[0]); Nt1_list.append(xyz_t1.shape[0])

    P = len(pair_b)
    if P == 0:
        pairs_batched = {
            "Xt": torch.zeros(0,1,3, dtype=torch.float32),
            "Ft": torch.zeros(0,1,(Cin_t or 1), dtype=torch.float32),
            "Xt1": torch.zeros(0,1,3, dtype=torch.float32),
            "Ft1": torch.zeros(0,1,(Cin_t1 or 1), dtype=torch.float32),
            "validN_t": torch.zeros(0, dtype=torch.long),
            "pair_b": torch.zeros(0, dtype=torch.long),
            "pair_t": torch.zeros(0, dtype=torch.long),
        }
        cam = {
            "I_t": [], "I_t1": [], "K": torch.zeros(0,3,3, dtype=torch.float64),
            "T_cam_from_imu": torch.zeros(0,4,4, dtype=torch.float64),
            "img_size": torch.zeros(0,2, dtype=torch.long),
        }
        return pairs_batched, cam, y_out, sid_o, lengths

    Nmax_t  = int(max(Nt_list))
    Nmax_t1 = int(max(Nt1_list))

    Xt  = torch.zeros(P, Nmax_t,  3,     dtype=torch.float32)
    Ft  = torch.zeros(P, Nmax_t,  Cin_t, dtype=torch.float32)
    Xt1 = torch.zeros(P, Nmax_t1, 3,     dtype=torch.float32)
    Ft1 = torch.zeros(P, Nmax_t1, Cin_t1,dtype=torch.float32)
    validN_t = torch.tensor(Nt_list, dtype=torch.long)

    I_t_list, I_t1_list = [], []
    H_list, W_list = [], []
    K_list, Tci_list = [], []
    weather_list = []
    illum_list   = []


    p = 0
    for b_idx, (pairs, extra) in enumerate(zip(pairs_list, extras)):
        for t_idx, (xyz_t, feat_t, xyz_t1, feat_t1) in enumerate(pairs):
            Ni, Ni1 = xyz_t.shape[0], xyz_t1.shape[0]
            Xt[p, :Ni,  :] = torch.from_numpy(xyz_t.astype(np.float32))
            Ft[p, :Ni,  :] = torch.from_numpy(feat_t.astype(np.float32))
            Xt1[p,:Ni1, :] = torch.from_numpy(xyz_t1.astype(np.float32))
            Ft1[p,:Ni1, :] = torch.from_numpy(feat_t1.astype(np.float32))
            if Ni < Nmax_t:
                Xt[p, Ni:, :] = Xt[p, Ni-1:Ni, :].expand(Nmax_t-Ni, -1)
                Ft[p, Ni:,  :] = Ft[p, Ni-1:Ni, :].expand(Nmax_t-Ni, -1)
            if Ni1 < Nmax_t1:
                Xt1[p, Ni1:, :] = Xt1[p, Ni1-1:Ni1, :].expand(Nmax_t1-Ni1, -1)
                Ft1[p, Ni1:, :] = Ft1[p, Ni1-1:Ni1, :].expand(Nmax_t1-Ni1, -1)

            I_t_list.append(extra["img_t"][t_idx])
            I_t1_list.append(extra["img_t1"][t_idx])
            H, W = extra["img_size"][t_idx]
            H_list.append(H); W_list.append(W)
            weather_list.append(float(extra["weather_sev"]))
            illum_list.append(float(extra["illum_sev"]))


            K_list.append(extra["K"].unsqueeze(0))                 # (1,3,3) float64
            Tci = extra["T_cam_from_imu"].unsqueeze(0)            # (1,4,4) float64
            if Tci.dtype != torch.float64:
                Tci = Tci.double()
            Tci_list.append(Tci)
            p += 1

    pairs_batched = {
        "Xt": Xt, "Ft": Ft, "Xt1": Xt1, "Ft1": Ft1,
        "validN_t": validN_t,
        "pair_b": torch.tensor(pair_b, dtype=torch.long),
        "pair_t": torch.tensor(pair_t, dtype=torch.long),
    }

    same_size = all((H == H_list[0] and W == W_list[0]) for H,W in zip(H_list, W_list))
    if same_size:
        I_t  = torch.stack(I_t_list,  dim=0)  # (P,3,H,W)
        I_t1 = torch.stack(I_t1_list, dim=0)
    else:
        I_t, I_t1 = I_t_list, I_t1_list

    cam = {
        "I_t": I_t,
        "I_t1": I_t1,
        "K": torch.cat(K_list, dim=0),                       # (P,3,3) float64
        "T_cam_from_imu": torch.cat(Tci_list, dim=0),       # (P,4,4) float64
        "img_size": torch.tensor(list(zip(H_list, W_list)), dtype=torch.long),
        "weather_sev": torch.tensor(weather_list, dtype=torch.float32),  # (P,)
        "illum_sev":   torch.tensor(illum_list,   dtype=torch.float32),  # (P,)

    }
    return pairs_batched, cam, y_out, sid_o, lengths


def make_loaders_radar_cam_pose_raw_batched(
        seqs: List[SeqConfigCam], batch_size=8, perc=(0.70,0.15,0.15),
        seed=0, num_workers=8):
    ds = Radar4DPlusCamDatasetPairsRAW(seqs)

    # split per-sequenza (mantiene l'ordine temporale)
    sid_to_indices: Dict[int, List[int]] = {sid: [] for sid in range(len(ds.seqs))}
    for gidx, (sid_i, _pair_idx) in enumerate(ds.samples):
        sid_to_indices[sid_i].append(gidx)

    idx_train, idx_val, idx_test = [], [], []
    for sid in range(len(ds.seqs)):
        idx_all = sid_to_indices[sid]
        n = len(idx_all)
        ntr = int(perc[0]*n); nva = int(perc[1]*n)
        tr = idx_all[:ntr]; va = idx_all[ntr:ntr+nva]; te = idx_all[ntr+nva:]
        idx_train.extend(tr); idx_val.extend(va); idx_test.extend(te)

    max_valid = len(ds) - 1
    def _sanitize(arr):
        return [i for i in arr if (0 <= i <= max_valid)]

    idx_train = _sanitize(idx_train)
    idx_val   = _sanitize(idx_val)
    idx_test  = _sanitize(idx_test)

    train_loader = DataLoader(
       Subset(ds, idx_train),
       batch_size=batch_size,
       shuffle=True,
       num_workers=num_workers,
       pin_memory=True,
       persistent_workers=(num_workers > 0),
       collate_fn=pad_collate_radar_cam_raw_batched,
    )
    val_loader = DataLoader(
       Subset(ds, idx_val),
       batch_size=batch_size,
       shuffle=False,
       num_workers=num_workers,
       pin_memory=True,
       persistent_workers=(num_workers > 0),
       collate_fn=pad_collate_radar_cam_raw_batched,
    )
    test_loader = DataLoader(
       Subset(ds, idx_test),
       batch_size=batch_size,
       shuffle=False,
       num_workers=num_workers,
       pin_memory=True,
       persistent_workers=(num_workers > 0),
       collate_fn=pad_collate_radar_cam_raw_batched,
    )

    return ds, train_loader, val_loader, test_loader
