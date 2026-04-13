#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
radar4d_cam_imu_pose_dataloader_raw_batched.py

Dataset batched che importa le funzioni da dataloader radar e radar-camera
Stessa logica, solo aggiunta e sincronizzazione IMU
Creazione finestra temporale per IMU tra coppia radar-camera

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# =========================
# IMPORT DATALOADER ESISTENTI
# =========================

from src.dataloaders.radar4d_cam_pose_dataloader_raw import (
    SeqConfigCam,
    Radar4DPlusCamDatasetPairsRAW
)

from src.dataloaders.radar4d_cam_pose_dataloader_raw_batched import (
    pad_collate_radar_cam_raw_batched
)


from src.models.imu_data_imustep_quat_localpatch import load_imu_fixed_csv


# ============================================================
# CONFIG ESTESA (aggiunge solo il path IMU)
# ============================================================

class SeqConfigCamImu(SeqConfigCam):
    def __init__(self, *args, imu_path=None, window_stride=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.imu_path = imu_path
        self.window_stride = int(window_stride)


# ============================================================
# DATASET UNIFICATO
# ============================================================

class Radar4DCamImuDatasetPairsRAW(Dataset):
    """
    Wrapper:
    - Radar + Camera: identici al dataset originale
    - IMU: finestra temporale allineata alla coppia radar-camera (t, t+1)
    """

    def __init__(self, seqs_cfg, *args, **kwargs):
        super().__init__()

        self.rc_dataset = Radar4DPlusCamDatasetPairsRAW(
            seqs_cfg, *args, **kwargs
        )

        # carica IMU per sequenza
        self.imu_data = {}
        # ---- IMU norm (impostata da make_loaders SOLO sul train) ----
        self.imu_mean = None  # torch tensor (1,1,D)
        self.imu_std  = None  # torch tensor (1,1,D)

        for sid, cfg in enumerate(seqs_cfg):
            assert cfg.imu_path is not None, f"IMU path mancante per seq {cfg.name}"
            imu = load_imu_fixed_csv(cfg.imu_path)
            self.imu_data[sid] = imu

    def __len__(self):
        return len(self.rc_dataset)

    def set_imu_norm(self, mean_np, std_np):
        """
        mean_np/std_np: numpy (D,)
        Salviamo come torch (1,1,D) per broadcast su (L,D).
        """
        mean = torch.as_tensor(mean_np, dtype=torch.float32).view(1, 1, -1)
        std  = torch.as_tensor(std_np,  dtype=torch.float32).view(1, 1, -1)
        self.imu_mean = mean
        self.imu_std  = std


    def _slice_imu(self, imu_tuple, t0, t1):
        """
        Estrae finestra IMU allineata alla coppia radar (t0=ts_t, t1=ts_t1).
        Mantiene la tua logica: radar (e quindi camera) sono il riferimento temporale.
        """
        ts, X = imu_tuple  # load_imu_fixed_csv -> (ts, X) :contentReference[oaicite:6]{index=6}

        # Assumiamo ts ordinato (tipico). Se vuoi super-robustezza, puoi sortare una volta in init.
        i0 = np.searchsorted(ts, t0, side="left")
        i1 = np.searchsorted(ts, t1, side="right")

        Xi = X[i0:i1]
        if Xi.shape[0] == 0:
            # fallback robusto: almeno 1 campione, quello più vicino a t0
            j = int(np.clip(i0, 0, len(ts) - 1))
            Xi = X[j:j+1]

        return torch.from_numpy(Xi).float()


    def __getitem__(self, idx):
        
        pairs, y, sid, L, extra = self.rc_dataset[idx]  # <-- formato camera-only corretto

        t0_list = extra["ts_t"]    # lista length L
        t1_list = extra["ts_t1"]   # lista length L
        imu_tuple = self.imu_data[sid]


        imu_windows = []
        for t0, t1 in zip(t0_list, t1_list):
            w = self._slice_imu(imu_tuple, t0, t1)
            if (self.imu_mean is not None) and (self.imu_std is not None):
                w = (w - self.imu_mean[0,0]) / (self.imu_std[0,0] + 1e-8)
            imu_windows.append(w)

        return pairs, y, sid, L, extra, imu_windows, (t0_list, t1_list)




# ============================================================
# COLLATE FUNCTION (RADAR + CAM + IMU)
# ============================================================
def pad_collate_radar_cam_imu_raw_batched(batch):
    pairs, y, sid, L, extra, imu_list, ts = zip(*batch)

    # 1) collate radar+cam “vero”
    radar_batched, cam_batched, y_out, sid_out, lengths_out = pad_collate_radar_cam_raw_batched(
        list(zip(pairs, y, sid, L, extra))
    )

    # 2) padding IMU PER-STEP
    # imu_list: tuple length B
    # imu_list[b] = list length L (es: 4)
    # imu_list[b][t] = Tensor (Li_bt, D)

    B = len(imu_list)
    # L può essere un int oppure una tupla, qui prendiamo quello massimo (tipicamente 4 fisso)
    Lmax_steps = int(max(L)) if hasattr(L, "__len__") else int(L)

    # dimensione feature IMU D
    D = imu_list[0][0].shape[1]

    # lengths IMU (B, L)
    imu_lengths = torch.zeros(B, Lmax_steps, dtype=torch.long)

    # trova Lmax_imu globale (massima lunghezza finestra IMU tra tutti i sample e step)
    Lmax_imu = 1
    for b in range(B):
        for t in range(Lmax_steps):
            x = imu_list[b][t]
            Li = int(x.shape[0])
            imu_lengths[b, t] = Li
            if Li > Lmax_imu:
                Lmax_imu = Li

    # tensor padded (B, L, Lmax_imu, D)
    imu_padded = torch.zeros(B, Lmax_steps, Lmax_imu, D, dtype=imu_list[0][0].dtype)
    for b in range(B):
        for t in range(Lmax_steps):
            x = imu_list[b][t]  # (Li, D)
            Li = int(x.shape[0])
            if Li > 0:
                imu_padded[b, t, :Li] = x

    # ts: tuple length B, ts[b] = (t0_list, t1_list) dove ciascuno è list length L
    t0 = torch.zeros(B, Lmax_steps, dtype=torch.float64)
    t1 = torch.zeros(B, Lmax_steps, dtype=torch.float64)
    for b in range(B):
        t0_list, t1_list = ts[b]
        for t in range(Lmax_steps):
            t0[b, t] = float(t0_list[t])
            t1[b, t] = float(t1_list[t])

    imu = {
        "X": imu_padded,         # (B, L, Lmax_imu, D)
        "lengths": imu_lengths,  # (B, L)
        "t0": t0,                # (B, L)
        "t1": t1,                # (B, L)
    }

    return radar_batched, cam_batched, imu, y_out, sid_out, lengths_out




# ============================================================
# FACTORY FUNCTION (TRAIN / VAL / TEST)
# ============================================================

def make_loaders_radar_cam_imu_pose_raw_batched(
    seqs_cfg,
    batch_size=2,
    num_workers=8,
    shuffle_train=True,
    perc=(0.70, 0.15, 0.15),
    imu_norm_stats=None,
):
    ds = Radar4DCamImuDatasetPairsRAW(seqs_cfg)

    if imu_norm_stats is not None:
        imu_mean, imu_std = imu_norm_stats
        ds.imu_mean = imu_mean
        ds.imu_std  = imu_std


    # SPLIT deterministico per scena usando l’ordine di samples
    sid_to_indices = {}
    for i, s in enumerate(ds.rc_dataset.samples):
        sid, _ = s
        sid_to_indices.setdefault(sid, []).append(i)


    idx_train, idx_val, idx_test = [], [], []
    for sid, idx_all in sid_to_indices.items():
        n = len(idx_all)
        ntr = int(n * perc[0])
        nva = int(n * perc[1])
        idx_train += idx_all[:ntr]
        idx_val   += idx_all[ntr:ntr+nva]
        idx_test  += idx_all[ntr+nva:]

    train_subset = Subset(ds, idx_train)
    val_subset   = Subset(ds, idx_val)
    test_subset  = Subset(ds, idx_test)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_collate_radar_cam_imu_raw_batched,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_collate_radar_cam_imu_raw_batched,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_collate_radar_cam_imu_raw_batched,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, ds
