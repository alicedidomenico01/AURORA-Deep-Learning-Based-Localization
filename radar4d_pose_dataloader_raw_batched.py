# -*- coding: utf-8 -*-
"""
radar4d_pose_dataloader_raw_batched.py

Collate "batched" per impacchettare tutte le coppie del batch in tensori unici.
Usata solo per TRAIN/VAL. Eval resta invariata (usa i tuoi loaders originali).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import List, Tuple
from typing import Dict, List


# importa la nuova definizione con calib_dir
from radar4d_pose_dataloader_raw import Radar4DDatasetPairsRAW, SeqConfig

__all__ = ["pad_collate_radar_raw_batched", "make_loaders_radar_pose_raw_batched"]


def pad_collate_radar_raw_batched(batch: List[Tuple]):
    """
    batch element: (pairs:list[L] di (xyz_t, feat_t, xyz_t1, feat_t1), y(7 torch), sid(int), L(int))

    Ritorna:
      pairs_batched: dict con
        Xt:  (P, Nmax_t,  3) float32
        Ft:  (P, Nmax_t,  Cin_t) float32
        Xt1: (P, Nmax_t1, 3) float32
        Ft1: (P, Nmax_t1, Cin_t1) float32
        validN_t: (P,) long
        pair_b:   (P,) long
        pair_t:   (P,) long
      y_out: (B,7) float32
      sid_o: (B,) long
      lengths:(B,) long
    """
    B = len(batch)
    pairs_list = [b[0] for b in batch]
    y_out = torch.stack([b[1] for b in batch], dim=0)  # (B,7)
    sid_o = torch.tensor([b[2] for b in batch], dtype=torch.long)
    lengths = torch.tensor([b[3] for b in batch], dtype=torch.long)

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
        return pairs_batched, y_out, sid_o, lengths

    Nmax_t  = int(max(Nt_list))
    Nmax_t1 = int(max(Nt1_list))

    Xt  = torch.zeros(P, Nmax_t,  3,     dtype=torch.float32)
    Ft  = torch.zeros(P, Nmax_t,  Cin_t, dtype=torch.float32)
    Xt1 = torch.zeros(P, Nmax_t1, 3,     dtype=torch.float32)
    Ft1 = torch.zeros(P, Nmax_t1, Cin_t1,dtype=torch.float32)
    validN_t = torch.tensor(Nt_list, dtype=torch.long)

    p = 0
    for b_idx, pairs in enumerate(pairs_list):
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
            p += 1

    pairs_batched = {
        "Xt": Xt, "Ft": Ft, "Xt1": Xt1, "Ft1": Ft1,
        "validN_t": validN_t,
        "pair_b": torch.tensor(pair_b, dtype=torch.long),
        "pair_t": torch.tensor(pair_t, dtype=torch.long),
    }
    return pairs_batched, y_out, sid_o, lengths


def make_loaders_radar_pose_raw_batched(seqs: List[SeqConfig], batch_size=8, perc=(0.70,0.15,0.15),
                                        seed=0, num_workers=0):
    ds = Radar4DDatasetPairsRAW(seqs)

    # === 1) Mappa "quali indici globali appartengono a quale sequenza" leggendo ds.samples ===
    sid_to_indices: Dict[int, List[int]] = {sid: [] for sid in range(len(ds.seqs))}
    for gidx, (sid_i, _pair_idx) in enumerate(ds.samples):
        sid_to_indices[sid_i].append(gidx)

    rng = np.random.RandomState(seed)

    idx_train, idx_val, idx_test = [], [], []
    per_scene_idx = {"train": {}, "val": {}, "test": {}}

    for sid in range(len(ds.seqs)):
        idx_all = sid_to_indices[sid]
        n = len(idx_all)
        if n == 0:
            per_scene_idx["train"][sid] = []
            per_scene_idx["val"][sid]   = []
            per_scene_idx["test"][sid]  = []
            continue

        # mantieni ordine temporale (se vuoi random: rng.shuffle(idx_all))
        ntr = int(perc[0] * n)
        nva = int(perc[1] * n)

        tr = idx_all[:ntr]
        va = idx_all[ntr:ntr+nva]
        te = idx_all[ntr+nva:]

        idx_train.extend(tr); per_scene_idx["train"][sid] = tr
        idx_val.extend(va);   per_scene_idx["val"][sid]   = va
        idx_test.extend(te);  per_scene_idx["test"][sid]  = te

    # === 2) Sanity check: tutti gli indici devono essere in range ===
    max_valid = len(ds) - 1
    def _sanitize(tag, arr):
        bad = [i for i in arr if (i < 0 or i > max_valid)]
        if bad:
            print(f"[ERR] {tag}: {len(bad)} indici fuori range (max={max_valid}). Esempi: {bad[:10]}")
            arr = [i for i in arr if (0 <= i <= max_valid)]
            print(f"[FIX] {tag}: dopo filtro -> {len(arr)}")
        else:
            print(f"[OK ] {tag}: {len(arr)} indici (0..{max_valid})")
        return arr

    idx_train = _sanitize("train", idx_train)
    idx_val   = _sanitize("val",   idx_val)
    idx_test  = _sanitize("test",  idx_test)

    # === 3) DataLoader (collate batched) ===
    train_loader = DataLoader(
       Subset(ds, idx_train),
       batch_size=batch_size,
       shuffle=True,
       num_workers=num_workers,
       pin_memory=True,
       persistent_workers=(num_workers > 0),
       collate_fn=pad_collate_radar_raw_batched,
    )
    val_loader = DataLoader(
       Subset(ds, idx_val),
       batch_size=batch_size,
       shuffle=False,
       num_workers=num_workers,
       pin_memory=True,
       persistent_workers=(num_workers > 0),
       collate_fn=pad_collate_radar_raw_batched,
    )
    test_loader = DataLoader(
       Subset(ds, idx_test),
       batch_size=batch_size,
       shuffle=False,
       num_workers=num_workers,
       pin_memory=True,
       persistent_workers=(num_workers > 0),
       collate_fn=pad_collate_radar_raw_batched,
    )

    return ds, train_loader, val_loader, test_loader, per_scene_idx



