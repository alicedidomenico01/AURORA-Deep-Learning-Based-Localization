#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Variante "IMU-step":
# - La GT (local_inspva.txt) è già nel frame IMU -> NON viene trasformata.
# - I vettori IMU (acc, gyr, mag) restano nel frame IMU (nessuna rotazione).
# - Target: Δt nel frame IMU locale + dq fra orientazioni IMU successive.
#
# Struttura GT: righe "ts tx ty tz qx qy qz qw" (frame IMU).
# Sincronizzazione temporale: invariata (finestra IMU tra due GT adiacenti).

import os, math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# ============================
# CONFIG PERCORSI
# ============================
ROOT_DIR = "/media/arrubuntu20/HDD/Hercules"
IMU_REL  = "sensor_data/xsens_imu.csv"

# Posizioni candidate del file GT per la sequenza XX:
# - <seq>/PR_GT/local_inspva.txt
# - <seq>/GT/local_inspva.txt
# - <ROOT_DIR>/GT/XX/local_inspva.txt
GT_CANDIDATES = [
    os.path.join("PR_GT", "local_inspva.txt"),
    os.path.join("GT", "local_inspva.txt"),
    # la terza è costruita a runtime perché dipende da ROOT_DIR e dalla seq
]

IMU_COLS = {
    "ts": 0,
    "qx": 1, "qy": 2, "qz": 3, "qw": 4,
    "eulx": 5, "euly": 6, "eulz": 7,
    "gyrx": 8, "gyry": 9, "gyrz": 10,
    "accx": 11, "accy": 12, "accz": 13,
    "magx": 14, "magy": 15, "magz": 16,
    "extra": 17
}
USE_MAG = True

# ============================
# CORREZIONE BIAS MAGNETOMETRO (HARD-IRON)
# ============================
MAG_BIAS_X = 0.021831
MAG_BIAS_Y = 0.217830
MAG_BIAS_Z = -0.956765

# ============================
# Utility pose / algebra (xyzw)
# ============================
def quat_to_mat(q):
    qx,qy,qz,qw = q
    n = qx*qx+qy*qy+qz*qz+qw*qw
    if n < 1e-12: return np.eye(3)
    s = 2.0/n
    X = qx*s; Y = qy*s; Z = qz*s
    wX = qw*X; wY = qw*Y; wZ = qw*Z
    xX = qx*X; xY = qx*Y; xZ = qx*Z
    yY = qy*Y; yZ = qy*Z; zZ = qz*Z
    R = np.array([
        [1-(yY+zZ), xY-wZ,     xZ+wY],
        [xY+wZ,     1-(xX+zZ), yZ-wX],
        [xZ-wY,     yZ+wX,     1-(xX+yY)]
    ], dtype=np.float64)
    return R

def pose_to_T(tx,ty,tz,qx,qy,qz,qw):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = quat_to_mat([qx,qy,qz,qw])
    T[:3,3]  = np.array([tx,ty,tz], dtype=np.float64)
    return T

def mat_to_quat(R):
    m=R; t=np.trace(m)
    if t>0:
        S=math.sqrt(t+1.0)*2
        qw=0.25*S; qx=(m[2,1]-m[1,2])/S; qy=(m[0,2]-m[2,0])/S; qz=(m[1,0]-m[0,1])/S
    else:
        if m[0,0]>m[1,1] and m[0,0]>m[2,2]:
            S=math.sqrt(1.0+m[0,0]-m[1,1]-m[2,2])*2
            qw=(m[2,1]-m[1,2])/S; qx=0.25*S
            qy=(m[0,1]+m[1,0])/S; qz=(m[0,2]+m[2,0])/S
        elif m[1,1]>m[2,2]:
            S=math.sqrt(1.0+m[1,1]-m[0,0]-m[2,2])*2
            qw=(m[0,2]-m[2,0])/S; qx=(m[0,1]+m[1,0])/S
            qy=0.25*S;             qz=(m[1,2]+m[2,1])/S
        else:
            S=math.sqrt(1.0+m[2,2]-m[0,0]-m[1,1])*2
            qw=(m[1,0]-m[0,1])/S; qx=(m[0,2]+m[2,0])/S
            qy=(m[1,2]+m[2,1])/S; qz=0.25*S
    q=np.array([qx,qy,qz,qw], dtype=np.float64)
    q/=np.linalg.norm(q)+1e-12
    return q

# ============================
# Letture file
# ============================
def load_imu_fixed_csv(path):
    # parsing robusto (come tua versione)
    for s in [',', r'\s+', ';', '\t']:
        try:
            df=pd.read_csv(path,sep=s,engine='python',header=None)
            if df.shape[1]>=14: break
        except: continue

    n=df.shape[1]
    ts=df.iloc[:,IMU_COLS["ts"]].astype(np.float64).values * 1e-9
    acc=df.iloc[:,[IMU_COLS["accx"],IMU_COLS["accy"],IMU_COLS["accz"]]].astype(np.float32).values
    gyr=df.iloc[:,[IMU_COLS["gyrx"],IMU_COLS["gyry"],IMU_COLS["gyrz"]]].astype(np.float32).values

    if USE_MAG and n>IMU_COLS["magz"]:
        mag=df.iloc[:,[IMU_COLS["magx"],IMU_COLS["magy"],IMU_COLS["magz"]]].astype(np.float32).values
        mag_bias_vec = np.array([MAG_BIAS_X, MAG_BIAS_Y, MAG_BIAS_Z], dtype=np.float32)
        mag_corrected = mag - mag_bias_vec
        X=np.concatenate([acc, gyr, mag_corrected], axis=1)
    else:
        X=np.concatenate([acc,gyr],axis=1)

    return ts,X

def resolve_gt_path(seq_dir, seq_name):
    # prova le posizioni locali alla sequenza
    for rel in GT_CANDIDATES[:-1]:
        cand = os.path.join(seq_dir, rel)
        if os.path.isfile(cand):
            return cand
    # prova ROOT_DIR/GT/<seq>/local_inspva.txt
    cand3 = os.path.join(ROOT_DIR, "GT", seq_name, "local_inspva.txt")
    if os.path.isfile(cand3):
        return cand3
    # altrimenti ultima chance: <seq_dir>/local_inspva.txt
    cand4 = os.path.join(seq_dir, "local_inspva.txt")
    if os.path.isfile(cand4):
        return cand4
    raise FileNotFoundError(
        f"local_inspva.txt non trovato per {seq_name}. "
        f"Prova a metterlo in PR_GT/ o GT/ sotto la sequenza, oppure in ROOT_DIR/GT/{seq_name}/."
    )

def load_gt_imu_txt(path):
    """
    Legge GT già nel frame IMU con colonne:
    ts tx ty tz qx qy qz qw   (senza header o con #)
    """
    expected = ["ts","tx","ty","tz","qx","qy","qz","qw"]
    df = None
    for sep in [r"\s+", ",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, engine='python', comment="#", header=None)
            if df is not None and df.shape[1] >= 8:
                break
        except Exception:
            pass
    if df is None:
        raise ValueError(f"Impossibile parsare GT IMU: {path}")
    if df.shape[1] > 8:
        df = df.iloc[:, :8]
    if df.shape[1] < 8:
        raise ValueError(f"GT ha {df.shape[1]} colonne, attese 8: {path}")
    df.columns = expected
    return df

# ============================
# Dataset per intervalli IMU (Δt in frame locale + dq)
# ============================
class MultiSeqIMUStepsQuatLocalPatch(Dataset):
    """
    Target y = [dx_local, dy_local, dz_local, dqx, dqy, dqz, dqw]
    con:
      dt_local = R0^T (t1 - t0)     (frame IMU)
      dq = quat(R0^T R1)            (xyzw, frame IMU)
    """
    def __init__(self, seq_ids):
        self.seq_ids=list(seq_ids)
        self.X_list_raw=[]; self.ts_list=[]
        self.gt_T_list=[];  self.gt_ts_list=[]
        self.samples=[]

        # norm placeholders (impostati da make_loaders SOLO sul train)
        self.mean=None
        self.std=None
        self.X_norm=None  # popolato da regenerate_norm()

        for sid,name in enumerate(self.seq_ids):
            seq_dir=os.path.join(ROOT_DIR,name)

            # 1) IMU grezza (frame IMU, non ruotiamo nulla)
            ts_imu, X_imu = load_imu_fixed_csv(os.path.join(seq_dir,IMU_REL))

            # 2) GT in frame IMU (non trasformare!)
            gt_path = resolve_gt_path(seq_dir, name)
            gt = load_gt_imu_txt(gt_path)

            # 3) Trim GT al range IMU
            t0_imu, t1_imu = ts_imu[0], ts_imu[-1]
            gt = gt[(gt["ts"] >= t0_imu) & (gt["ts"] <= t1_imu)].reset_index(drop=True)
            if len(gt) < 2:
                raise ValueError(f"{name}: troppo poca GT dopo trimming (no overlap con IMU)")

            # 4) Lista di T in frame IMU (nessuna trasformazione)
            T_list=[]
            for _,r in gt.iterrows():
                Tr = pose_to_T(r.tx,r.ty,r.tz,r.qx,r.qy,r.qz,r.qw)  # già IMU
                T_list.append(Tr)
            T_list=np.stack(T_list)

            self.X_list_raw.append(X_imu.astype(np.float32))
            self.ts_list.append(ts_imu.astype(np.float64))
            self.gt_T_list.append(T_list.astype(np.float64))
            self.gt_ts_list.append(gt["ts"].values.astype(np.float64))

            for k in range(len(self.gt_ts_list[-1])-1):
                self.samples.append((sid,k))

    def regenerate_norm(self, mean, std):
        self.mean = mean.astype(np.float32).reshape(1,-1)
        self.std  = (std.astype(np.float32).reshape(1,-1) + 1e-8)
        self.X_norm=[((Xi-self.mean)/self.std).astype(np.float32) for Xi in self.X_list_raw]

    def __len__(self): return len(self.samples)

    def _delta_pose_quat_local(self, T0, T1):
        R0 = T0[:3,:3]; t0 = T0[:3,3]
        R1 = T1[:3,:3]; t1 = T1[:3,3]
        dt_world = (t1 - t0).astype(np.float64)
        dt_local = (R0.T @ dt_world).astype(np.float64)
        dR = R0.T @ R1
        dq = mat_to_quat(dR).astype(np.float64)  # (x,y,z,w), unit
        return np.concatenate([dt_local.astype(np.float32), dq.astype(np.float32)], axis=0)

    def __getitem__(self, idx):
        sid, k = self.samples[idx]
        ts_imu = self.ts_list[sid]
        Xn     = self.X_norm[sid] if self.X_norm is not None else self.X_list_raw[sid]
        ts_gt  = self.gt_ts_list[sid]
        Tlist  = self.gt_T_list[sid]

        # finestre IMU tra le due GT adiacenti, con fallback al campione più vicino
        t0 = ts_gt[k]; t1 = ts_gt[k+1]
        i0 = np.searchsorted(ts_imu, t0, side='left')
        i1 = np.searchsorted(ts_imu, t1, side='right')
        Xi = Xn[i0:i1]
        if Xi.shape[0] == 0:
            j = max(0, min(len(ts_imu)-1, i0))
            Xi = Xn[j:j+1]

        y7 = self._delta_pose_quat_local(Tlist[k], Tlist[k+1])  # [3 + 4] = 7
        return torch.from_numpy(Xi), torch.from_numpy(y7), k, Xi.shape[0]

# ----------------------------
# Collate con padding per batch (identico)
# ----------------------------
def pad_collate_quat_localpatch(batch):
    lengths = [b[3] for b in batch]
    B = len(batch)
    Lmax = max(lengths)
    D = batch[0][0].shape[1]
    X_pad = torch.zeros(B, Lmax, D, dtype=batch[0][0].dtype)
    y_out = torch.zeros(B, 7, dtype=batch[0][1].dtype)
    pair_idx_o = torch.zeros(B, dtype=torch.long)  # indice locale di finestra k
    for i,(Xi, yi, pair_idx, li) in enumerate(batch):
        X_pad[i, :li] = Xi
        y_out[i] = yi
        pair_idx_o[i] = pair_idx
    return X_pad, y_out, pair_idx_o, torch.tensor(lengths, dtype=torch.long)


# ----------------------------
# Split + DataLoader (identico, ma export col nuovo nome)
# ----------------------------
def make_loaders_quat_localpatch(seqs, batch_size=16, perc=(0.70,0.15,0.15),
                                 seed=0, num_workers=0, cap_train_windows_per_scene=None):
    ds = MultiSeqIMUStepsQuatLocalPatch(seqs)
    rng = np.random.RandomState(seed)

    per_scene_idx = {"train": {}, "val": {}, "test": {}}
    idx_train, idx_val, idx_test = [], [], []

    # --- split per-scena a blocchi contigui ---
    for sid in range(len(ds.seq_ids)):
        n_k = len(ds.gt_ts_list[sid]) - 1
        ntr = int(perc[0] * n_k)
        nva = int(perc[1] * n_k)
        idx_all = [i for i, (s, _) in enumerate(ds.samples) if s == sid]

        idxs_tr = idx_all[:ntr]
        idxs_va = idx_all[ntr:ntr+nva]
        idxs_te = idx_all[ntr+nva:]

        idx_train.extend(idxs_tr)
        idx_val.extend(idxs_va)
        idx_test.extend(idxs_te)

        per_scene_idx["train"][sid] = idxs_tr
        per_scene_idx["val"][sid]   = idxs_va
        per_scene_idx["test"][sid]  = idxs_te

    # --- normalizzazione solo sul train (scene nel train) ---
    sids_train = set([ds.samples[i][0] for i in idx_train])
    Xcat = np.concatenate([ds.X_list_raw[sid] for sid in sids_train], axis=0)
    mean = Xcat.mean(axis=0).astype(np.float32)
    std  = Xcat.std(axis=0).astype(np.float32)
    ds.regenerate_norm(mean, std)

    if cap_train_windows_per_scene is not None:
        capped = []
        for sid, lst in per_scene_idx["train"].items():
            lst = lst.copy()
            rng.shuffle(lst)
            capped.extend(lst[:cap_train_windows_per_scene])
        idx_train = capped

    train_loader = DataLoader(Subset(ds, idx_train), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=pad_collate_quat_localpatch)
    val_loader   = DataLoader(Subset(ds, idx_val), batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=pad_collate_quat_localpatch)
    test_loader  = DataLoader(Subset(ds, idx_test), batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=pad_collate_quat_localpatch)

    return ds, train_loader, val_loader, test_loader, per_scene_idx

