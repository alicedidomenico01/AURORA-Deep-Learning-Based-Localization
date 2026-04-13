# -*- coding: utf-8 -*-
"""
radar4d_pose_dataloader_raw.py

Modifiche richieste:
1) GT: usare local_inspva.txt (frame IMU, 50 Hz) con sincronizzazione "nearest"
   
2) Punti radar trasformati nel frame IMU con:
     Continental_LiDAR.txt  -> "Tr_lidar_to_radar" (LiDAR -> Radar)
     IMU_LiDAR.txt          -> "Tr_lidar_to_imu"   (LiDAR -> IMU)
   => T_radar→imu = T_lidar→imu @ inv(T_lidar→radar)


"""

import os, re, glob, math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset

# =================== Normalizzazioni stile "cmflow_like" ===================
XYZ_SCALE_CM = 100.0
DOP_CLIP_CM  = 20.0
RCS_SCALE_CM = 255.0
EMIT_RANGE_NORM = True

DOPPLER_FIELD_CANDIDATES = ["velocity","doppler","vr","radial_velocity"]
RCS_FIELD_CANDIDATES     = ["intensity","rcs","power","reflectivity","amp","amplitude"]


# =================== Utils Pose/Quat (xyzw) ===================
def quat_to_mat(q):
    qx,qy,qz,qw = q
    n = qx*qx+qy*qy+qz*qz+qw*qw
    if n < 1e-12: return np.eye(3, dtype=np.float64)
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

def pose_to_T(tx,ty,tz,qx,qy,qz,qw):
    T = np.eye(4, dtype=np.float64)
    R = quat_to_mat([qx,qy,qz,qw])
    T[:3,:3] = R
    T[:3,3]  = np.array([tx,ty,tz], dtype=np.float64)
    return T

def delta_pose_local(T0, T1):
    R0 = T0[:3,:3]; t0 = T0[:3,3]
    R1 = T1[:3,:3]; t1 = T1[:3,3]
    dt_world = (t1 - t0).astype(np.float64)
    dt_local = (R0.T @ dt_world).astype(np.float32)
    dR = R0.T @ R1
    dq = mat_to_quat(dR).astype(np.float32)
    return np.concatenate([dt_local, dq], axis=0)  # (7,)



# ============== Piccole utility timestamp/indice ==============
def _nearest_idx(arr_sorted: np.ndarray, t: float) -> int:
    """Indice del valore in arr_sorted più vicino a t (arr_sorted crescente)."""
    i = int(np.searchsorted(arr_sorted, t))
    if i <= 0: return 0
    if i >= len(arr_sorted): return len(arr_sorted)-1
    return i-1 if abs(arr_sorted[i-1]-t) <= abs(arr_sorted[i]-t) else i

# =================== PCD IO + normalizzazione ===================
def _parse_pcd_header(fb):
    header, fields, sizes, types, counts = {}, [], [], [], []
    while True:
        line = fb.readline().decode('utf-8', errors='ignore')
        if not line: break
        s = line.strip()
        if not s or s.startswith("#"): continue
        key = s.split(" ", 1)[0].upper()
        if key == "FIELDS":
            fields = s.split()[1:]; header["FIELDS"]=fields
        elif key == "SIZE":
            sizes = list(map(int, s.split()[1:])); header["SIZE"]=sizes
        elif key == "TYPE":
            types = s.split()[1:]; header["TYPE"]=types
        elif key == "COUNT":
            counts = list(map(int, s.split()[1:])); header["COUNT"]=counts
        elif key == "WIDTH":
            header["WIDTH"]=int(s.split()[1])
        elif key == "HEIGHT":
            header["HEIGHT"]=int(s.split()[1])
        elif key == "POINTS":
            header["POINTS"]=int(s.split()[1])
        elif key == "DATA":
            header["DATA"]=s.split()[1].lower()
            header["_header_size"]=fb.tell()
            break
    if "COUNT" not in header:
        header["COUNT"] = [1]*len(header.get("FIELDS", []))
    return header

def _pcd_dtype(types, sizes):
    dtypes=[]
    for t,s in zip(types, sizes):
        if t == 'F':
            dtypes.append(np.float32 if s==4 else np.float64)
        elif t == 'U':
            dtypes.append({1:np.uint8,2:np.uint16,4:np.uint32}[s])
        elif t == 'I':
            dtypes.append({1:np.int8,2:np.int16,4:np.int32}[s])
        else:
            raise ValueError(f"Unsupported type {t}")
    return dtypes

def read_pcd(path):
    import gzip, io, struct
    with open(path, "rb") as f:
        h = _parse_pcd_header(f)
        fields = h["FIELDS"]; sizes=h.get("SIZE",[4]*len(fields)); types=h.get("TYPE",["F"]*len(fields))
        counts = h.get("COUNT",[1]*len(fields)); datafmt=h["DATA"]
        width=h.get("WIDTH", h.get("POINTS",0)); height=h.get("HEIGHT",1); npts=h.get("POINTS", width*height)
        dt_map=_pcd_dtype(types, sizes); dtype_list=[]
        for field,dt,c in zip(fields, dt_map, counts):
            if c==1: dtype_list.append((field, dt))
            else: dtype_list.extend([(f"{field}_{i}", dt) for i in range(c)])
        f.seek(h["_header_size"]); out={}
        if datafmt=="ascii":
            raw = f.read().decode("utf-8", errors="ignore")
            lines=[ln for ln in raw.splitlines() if ln.strip()]
            if not lines:
                arr=np.empty((0, sum(counts)), dtype=np.float32)
            else:
                sio=io.StringIO("\n".join(lines))
                arr=np.loadtxt(sio, comments="#", dtype=np.float32, ndmin=2)
            idx=0
            for field,c in zip(fields, counts):
                if c==1:
                    out[field]=arr[:, idx].astype(np.float32); idx+=1
                else:
                    for j in range(c):
                        out[f"{field}_{j}"]=arr[:, idx+j].astype(np.float32)
                    idx+=c
        elif datafmt=="binary":
            import struct as _st
            point_size=sum(s*c for s,c in zip(sizes, counts))
            raw=f.read(npts*point_size)
            rec=np.frombuffer(raw, dtype=np.dtype([(field, np.float32) for field in fields]), count=npts)
            out={name: rec[name].astype(np.float32) for name in rec.dtype.names}
        elif datafmt=="binary_compressed":
            import struct as _st, gzip as _gz
            comp_size=_st.unpack('I', f.read(4))[0]
            uncomp_size=_st.unpack('I', f.read(4))[0]
            comp=f.read(comp_size); raw=_gz.decompress(comp)
            rec=np.frombuffer(raw, dtype=np.dtype([(field, np.float32) for field in fields]), count=npts)
            out={name: rec[name].astype(np.float32) for name in rec.dtype.names}
        else:
            raise ValueError(f"Unsupported PCD DATA: {datafmt}")
    return {"header": h, "data": out}

def _pick_first_field(cands, data_dict):
    for n in cands:
        if n in data_dict: return n
    return None

def load_pcd_cmflow_like(path, emit_range_norm=EMIT_RANGE_NORM):
    rec = read_pcd(path); d=rec["data"]
    for k in ("x","y","z"):
        if k not in d: raise ValueError(f"Missing '{k}' in {path}")
    x=d["x"]; y=d["y"]; z=d["z"]
    dop_name=_pick_first_field(DOPPLER_FIELD_CANDIDATES, d)
    rcs_name=_pick_first_field(RCS_FIELD_CANDIDATES, d)
    dop = d[dop_name] if dop_name else np.zeros_like(x, np.float32)
    rcs = d[rcs_name] if rcs_name else np.zeros_like(x, np.float32)

    mask = np.isfinite(x)&np.isfinite(y)&np.isfinite(z)&np.isfinite(dop)&np.isfinite(rcs)
    x,y,z,dop,rcs = x[mask],y[mask],z[mask],dop[mask],rcs[mask]

    x = (x / XYZ_SCALE_CM).astype(np.float32)
    y = (y / XYZ_SCALE_CM).astype(np.float32)
    z = (z / XYZ_SCALE_CM).astype(np.float32)
    if DOP_CLIP_CM > 0:
        dop = np.clip(dop, -DOP_CLIP_CM, DOP_CLIP_CM) / DOP_CLIP_CM
    if RCS_SCALE_CM > 0:
        rcs = np.clip(rcs, 0.0, RCS_SCALE_CM) / RCS_SCALE_CM

    range_norm = None
    if emit_range_norm:
        rng = np.sqrt((x*XYZ_SCALE_CM)**2 + (y*XYZ_SCALE_CM)**2 + (z*XYZ_SCALE_CM)**2)
        denom = np.percentile(rng, 99.0) if rng.size>0 else 1.0
        if denom < 1e-3: denom = 1.0
        range_norm = np.clip(rng/denom, 0.0, 1.0).astype(np.float32).reshape(-1,1)

    xyz = np.stack([x,y,z], axis=1).astype(np.float32)
    dop = dop.reshape(-1,1).astype(np.float32)
    rcs = rcs.reshape(-1,1).astype(np.float32)
    if range_norm is not None:
        feat = np.concatenate([dop, rcs, range_norm], axis=1)
    else:
        feat = np.concatenate([dop, rcs], axis=1)
    return xyz, feat

# =================== Calibrazione ===================
def _parse_3x4_from_line(s: str, key: str) -> Optional[np.ndarray]:
    """
    Cerca 'key:' e 12 float dopo; ritorna (4x4) con ultima riga [0 0 0 1].
    """
    if key not in s: return None
    try:
        part = s.split(key, 1)[1]
        nums = [float(v) for v in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", part)]
        if len(nums) >= 12:
            M = np.array(nums[:12], dtype=np.float64).reshape(3,4)
            T = np.eye(4, dtype=np.float64); T[:3,:4] = M
            return T
    except Exception:
        pass
    return None

def load_radar_to_imu_T(calib_dir: str) -> np.ndarray:
    """
    Legge:
      - <calib_dir>/Continental_LiDAR.txt  con 'Tr_lidar_to_radar: 12 valori'
      - <calib_dir>/IMU_LiDAR.txt          con 'Tr_lidar_to_imu:  12 valori'
    Ritorna T_radar→imu (4x4).
    """
    p_lr = os.path.join(calib_dir, "Continental_LiDAR.txt")
    p_li = os.path.join(calib_dir, "IMU_LiDAR.txt")
    if not (os.path.isfile(p_lr) and os.path.isfile(p_li)):
        raise FileNotFoundError(f"Calib non trovata in {calib_dir} (attesi Continental_LiDAR.txt e IMU_LiDAR.txt)")
    with open(p_lr, "r") as f:
        txt_lr = f.read()
    with open(p_li, "r") as f:
        txt_li = f.read()
    T_lidar_to_radar = _parse_3x4_from_line(txt_lr, "Tr_lidar_to_radar:")
    T_lidar_to_imu   = _parse_3x4_from_line(txt_li, "Tr_lidar_to_imu:")
    if T_lidar_to_radar is None or T_lidar_to_imu is None:
        raise ValueError("Parsing calib fallito: assicurati che ci siano 12 numeri dopo i tag corretti.")
    T_radar_to_imu = T_lidar_to_imu @ np.linalg.inv(T_lidar_to_radar)
    return T_radar_to_imu

def transform_xyz(T_4x4: np.ndarray, xyz_m: np.ndarray) -> np.ndarray:
    """Applica T a punti (N,3) in metri."""
    if xyz_m.size == 0: return xyz_m
    N = xyz_m.shape[0]
    homog = np.concatenate([xyz_m, np.ones((N,1), dtype=np.float32)], axis=1)  # (N,4)
    out = (T_4x4.astype(np.float64) @ homog.T).T[:, :3]
    return out.astype(np.float32)

# =================== Dataset RAW ===================
def _parse_ts_from_name(name: str) -> Optional[float]:
    stem = os.path.splitext(os.path.basename(name))[0]
    m = re.search(r'(\d+\.\d+)', stem)
    if m:
        try: return float(m.group(1))
        except: pass
    m = re.search(r'(\d{10,})', stem)
    if m:
        try: return float(m.group(1)) * 1e-9
        except: pass
    m = re.search(r'(\d+)', stem)
    if m:
        try: return float(m.group(1))
        except: pass
    return None

@dataclass
class SeqConfig:
    name: str
    pcd_dir: str
    gt_path: str
    calib_dir: str  # <-- aggiunto

class Radar4DDatasetPairsRAW(Dataset):
    def __init__(self, seqs: List[SeqConfig]):
        super().__init__()
        self.seqs = seqs
        self.seq_frames: List[Dict] = []
        self.seq_gt: List[pd.DataFrame] = []
        self.seq_T_radar2imu: List[np.ndarray] = []
        self.samples: List[Tuple[int,int]] = []
        self.seq_ids: List[str] = []
        self.seq_start: List[int] = []
        self.seq_npairs: List[int] = []
        for sid, sc in enumerate(self.seqs):
            self.seq_ids.append(sc.name)

            # frame radar: path + timestamp dal nome
            files = sorted(glob.glob(os.path.join(sc.pcd_dir, "*.pcd")))
            if len(files) < 2:
                raise RuntimeError(f"{sc.name}: servono almeno 2 PCD in {sc.pcd_dir}")
            ts = []
            for fp in files:
                t = _parse_ts_from_name(fp)
                ts.append(t)
            if any(t is None for t in ts):
                ts = [float(i) for i in range(len(files))]
            order = np.argsort(np.array(ts))
            files = [files[i] for i in order]
            ts    = [ts[i]    for i in order]
            self.seq_frames.append({"paths": files, "ts": np.array(ts, dtype=np.float64)})

            # GT IMU (local_inspva.txt): ts, tx, ty, tz, qx, qy, qz, qw
            gt = self.load_gt_inspva_txt(sc.gt_path)
            self.seq_gt.append(gt)

            # Calibrazione Radar->IMU
            T_radar2imu = load_radar_to_imu_T(sc.calib_dir)
            self.seq_T_radar2imu.append(T_radar2imu)

            start = len(self.samples)
            n_pairs = max(0, len(files) - 1)
            for i in range(n_pairs):
                self.samples.append((sid, i))
            self.seq_start.append(start)
            self.seq_npairs.append(n_pairs)

    @staticmethod
    def load_gt_inspva_txt(path):
        import io, re
        import pandas as pd

        # 1) tentativi "pandas" con vari encoding e separatori
        encodings = [None, "utf-8", "latin-1", "utf-16", "utf-16le", "utf-16be"]
        seps = [r"\s+", ",", "\t", ";", r"\s*[;,]\s*"]  # whitespace, virgola, tab, puntoevirgola, misti
        df = None

        for enc in encodings:
            for sep in seps:
                try:
                    tmp = pd.read_csv(
                        path,
                        sep=sep,
                        engine="python",
                        comment="#",
                        header=None,
                        encoding=enc,
                        skip_blank_lines=True
                    )
                    # se ha almeno 8 colonne, prendiamo le prime 8
                    if tmp.shape[1] >= 8:
                        df = tmp.iloc[:, :8]
                        break
                except Exception:
                    pass
            if df is not None:
                break

        # 2) fallback manuale: split su qualunque combinazione di spazi/virgole/puntoevirgola
        if df is None or df.shape[1] < 8:
            rows = []
            # se un encoding ha funzionato sopra, riusalo; altrimenti usa 'utf-8' con ignore
            enc_fallback = "utf-8"
            try:
                open(path, "r", encoding=enc_fallback).read(1)
            except Exception:
                enc_fallback = None  # lascia decidere a Python (può comunque fallire su UTF-16)

            with open(path, "r", encoding=enc_fallback, errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.lstrip().startswith("#"):
                        continue
                    # normalizza spazi non-break e simili
                    s = s.replace("\xa0", " ").replace("\t", " ")
                    parts = re.split(r"[,\s;]+", s)
                    if len(parts) >= 8:
                        rows.append(parts[:8])

            if not rows:
                raise ValueError(f"Impossibile parsare GT: {path}")

            df = pd.DataFrame(rows)

        # 3) validazione finale: esattamente 8 colonne con nomi standard
        if df.shape[1] > 8:
            df = df.iloc[:, :8]
        if df.shape[1] < 8:
            raise ValueError(f"GT ha {df.shape[1]} colonne, attese 8: {path}")

        df.columns = ["ts", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]

        # 4) prova a convertire tutto a float; se ci sono stringhe strane, pandas le mette NaN
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # rimuovi eventuali righe vuote/non numeriche
        df = df.dropna().reset_index(drop=True)

        # --- CONVERSIONE TIMESTAMP: ns -> s ---
        # I tuoi timestamp GT sono in *nanosecondi*: converti in secondi.
        # (Usiamo float64 per evitare overflow durante la divisione.)
        ts = df["ts"].to_numpy(dtype=np.float64)
        ts = ts * 1e-9
        df["ts"] = ts

        # --- ORDINA PER TIMESTAMP (robusto a eventuali righe fuori ordine) ---
        df = df.sort_values("ts").reset_index(drop=True)

        if len(df) < 2:
            raise ValueError(f"GT troppo corta o non numerica dopo parsing: {path}")

        return df


    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        N = len(self.samples)
        if idx < 0 or idx >= N:
           raise IndexError(f"__getitem__ out of range: idx={idx}  valid=[0,{N-1}]")
        sid, i = self.samples[idx]
        frames = self.seq_frames[sid]
        gt = self.seq_gt[sid]
        T_radar2imu = self.seq_T_radar2imu[sid]

        a, b = i, i + 1
        pa = frames["paths"][a]; pb = frames["paths"][b]
        tA = frames["ts"][a];    tB = frames["ts"][b]

        # carica punti e porta nel frame IMU
        xyz_a, feat_a = load_pcd_cmflow_like(pa, emit_range_norm=True)
        xyz_b, feat_b = load_pcd_cmflow_like(pb, emit_range_norm=True)
        xyz_a = transform_xyz(T_radar2imu, xyz_a * XYZ_SCALE_CM) / XYZ_SCALE_CM
        xyz_b = transform_xyz(T_radar2imu, xyz_b * XYZ_SCALE_CM) / XYZ_SCALE_CM

        # pacchetto coppia (mantengo stessa API: lista di pair)
        pairs = [(xyz_a, feat_a, xyz_b, feat_b)]
        L = 1  # un solo pair per item (20 Hz)

        # label: nearest GT(tA) -> nearest GT(tB)
        gts = gt["ts"].values.astype(np.float64)
        iA = _nearest_idx(gts, tA)
        iB = _nearest_idx(gts, tB)

        if iB == iA:
            # scegli l'adiacente nella direzione temporale giusta
            if tB >= gts[iA] and iA + 1 < len(gts):
                iB = iA + 1
            elif iA - 1 >= 0:
                iB = iA - 1

        # ordina per tempo
        if gts[iB] < gts[iA]:
            iA, iB = iB, iA

        T0 = pose_to_T(gt.iloc[iA]["tx"], gt.iloc[iA]["ty"], gt.iloc[iA]["tz"],
                    gt.iloc[iA]["qx"], gt.iloc[iA]["qy"], gt.iloc[iA]["qz"], gt.iloc[iA]["qw"])
        T1 = pose_to_T(gt.iloc[iB]["tx"], gt.iloc[iB]["ty"], gt.iloc[iB]["tz"],
                    gt.iloc[iB]["qx"], gt.iloc[iB]["qy"], gt.iloc[iB]["qz"], gt.iloc[iB]["qw"])
        y = delta_pose_local(T0, T1).astype(np.float32)  # (7,)

        return pairs, torch.from_numpy(y), sid, L


# ------------------------ Collate RAW ------------------------
def pad_collate_radar_raw(batch):
    # pairs: list of variable-length lists; keep them as Python objects
    pairs_batch = [b[0] for b in batch]
    y_out       = torch.stack([b[1] for b in batch], dim=0)  # (B,7)
    sid_o       = torch.tensor([b[2] for b in batch], dtype=torch.long)
    lengths     = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return pairs_batch, y_out, sid_o, lengths

# ------------------------ Split + DataLoaders ------------------------
def make_loaders_radar_pose_raw(seqs: List[SeqConfig], batch_size=8, perc=(0.70,0.15,0.15),
                                seed=0, num_workers=0):
    ds = Radar4DDatasetPairsRAW(seqs)

    # 1) Indici REALI per sequenza, derivati da ds.samples (unico "source of truth")
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

        # mantieni ordine temporale (se preferisci casuale: rng.shuffle(idx_all))
        ntr = int(perc[0]*n)
        nva = int(perc[1]*n)

        tr = idx_all[:ntr]
        va = idx_all[ntr:ntr+nva]
        te = idx_all[ntr+nva:]

        idx_train.extend(tr); per_scene_idx["train"][sid] = tr
        idx_val.extend(va);   per_scene_idx["val"][sid]   = va
        idx_test.extend(te);  per_scene_idx["test"][sid]  = te

    # 2) VALIDAZIONE & SANITY CHECK
    N = len(ds) - 1
    def sanitize(tag, arr):
        bad = [i for i in arr if (i < 0 or i > N)]
        if bad:
            print(f"[ERR] {tag}: {len(bad)} indici fuori range (max={N}). Esempi: {bad[:10]}")
            arr = [i for i in arr if (0 <= i <= N)]
            print(f"[FIX] {tag}: dopo filtro -> {len(arr)}")
        else:
            print(f"[OK ] {tag}: {len(arr)} indici (0..{N})")
        return arr

    idx_train = sanitize("train", idx_train)
    idx_val   = sanitize("val",   idx_val)
    idx_test  = sanitize("test",  idx_test)

    # 3) Diagnostica utile una volta
    per_seq_counts = {sid: len(v) for sid, v in sid_to_indices.items()}
    print(f"[DBG] total ds samples={len(ds)}  sum per-seq={sum(per_seq_counts.values())}")
    print(f"[DBG] per-seq counts: {per_seq_counts}")
    print(f"[DBG] split sizes: train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    # 4) DataLoader (per debug iniziale: num_workers=0, batch_size=1)
    train_loader = DataLoader(Subset(ds, idx_train), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=pad_collate_radar_raw)
    val_loader   = DataLoader(Subset(ds, idx_val),   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=pad_collate_radar_raw)
    test_loader  = DataLoader(Subset(ds, idx_test),  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=pad_collate_radar_raw)
    return ds, train_loader, val_loader, test_loader, per_scene_idx





