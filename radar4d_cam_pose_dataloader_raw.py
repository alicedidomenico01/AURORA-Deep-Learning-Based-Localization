# -*- coding: utf-8 -*-
"""
radar4d_cam_pose_dataloader_raw.py

Dataset RAW stile "gemello" del radar only, esteso con:
- immagini stereo_left sincronizzate (distinte per t e t+1)
- intrinseci K
- extrinseci T_cam<-imu  (COERENTE: i punti radar sono già nel frame IMU)
- etichette delta-pose come nel tuo loader

Non modifichiamo la logica radar: stessa scala, stesse utility, stesso schema di output.

Struttura attesa per una sequenza "01":
  /media/arrubuntu20/HDD/Hercules/01/
    continental_pcd/            (o percorso equivalente ai tuoi .pcd)
    stereo_left/
    PR_GT/local_inspva.txt      (o file GT equivalente)
    Calibration/
      Stereo_LiDAR.txt          (contiene Tr_lidar_to_leftcam:)
      Continental_LiDAR.txt     (contiene Tr_lidar_to_radar:)
      IMU_LiDAR.txt             (contiene Tr_lidar_to_imu:)
      Stereo_left.yaml (o .yamal)  (contiene Intrinsic (K))


"""

import os, re, glob, math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import cv2
import torch
from torch.utils.data import Dataset

# ----------------- Costanti  -----------------
XYZ_SCALE_CM = 100.0
DOP_CLIP_CM  = 20.0
RCS_SCALE_CM = 255.0
EMIT_RANGE_NORM = True

# ============================
# Weather/Illumination supervision (sequence-level)
# ============================
WEATHER_SEV = {"Clear": 0.00, "Cloud": 0.33, "Rain": 0.66, "Snow": 1.00}
ILLUM_SEV   = {"Day": 0.00, "Dusk": 0.50, "Night": 1.00}

SEQ_META = {
    "00": ("Clear", "Day"),
    "01": ("Cloud", "Night"),
    "02": ("Snow",  "Day"),
    "03": ("Clear", "Day"),
    "04": ("Cloud", "Night"),
    "05": ("Snow",  "Day"),
    "06": ("Clear", "Day"),
    "07": ("Cloud", "Night"),
    "08": ("Snow",  "Day"),
    "09": ("Clear", "Day"),
    "10": ("Clear", "Day"),
    "11": ("Cloud", "Night"),
    "12": ("Snow",  "Day"),
    "13": ("Clear", "Day"),
    "14": ("Cloud", "Dusk"),
    "15": ("Cloud", "Day"),
    "16": ("Rain",  "Day"),
    "17": ("Cloud", "Night"),
    "18": ("Rain",  "Day"),
    "19": ("Clear", "Day"),
    "20": ("Cloud", "Night"),
}


DOPPLER_FIELD_CANDIDATES = ["velocity","doppler","vr","radial_velocity"]
RCS_FIELD_CANDIDATES     = ["intensity","rcs","power","reflectivity","amp","amplitude"]


# ----------------- Utility PCD (replicate/compatibili) -------------
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
    with open(path, "rb") as f:
        h = _parse_pcd_header(f)
        fields = h["FIELDS"]; sizes=h.get("SIZE",[4]*len(fields)); types=h.get("TYPE",["F"]*len(fields))
        counts = h.get("COUNT",[1]*len(fields)); datafmt=h["DATA"]
        width=h.get("WIDTH", h.get("POINTS",0)); height=h.get("HEIGHT",1); npts=h.get("POINTS", width*height)
        dtype=np.dtype([(field, np.float32) for field in fields])
        f.seek(h["_header_size"])
        if datafmt=="ascii":
            raw = f.read().decode("utf-8", errors="ignore")
            lines=[ln for ln in raw.splitlines() if ln.strip()]
            if not lines:
                arr=np.empty((0, len(fields)), dtype=np.float32)
            else:
                arr=np.loadtxt(lines, comments="#", dtype=np.float32, ndmin=2)
            out={field: arr[:, i].astype(np.float32) for i,field in enumerate(fields)}
        elif datafmt=="binary":
            raw=f.read(npts*sum(s*c for s,c in zip(sizes, counts)))
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
    feat = np.concatenate([dop, rcs, range_norm], axis=1) if range_norm is not None else np.concatenate([dop, rcs], axis=1)
    return xyz, feat


# --------------- Pose/quat util ---------------
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
    return np.concatenate([dt_local, dq], axis=0)


# ----------------- Calibrazioni -----------------
def _parse_3x4_from_line(s: str, key: str) -> Optional[np.ndarray]:
    if key not in s: return None
    nums = [float(v) for v in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s.split(key,1)[1])]
    if len(nums) >= 12:
        M = np.array(nums[:12], dtype=np.float64).reshape(3,4)
        T = np.eye(4, dtype=np.float64); T[:3,:4] = M
        return T
    return None

def load_T_radar_to_imu(calib_dir: str) -> np.ndarray:
    # Radar->IMU = (LiDAR->IMU) * inv(LiDAR->Radar)
    p_lr = os.path.join(calib_dir, "Continental_LiDAR.txt")
    p_li = os.path.join(calib_dir, "IMU_LiDAR.txt")
    with open(p_lr, "r") as f: txt_lr = f.read()
    with open(p_li, "r") as f: txt_li = f.read()
    T_lidar_to_radar = _parse_3x4_from_line(txt_lr, "Tr_lidar_to_radar:")
    T_lidar_to_imu   = _parse_3x4_from_line(txt_li, "Tr_lidar_to_imu:")
    if T_lidar_to_radar is None or T_lidar_to_imu is None:
        raise ValueError("Parsing calib fallito (Radar/IMU).")
    return T_lidar_to_imu @ np.linalg.inv(T_lidar_to_radar)

def load_cam_K_Tcam_from_lidar(calib_dir: str):
    # K e T_cam<-lidar (da Stereo_LiDAR.txt + Stereo_left.yaml/yamal)
    p_sl = os.path.join(calib_dir, "Stereo_LiDAR.txt")
    with open(p_sl, "r") as f:
        txt = f.read()
    T_lidar_to_cam = _parse_3x4_from_line(txt, "Tr_lidar_to_leftcam:")
    if T_lidar_to_cam is None:
        raise ValueError("Tr_lidar_to_leftcam mancante in Stereo_LiDAR.txt")
    T_cam_from_lidar = T_lidar_to_cam

    p_sy = os.path.join(calib_dir, "Stereo_left.yaml")
    if not os.path.isfile(p_sy):
        alt = os.path.join(calib_dir, "Stereo_left.yamal")
        if os.path.isfile(alt): p_sy = alt
        else: raise FileNotFoundError(f"File intrinseci non trovato: {p_sy}")
    with open(p_sy, "r") as f:
        lines = f.readlines()

    K = None
    for i, ln in enumerate(lines):
        if "Intrinsic" in ln and "K" in ln:
            buf = " ".join(lines[i:i+3])
            nums = [float(v) for v in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", buf)]
            if len(nums) >= 9:
                K = np.array(nums[:9], dtype=np.float64).reshape(3,3)
                break
    if K is None:
        nums = [float(v) for v in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", "\n".join(lines))]
        if len(nums) >= 9:
            K = np.array(nums[:9], dtype=np.float64).reshape(3,3)
        else:
            raise ValueError("Impossibile parsare K da Stereo_left.yaml")
    return K, T_cam_from_lidar

def load_T_imu_from_lidar(calib_dir: str) -> np.ndarray:
    p_li = os.path.join(calib_dir, "IMU_LiDAR.txt")
    with open(p_li, "r") as f:
        txt_li = f.read()
    T_lidar_to_imu = _parse_3x4_from_line(txt_li, "Tr_lidar_to_imu:")
    if T_lidar_to_imu is None:
        raise ValueError("Tr_lidar_to_imu mancante in IMU_LiDAR.txt")
    return T_lidar_to_imu


# ----------------- Proiezioni/trasformazioni ---------------
def transform_xyz(T_4x4: np.ndarray, xyz_m: np.ndarray) -> np.ndarray:
    if xyz_m.size == 0: return xyz_m
    N = xyz_m.shape[0]
    homog = np.concatenate([xyz_m, np.ones((N,1), dtype=np.float32)], axis=1)
    out = (T_4x4.astype(np.float64) @ homog.T).T[:, :3]
    return out.astype(np.float32)

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

def _nearest_idx(arr_sorted: np.ndarray, t: float) -> int:
    i = int(np.searchsorted(arr_sorted, t))
    if i <= 0: return 0
    if i >= len(arr_sorted): return len(arr_sorted)-1
    return i-1 if abs(arr_sorted[i-1]-t) <= abs(arr_sorted[i]-t) else i

def _read_image_rgb_chw(path: str) -> Tuple[torch.Tensor, Tuple[int,int], float]:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Immagine non leggibile: {path}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    H, W = im.shape[:2]
    ten = torch.from_numpy(im).permute(2,0,1).contiguous()  # (3,H,W) uint8
    ts = _parse_ts_from_name(path) or 0.0
    return ten, (H, W), float(ts)

def _pick_two_distinct_images(ts_img: np.ndarray, tA: float, tB: float, tol: float = 0.08) -> Tuple[int,int]:
    """
    Seleziona indici immagini per tA e tB garantendo che siano distinti (camera 15Hz vs radar 20Hz).
    - tol: tolleranza (s) per riassegnamento dell'immagine di tB.
    """
    n = len(ts_img)
    ia = _nearest_idx(ts_img, tA)
    ib = _nearest_idx(ts_img, tB)
    if ia == ib and n > 1:
        # prova il vicino più adatto a tB
        if ib + 1 < n and abs(ts_img[ib + 1] - tB) <= tol:
            ib = ib + 1
        elif ib - 1 >= 0 and abs(ts_img[ib - 1] - tB) <= tol:
            ib = ib - 1
        else:
            # forza diversità in modo deterministico
            if ia + 1 < n:
                ib = ia + 1
            elif ia - 1 >= 0:
                ib = ia - 1
    # monotonia temporale
    if tA <= tB and ts_img[ib] < ts_img[ia]:
        ia, ib = ib, ia
    if tA > tB and ts_img[ib] > ts_img[ia]:
        ia, ib = ib, ia
    return ia, ib


# ----------------- Dataset -----------------
@dataclass
class SeqConfigCam:
    name: str
    pcd_dir: str
    img_dir: str
    gt_path: str
    calib_dir: str
    seq_len: int = 1
    window_stride: int = 1

class Radar4DPlusCamDatasetPairsRAW(Dataset):
    """
    __getitem__ restituisce:
      pairs: [(xyz_t, feat_t, xyz_t1, feat_t1)]
      y:     torch.float32(7,)   # delta pose locale (IMU)
      sid:   int
      L:     int (=1)
      extra: dict con
        img_t, img_t1: torch.uint8 (3,H,W) RGB
        K:              torch.float64 (3,3)
        T_cam_from_imu: torch.float64 (4,4)
        img_size:       (H,W)
        ts_t, ts_t1:    float (s)
        idx_img_t, idx_img_t1: int
        ts_img_t, ts_img_t1:    float (s)
    """
    def __init__(self, seqs: List[SeqConfigCam], enforce_distinct_images: bool = True, img_tolerance_s: float = 0.08):
        super().__init__()
        self.seqs = seqs
        # usa seq_len del config (assumo uguale per tutte; se differisce, prendiamo la prima)
        self.seq_len = int(seqs[0].seq_len) if len(seqs) > 0 else 1
        assert self.seq_len >= 1
        self.enforce_distinct_images = enforce_distinct_images
        self.img_tolerance_s = float(img_tolerance_s)

        self.seq_frames_pcd: List[Dict] = []
        self.seq_frames_img: List[Dict] = []
        self.seq_gt: List[pd.DataFrame] = []
        self.seq_T_radar2imu: List[np.ndarray] = []
        self.seq_K: List[np.ndarray] = []
        self.seq_T_cam_from_imu: List[np.ndarray] = []
        self.samples: List[Tuple[int,int]] = []
        self.seq_ids: List[str] = []
        self.seq_start: List[int] = []
        self.seq_npairs: List[int] = []

        # contatore globale per il debug
        self._dbg_timing_count = 0

        for sid, sc in enumerate(self.seqs):
            self.seq_ids.append(sc.name)

            # Radar frames
            pcd_files = sorted(glob.glob(os.path.join(sc.pcd_dir, "*.pcd")))
            if len(pcd_files) < 2:
                raise RuntimeError(f"{sc.name}: servono >=2 PCD in {sc.pcd_dir}")
            ts_pcd = []
            for fp in pcd_files:
                t = _parse_ts_from_name(fp)
                ts_pcd.append(t if t is not None else float(len(ts_pcd)))
            order = np.argsort(np.array(ts_pcd))
            pcd_files = [pcd_files[i] for i in order]
            ts_pcd    = [ts_pcd[i]    for i in order]
            self.seq_frames_pcd.append({"paths": pcd_files, "ts": np.array(ts_pcd, dtype=np.float64)})

            # Camera frames (stereo_left)
            img_files = sorted(glob.glob(os.path.join(sc.img_dir, "*")))
            img_files = [f for f in img_files if os.path.splitext(f)[1].lower() in (".png",".jpg",".jpeg",".bmp",".tif",".tiff")]
            if len(img_files) == 0:
                raise RuntimeError(f"{sc.name}: nessuna immagine in {sc.img_dir}")
            ts_img = []
            for fp in img_files:
                t = _parse_ts_from_name(fp)
                ts_img.append(t if t is not None else float(len(ts_img)))
            order = np.argsort(np.array(ts_img))
            img_files = [img_files[i] for i in order]
            ts_img    = [ts_img[i]    for i in order]
            self.seq_frames_img.append({"paths": img_files, "ts": np.array(ts_img, dtype=np.float64)})

            # GT IMU
            gt = self._load_gt_inspva_txt(sc.gt_path)
            self.seq_gt.append(gt)

            # Calibrazioni
            T_radar2imu = load_T_radar_to_imu(sc.calib_dir)
            K, T_cam_from_lidar = load_cam_K_Tcam_from_lidar(sc.calib_dir)
            T_lidar_to_imu = load_T_imu_from_lidar(sc.calib_dir)
            T_cam_from_imu = T_cam_from_lidar @ np.linalg.inv(T_lidar_to_imu)

            self.seq_T_radar2imu.append(T_radar2imu)
            self.seq_K.append(K)
            self.seq_T_cam_from_imu.append(T_cam_from_imu)

            # ----------------- Pairs filtrati (solo dove la camera esiste davvero) -----------------
            start = len(self.samples)

            # numero di coppie radar grezze (i, i+1)
            n_pairs_raw = max(0, len(pcd_files) - 1)

            # timestamp radar e camera per questa sequenza
            ts_pcd = self.seq_frames_pcd[sid]["ts"]      # np.array float64
            ts_img = self.seq_frames_img[sid]["ts"]      # np.array float64

            first_img_ts = float(ts_img[0])
            last_img_ts  = float(ts_img[-1])

            # soglia massima ammessa tra radar e camera (in secondi)
            
            MAX_IMG_DT = 0.30
            n_pairs_kept=0
            TOL_GT = 0.05

            
            # costruiamo prima una mask keep sulle coppie singole valide
            # timestamp GT (già in secondi nel tuo loader)
            gts = gt["ts"].values.astype(np.float64)

            keep = np.zeros(n_pairs_raw, dtype=bool)

            for i in range(n_pairs_raw):
                tA = float(ts_pcd[i])
                tB = float(ts_pcd[i+1])

                # 1) entrambi i radar devono cadere nel range temporale coperto dalla camera
                if not (first_img_ts <= tA <= last_img_ts and first_img_ts <= tB <= last_img_ts):
                    continue

                # 2) trova le immagini più vicine a tA e tB
                ia = _nearest_idx(ts_img, tA)
                ib = _nearest_idx(ts_img, tB)

                dtA = abs(float(ts_img[ia]) - tA)
                dtB = abs(float(ts_img[ib]) - tB)

                # 3) se la camera più vicina è comunque troppo lontana, scarta la coppia
                if dtA > MAX_IMG_DT or dtB > MAX_IMG_DT:
                    continue

                # >>> AGGIUNGI: nearest su GT + scarto se troppo lontano
                iA = _nearest_idx(gts, tA)
                iB = _nearest_idx(gts, tB)

                if abs(float(gts[iA]) - tA) > TOL_GT or abs(float(gts[iB]) - tB) > TOL_GT:
                    continue

                keep[i] = True

            # ora creiamo gli start-index validi per sequenze lunghe L (es: 4)
            L = getattr(self, "seq_len", 1)  # o self.seq_len se l'hai già salvata in __init__
            n_pairs_kept = int(keep.sum())
            n_starts_kept = 0
            stride = getattr(sc, "window_stride", 1)
            stride = max(1, int(stride))

            for i0 in range(0, n_pairs_raw - L + 1, stride):
                # deve essere valida la finestra di L coppie consecutive
                if keep[i0:i0+L].all():
                    self.samples.append((sid, i0))  
                    n_starts_kept += 1

            self.seq_start.append(start)
            self.seq_npairs.append(n_starts_kept)

            print(
                f"[DBG FILTER] seq {sc.name}: kept {n_pairs_kept} / {n_pairs_raw} single pairs; "
                f"starts {n_starts_kept} for seq_len={L}"
            )


    def __len__(self): return len(self.samples)

    # ---- robust GT loader (come nel tuo) ----
    @staticmethod
    def _load_gt_inspva_txt(path):
        encodings = [None, "utf-8", "latin-1", "utf-16", "utf-16le", "utf-16be"]
        seps = [r"\s+", ",", "\t", ";", r"\s*[;,]\s*"]
        df = None
        for enc in encodings:
            for sep in seps:
                try:
                    tmp = pd.read_csv(path, sep=sep, engine="python", comment="#", header=None, encoding=enc, skip_blank_lines=True)
                    if tmp.shape[1] >= 8:
                        df = tmp.iloc[:, :8]; break
                except Exception:
                    pass
            if df is not None: break
        if df is None or df.shape[1] < 8:
            rows = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.lstrip().startswith("#"): continue
                    s = s.replace("\xa0"," ").replace("\t"," ")
                    parts = re.split(r"[,\s;]+", s)
                    if len(parts) >= 8: rows.append(parts[:8])
            if not rows: raise ValueError(f"Impossibile parsare GT: {path}")
            df = pd.DataFrame(rows)
        if df.shape[1] > 8: df = df.iloc[:, :8]
        if df.shape[1] < 8: raise ValueError(f"GT ha {df.shape[1]} colonne, attese 8: {path}")
        df.columns = ["ts","tx","ty","tz","qx","qy","qz","qw"]
        for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().reset_index(drop=True)
        ts = df["ts"].to_numpy(dtype=np.float64) * 1e-9
        df["ts"] = ts
        df = df.sort_values("ts").reset_index(drop=True)
        return df

    def __getitem__(self, idx):
        sid, i0 = self.samples[idx]
        fr_pcd = self.seq_frames_pcd[sid]
        fr_img = self.seq_frames_img[sid]
        gt     = self.seq_gt[sid]
        T_radar2imu = self.seq_T_radar2imu[sid]
        K_cam       = self.seq_K[sid]
        T_cam_from_imu = self.seq_T_cam_from_imu[sid]

        # L = lunghezza sequenza (es: 4)
        L = getattr(self, "seq_len", 1)
        assert L >= 1

        pairs = []

        # liste per extra (una entry per step k)
        img_t_list   = []
        img_t1_list  = []
        img_size_list = []
        ts_t_list    = []
        ts_t1_list   = []
        idx_img_t_list  = []
        idx_img_t1_list = []
        ts_img_t_list   = []
        ts_img_t1_list  = []

        
        ts_img = fr_img["ts"]
        gts = gt["ts"].values.astype(np.float64)

        # per label (la facciamo sull'ultima coppia)
        tA_last = None
        tB_last = None
        ia_last = None
        ib_last = None
        tsA_last = None
        tsB_last = None
        iA_last = None
        iB_last = None

        for k in range(L):
            a = i0 + k
            b = i0 + k + 1

            pa, pb = fr_pcd["paths"][a], fr_pcd["paths"][b]
            tA, tB = fr_pcd["ts"][a],   fr_pcd["ts"][b]

            # --- Radar: carica e porta nel frame IMU ---
            xyz_a, feat_a = load_pcd_cmflow_like(pa, emit_range_norm=True)
            xyz_b, feat_b = load_pcd_cmflow_like(pb, emit_range_norm=True)
            xyz_a = transform_xyz(T_radar2imu, xyz_a * XYZ_SCALE_CM) / XYZ_SCALE_CM
            xyz_b = transform_xyz(T_radar2imu, xyz_b * XYZ_SCALE_CM) / XYZ_SCALE_CM

            pairs.append((xyz_a, feat_a, xyz_b, feat_b))

            # --- Selezione immagini (garantisce immagini diverse) ---
            ia, ib = _pick_two_distinct_images(ts_img, tA, tB, tol=self.img_tolerance_s)
            imgA_path = fr_img["paths"][ia]
            imgB_path = fr_img["paths"][ib]
            imgA, (Ha,Wa), tsA = _read_image_rgb_chw(imgA_path)
            imgB, (Hb,Wb), tsB = _read_image_rgb_chw(imgB_path)

            # accumula extra per questo step
            img_t_list.append(imgA)
            img_t1_list.append(imgB)
            img_size_list.append((Ha, Wa))
            ts_t_list.append(float(tA))
            ts_t1_list.append(float(tB))
            idx_img_t_list.append(int(ia))
            idx_img_t1_list.append(int(ib))
            ts_img_t_list.append(float(tsA))
            ts_img_t1_list.append(float(tsB))

            # tieni gli ultimi per label e debug timing (ultimo step k=L-1)
            if k == L - 1:
                tA_last, tB_last = float(tA), float(tB)
                ia_last, ib_last = int(ia), int(ib)
                tsA_last, tsB_last = float(tsA), float(tsB)

                # --- Label delta pose IMU (nearest su GT come nel tuo) ---
                iA = _nearest_idx(gts, tA_last)
                iB = _nearest_idx(gts, tB_last)
                if iB == iA:
                    if tB_last >= gts[iA] and iA + 1 < len(gts):
                        iB = iA + 1
                    elif iA - 1 >= 0:
                        iB = iA - 1
                if gts[iB] < gts[iA]:
                    iA, iB = iB, iA
                iA_last, iB_last = int(iA), int(iB)

        # --- DEBUG IMG  ---
        # qui stampo le shape di step 0 (puoi anche mettere step L-1 se preferisci)
        if idx < 5:
            imgA0 = img_t_list[0]
            imgB0 = img_t1_list[0]
            print("[DBG IMG]", self.seqs[sid].name,
                "idx", idx,
                "L", L,
                "imgA0 shape", imgA0.shape, "min/max", imgA0.min().item(), imgA0.max().item(),
                "imgB0 shape", imgB0.shape, "min/max", imgB0.min().item(), imgB0.max().item())

        # --- DEBUG TEMPORALE per le prime 5 sequenze globali ---
        # stampiamo il timing dell'ULTIMA coppia (k=L-1)
        if self._dbg_timing_count < 5:
            seq_name = self.seqs[sid].name if hasattr(self, "seqs") else str(sid)
            print("\n[DBG TIMING] seq", seq_name, "sample idx", idx,
                "start", i0, "L", L,
                "last pair", (i0 + L - 1), "->", (i0 + L))
            print(f"  radar tA = {tA_last:.6f}, tB = {tB_last:.6f}")
            print(f"  img  A  : idx={ia_last:4d}, ts_imgA={tsA_last:.6f}, |tA-tsA|={abs(tA_last-tsA_last):.6f} s")
            print(f"  img  B  : idx={ib_last:4d}, ts_imgB={tsB_last:.6f}, |tB-tsB|={abs(tB_last-tsB_last):.6f} s")
            print(f"  gt   A  : idx={iA_last:4d}, ts_gtA ={gts[iA_last]:.6f}, |tA-ts_gtA|={abs(tA_last-gts[iA_last]):.6f} s")
            print(f"  gt   B  : idx={iB_last:4d}, ts_gtB ={gts[iB_last]:.6f}, |tB-ts_gtB|={abs(tB_last-gts[iB_last]):.6f} s")
            self._dbg_timing_count += 1

        # --- label y: delta pose locale sull'ULTIMA coppia (k=L-1) ---
        T0 = pose_to_T(gt.iloc[iA_last]["tx"], gt.iloc[iA_last]["ty"], gt.iloc[iA_last]["tz"],
                    gt.iloc[iA_last]["qx"], gt.iloc[iA_last]["qy"], gt.iloc[iA_last]["qz"], gt.iloc[iA_last]["qw"])
        T1 = pose_to_T(gt.iloc[iB_last]["tx"], gt.iloc[iB_last]["ty"], gt.iloc[iB_last]["tz"],
                    gt.iloc[iB_last]["qx"], gt.iloc[iB_last]["qy"], gt.iloc[iB_last]["qz"], gt.iloc[iB_last]["qw"])
        y = delta_pose_local(T0, T1).astype(np.float32)

        extra = {
            # ora sono LISTE length L
            "img_t":  img_t_list,          # list of torch.uint8 (3,H,W)
            "img_t1": img_t1_list,
            "K":      torch.from_numpy(K_cam.astype(np.float64)),                   # (3,3)
            "T_cam_from_imu": torch.from_numpy(T_cam_from_imu.astype(np.float64)),  # (4,4)
            "img_size": img_size_list,     # list of (H,W)
            "ts_t": ts_t_list,             # list of float
            "ts_t1": ts_t1_list,           # list of float
            "idx_img_t": idx_img_t_list,
            "idx_img_t1": idx_img_t1_list,
            "ts_img_t": ts_img_t_list,
            "ts_img_t1": ts_img_t1_list,
        }
        # --- Weather/Illum supervision (sequence-level) ---
        seq_name = self.seqs[sid].name  # "00", "01", ...
        w_str, t_str = SEQ_META.get(seq_name, ("Clear", "Day"))  # fallback safe
        extra["weather_sev"] = float(WEATHER_SEV[w_str])
        extra["illum_sev"]   = float(ILLUM_SEV[t_str])

        return pairs, torch.from_numpy(y), sid, L, extra
