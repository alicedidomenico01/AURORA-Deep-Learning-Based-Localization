# -*- coding: utf-8 -*-
# radar4d_encoder_pn2.py

import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils as p2u

def as_BCN(x):  # (B,N,C)->(B,C,N)
    return x.transpose(1, 2).contiguous()

def as_BNC(x):  # (B,C,N)->(B,N,C)
    return x.transpose(1, 2).contiguous()

class MLP2d(nn.Module): #applicata ad ogni coppia di punti (quindi col vicino, tra tutti i vicini prendi il più informativo)
    def __init__(self, channels):
        super().__init__()
        layers=[]
        for i in range(len(channels)-1):
            layers += [
                nn.Conv2d(channels[i], channels[i+1], 1, bias=False),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(inplace=True)
            ]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class MLP1d(nn.Module): #trasforma feature per punto
    def __init__(self, channels):
        super().__init__()
        layers=[]
        for i in range(len(channels)-1):
            layers += [
                nn.Conv1d(channels[i], channels[i+1], 1, bias=False),
                nn.BatchNorm1d(channels[i+1]),
                nn.ReLU(inplace=True)
            ]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class SetConvNoDown(nn.Module):
    """
    PointNet++-style senza downsample: per ogni punto usa ball_query su più scale,
    concatena [rel_xyz | feat_vicini], MLP2d + max su nsample, poi fuse 1x1.
    radii_norm sono nella stessa scala delle tue xyz normalizzate (es. xyz/100 -> 0.02 ~ 2m).
    """
    def __init__(self, radii_norm, nsamples, in_ch, out_ch_per_scale):
        super().__init__()
        assert len(radii_norm)==len(nsamples)
        self.radii = radii_norm
        self.ns = nsamples
        self.branches = nn.ModuleList([
            MLP2d([in_ch + 3, out_ch_per_scale//2, out_ch_per_scale])
            for _ in self.radii
        ])
        self.fuse = MLP1d([len(self.radii)*out_ch_per_scale, out_ch_per_scale, out_ch_per_scale])

    def forward(self, xyz, feat):
        # xyz:(B,N,3) feat:(B,N,Cin) -> (B,N,out_ch_per_scale)
        xyz  = xyz.contiguous()
        feat = feat.contiguous() if feat is not None else None

        # Database e centroidi sono entrambi xyz (no downsample)
        base_xyz = xyz.contiguous()   # (B,Nd,3)
        new_xyz  = xyz.contiguous()   # (B,Nc,3) — qui Nc==Nd

        xyz_BCN  = as_BCN(base_xyz)   # (B,3,Nd)
        feat_BCN = as_BCN(feat) if feat is not None else None  # (B,Cin,Nd)

        outs=[]
        for r, k, mlp in zip(self.radii, self.ns, self.branches):
            # ✅ ordine giusto: database PRIMA, centroidi DOPO
            idx = p2u.ball_query(r, k, base_xyz, new_xyz)                # (B,Nc,k)

            grouped_xyz = p2u.grouping_operation(xyz_BCN, idx).contiguous()    # (B,3,Nc,k)
            new_xyz_BCN = as_BCN(new_xyz)                                        # (B,3,Nc)
            center_xyz  = new_xyz_BCN.unsqueeze(-1).expand_as(grouped_xyz).contiguous()
            rel_xyz = (grouped_xyz - center_xyz).contiguous()                    # (B,3,Nc,k)

            if feat_BCN is not None:
                grouped_feat = p2u.grouping_operation(feat_BCN, idx).contiguous()  # (B,Cin,Nc,k)
                x = torch.cat([rel_xyz, grouped_feat], dim=1).contiguous()         # (B,3+Cin,Nc,k)
            else:
                x = rel_xyz

            x = mlp(x)                           # (B,out,Nc,k)
            x = x.max(dim=-1).values             # (B,out,Nc)
            outs.append(x)

        x = torch.cat(outs, dim=1).contiguous()  # (B,out*#scales,Nc)
        x = self.fuse(x)                         # (B,out,Nc)
        return as_BNC(x)                         # (B,Nc,out)

class LocalCostVolume(nn.Module):
    """
    Cost volume locale: per ogni punto in t (centroidi), prendo vicini in t1 (supporto).
    """
    def __init__(self, in_ch_t, in_ch_t1, out_ch, radius_norm, nsample):
        super().__init__()
        self.r = radius_norm
        self.k = nsample
        self.mlp = MLP2d([in_ch_t + in_ch_t1 + 3, out_ch//2, out_ch])

    def forward(self, xyz_t, feat_t, xyz_t1, feat_t1):
        # Contiguità per op CUDA
        xyz_t   = xyz_t.contiguous()     # coordinate (B,Nc,3)
        xyz_t1  = xyz_t1.contiguous()    #   (B,Nd,3)
        feat_t  = feat_t.contiguous()    #features
        feat_t1 = feat_t1.contiguous()

        new_xyz  = xyz_t  #punti su cui voglio la feature   
        base_xyz = xyz_t1   #punti su cui cerco i vicini 

        # ✅ per ogni punto t trova i vicini in t1
        idx = p2u.ball_query(self.r, self.k, base_xyz, new_xyz)  #per ogni punto di t trova k vicini in t+1 entro il raggio r

        base_xyz_BCN = as_BCN(base_xyz)                                  # prendere coordinate (B,3,Nd)
        grouped_xyz_t1 = p2u.grouping_operation(base_xyz_BCN, idx).contiguous()  # (B,3,Nc,k)

        new_xyz_BCN = as_BCN(new_xyz)                                    # (B,3,Nc)
        center_xyz_t = new_xyz_BCN.unsqueeze(-1).expand_as(grouped_xyz_t1).contiguous()
        rel = (grouped_xyz_t1 - center_xyz_t).contiguous()               # offset relativo

        ft_BCN  = as_BCN(feat_t)                                         # (B,Ct,Nc)   (assumiamo N matcha Nc)
        ft1_BCN = as_BCN(feat_t1)                                        # (B,Ct1,Nd)

        grouped_ft1 = p2u.grouping_operation(ft1_BCN, idx).contiguous()  # prrendo feature vicini (B,Ct1,Nc,k)
        ft_rep = ft_BCN.unsqueeze(-1).expand(-1, -1, ft_BCN.size(2), self.k).contiguous()  # (B,Ct,Nc,k)

        x = torch.cat([ft_rep, grouped_ft1, rel], dim=1).contiguous()    # feature punto t e t+1 e offset (B,Ct+Ct1+3,Nc,k)
        x = self.mlp(x)                                                  # (B,out,Nc,k)
        x = x.max(dim=-1).values                                         # (B,out,Nc)
        return as_BNC(x)                                                 # (B,Nc,out)
    
class SetAbstractionMSG(nn.Module):
    """
    Set Abstraction Multi-Scale Grouping stile PointNet++:
      - Farthest Point Sampling per ottenere new_xyz (centroidi)
      - ball_query su più raggi
      - grouping_operation per xyz e feat
      - MLP2d + max pooling su nsample
      - fusione 1x1 su canale (MLP1d)
    """
    def __init__(self, npoint, radii_norm, nsamples, in_ch, mlp_out_per_scale):
        super().__init__()
        assert len(radii_norm) == len(nsamples)
        self.npoint = npoint
        self.radii = radii_norm
        self.ns = nsamples

        # Un MLP2d per ciascun raggio
        self.branches = nn.ModuleList([
            MLP2d([in_ch + 3, mlp_out_per_scale // 2, mlp_out_per_scale])
            for _ in self.radii
        ])

        # Fusione finale tra le diverse scale
        self.fuse = MLP1d([
            len(self.radii) * mlp_out_per_scale,
            mlp_out_per_scale,
            mlp_out_per_scale
        ])

    def forward(self, xyz, feat):
        """
        xyz:  (B, N, 3)
        feat: (B, N, Cin) oppure None
        Ritorna:
          new_xyz:  (B, npoint_eff, 3)
          new_feat: (B, npoint_eff, Cout)
        dove npoint_eff = min(self.npoint, N).
        """
        B, N, _ = xyz.shape
        xyz = xyz.contiguous()
        if feat is not None:
            feat = feat.contiguous()

        # npoint effettivo = min(self.npoint, N) per evitare crash se N < self.npoint
        npoint = min(self.npoint, N)

        # 1) Farthest Point Sampling
        idx_fps = p2u.furthest_point_sample(xyz, npoint)  # n points ben spaziati (B, npoint)
        xyz_BCN = as_BCN(xyz)                             # (B,3,N)
        new_xyz = as_BNC(p2u.gather_operation(xyz_BCN, idx_fps))  # (B,npoint,3)

        feat_BCN = as_BCN(feat) if feat is not None else None  # (B,Cin,N) oppure None

        outs = []
        for r, k, mlp in zip(self.radii, self.ns, self.branches): #multi-scale loop
            # 2) Neighborhood: database = xyz, centroidi = new_xyz
            idx = p2u.ball_query(r, k, xyz, new_xyz)                 #trova vicini (B,npoint,k)

            grouped_xyz = p2u.grouping_operation(xyz_BCN, idx)       # (B,3,npoint,k)
            new_xyz_BCN = as_BCN(new_xyz)                            # (B,3,npoint)
            center = new_xyz_BCN.unsqueeze(-1).expand_as(grouped_xyz)
            rel_xyz = (grouped_xyz - center).contiguous()            #calcola coordinate relative (B,3,npoint,k)

            if feat_BCN is not None:
                grouped_feat = p2u.grouping_operation(feat_BCN, idx) # (B,Cin,npoint,k)
                x = torch.cat([rel_xyz, grouped_feat], dim=1)        # (B,3+Cin,npoint,k)
            else:
                x = rel_xyz                                          # (B,3,npoint,k)

            x = mlp(x)                  # trasforma la descrizione di ogni vicino (punto per punto, vicino per vicino) (B,out,npoint,k)
            x = x.max(dim=-1).values      # per ogni centroide, prendi la risposta più forte tra i suoi vicini
            outs.append(x)

        x = torch.cat(outs, dim=1)               # (B,out*#scales,npoint)
        x = self.fuse(x)                         # (B,out,npoint)
        new_feat = as_BNC(x)                     # (B,npoint,out)

        return new_xyz, new_feat



class Radar4DEncoderPN2(nn.Module):
    """
    Encoder Radar 4D in stile PointNet++ multi-scala con:
      - 4 livelli di Set Abstraction (MSG) sia per t che per t+1
      - cost volume locale a ciascun livello
      - propagazione coarse-to-fine (lvl4 -> lvl1) con three_nn + three_interpolate

    NOTA: ho mantenuto la firma del costruttore compatibile con la versione precedente,
    
    """
    def __init__(self,
                 in_ch=3,
                 radii_norm=(0.01, 0.02, 0.05),   # non usato 1:1, solo per scalare i raggi
                 nsamples=(8, 16, 32),            # idem
                 width=64,
                 cv_radius_norm=0.05,             # non più usato direttamente
                 cv_nsample=32,                   # non più usato direttamente
                 out_ch=256):
        super().__init__()
        self.in_ch = in_ch
        self.width = width
        self.out_ch = out_ch

        # -------------------------------------------------
        # 1) Iperparametri piramide (puoi aggiustarli)
        #    n1..n4 DEVONO essere <= numero di punti radar per frame
        # -------------------------------------------------
        n1, n2, n3, n4 = 256, 128, 64, 32

        # Raggio base da cui scalare (se passi tuple diverse, prendo il primo)
        base_r = radii_norm[0] if len(radii_norm) > 0 else 0.02

        # Raggi multi-scala per ciascun livello (due scale per livello)
        r1 = (base_r * 1.0, base_r * 2.0)
        r2 = (base_r * 2.0, base_r * 4.0)
        r3 = (base_r * 4.0, base_r * 8.0)
        r4 = (base_r * 8.0, base_r * 16.0)

        # Numero di vicini per ciascun raggio
        k1 = (16, 32)
        k2 = (16, 32)
        k3 = (16, 32)
        k4 = (16, 32)

        # -------------------------------------------------
        # 2) Set Abstraction MSG per t
        # -------------------------------------------------
        self.sa1_t = SetAbstractionMSG(
            npoint=n1,
            radii_norm=r1,
            nsamples=k1,
            in_ch=in_ch,
            mlp_out_per_scale=width,
        )
        self.sa2_t = SetAbstractionMSG(
            npoint=n2,
            radii_norm=r2,
            nsamples=k2,
            in_ch=width,
            mlp_out_per_scale=width,
        )
        self.sa3_t = SetAbstractionMSG(
            npoint=n3,
            radii_norm=r3,
            nsamples=k3,
            in_ch=width,
            mlp_out_per_scale=width,
        )
        self.sa4_t = SetAbstractionMSG(
            npoint=n4,
            radii_norm=r4,
            nsamples=k4,
            in_ch=width,
            mlp_out_per_scale=width,
        )

        # -------------------------------------------------
        # 3) Set Abstraction MSG per t+1 (stessa struttura)
        # -------------------------------------------------
        self.sa1_t1 = SetAbstractionMSG(n1, r1, k1, in_ch,  width)
        self.sa2_t1 = SetAbstractionMSG(n2, r2, k2, width, width)
        self.sa3_t1 = SetAbstractionMSG(n3, r3, k3, width, width)
        self.sa4_t1 = SetAbstractionMSG(n4, r4, k4, width, width)

        # -------------------------------------------------
        # 4) Cost volume a 4 livelli (più grossolani -> r più grandi)
        # -------------------------------------------------
        cv_ch = width  # puoi cambiarlo se vuoi più canali
        self.cv4 = LocalCostVolume(width, width, cv_ch, radius_norm=r4[1], nsample=32)
        self.cv3 = LocalCostVolume(width, width, cv_ch, radius_norm=r3[1], nsample=32)
        self.cv2 = LocalCostVolume(width, width, cv_ch, radius_norm=r2[1], nsample=32)
        self.cv1 = LocalCostVolume(width, width, cv_ch, radius_norm=r1[1], nsample=32)

        # -------------------------------------------------
        # 5) Propagazione coarse-to-fine 4 -> 1
        #    (x4 -> x3 -> x2 -> x1)
        # -------------------------------------------------
        # livello 4: concat([feat4, cv4]) -> width
        self.proj4 = MLP1d([width + cv_ch, width, width])

        # livello 3: concat([feat3, cv3, x4_up3]) -> width
        self.proj3 = MLP1d([width + cv_ch + width, width, width])

        # livello 2: concat([feat2, cv2, x3_up2]) -> width
        self.proj2 = MLP1d([width + cv_ch + width, width, width])

        # livello 1: concat([feat1, cv1, x2_up1]) -> width
        self.proj1 = MLP1d([width + cv_ch + width, width, width])

        # Head finale sul livello 1
        self.out_head = MLP1d([width, out_ch])

    # -------------------------------------------------
    # Helper: piramide di feature radar
    # -------------------------------------------------
    def encode_pyramid(self, xyz, feat, which="t"):
        """
        xyz:  (B, N, 3)
        feat: (B, N, Cin)
        which: "t" oppure "t1" per scegliere i SA corretti

        Ritorna:
          xyzs  = [xyz1, xyz2, xyz3, xyz4]   coordinate per livello
          feats = [f1,   f2,   f3,   f4]     feature per livello
        """
        if which == "t":
            sa1, sa2, sa3, sa4 = self.sa1_t, self.sa2_t, self.sa3_t, self.sa4_t
        else:
            sa1, sa2, sa3, sa4 = self.sa1_t1, self.sa2_t1, self.sa3_t1, self.sa4_t1

        xyz1, f1 = sa1(xyz, feat)      # (B,N1, C)
        xyz2, f2 = sa2(xyz1, f1)       # (B,N2, C)
        xyz3, f3 = sa3(xyz2, f2)       # (B,N3, C)
        xyz4, f4 = sa4(xyz3, f3)       # (B,N4, C)

        return [xyz1, xyz2, xyz3, xyz4], [f1, f2, f3, f4]

    # -------------------------------------------------
    # Helper: cost volume multi-livello
    # -------------------------------------------------
    def cost_pyramid(self, xyzs_t, feats_t, xyzs_t1, feats_t1):
        """
        xyzs_t : [xyz1, xyz2, xyz3, xyz4] per t
        feats_t: [f1,   f2,   f3,   f4] per t
        idem per t1

        Ritorna lista [cv1, cv2, cv3, cv4] nello stesso ordine.
        """
        cv4 = self.cv4(xyzs_t[3], feats_t[3], xyzs_t1[3], feats_t1[3])
        cv3 = self.cv3(xyzs_t[2], feats_t[2], xyzs_t1[2], feats_t1[2])
        cv2 = self.cv2(xyzs_t[1], feats_t[1], xyzs_t1[1], feats_t1[1])
        cv1 = self.cv1(xyzs_t[0], feats_t[0], xyzs_t1[0], feats_t1[0])
        return [cv1, cv2, cv3, cv4]

    # -------------------------------------------------
    # Helper: interpolazione 3-NN da sorgente (coarse) a destinazione (fine)
    # -------------------------------------------------
    def _interp(self, xyz_src, feat_src, xyz_dst):
        """
        xyz_src:  (B, Ns, 3)
        feat_src: (B, Ns, C)
        xyz_dst:  (B, Nd, 3)

        Ritorna feat_dst: (B, Nd, C) tramite three_nn + three_interpolate.
        """
        xyz_src = xyz_src.contiguous()
        xyz_dst = xyz_dst.contiguous()
        feat_src = feat_src.contiguous()

        xyz_src_BCN = as_BCN(xyz_src)         # (B,3,Ns)
        xyz_dst_BCN = as_BCN(xyz_dst)         # (B,3,Nd)

        dists, idx = p2u.three_nn(xyz_dst_BCN, xyz_src_BCN)          # (B,Nd,3), (B,Nd,3)
        weights = 1.0 / (dists + 1e-8)
        weights = weights / weights.sum(dim=-1, keepdim=True)        # (B,Nd,3)

        feat_src_BCN = as_BCN(feat_src)       # (B,C,Ns)
        feat_interp = p2u.three_interpolate(feat_src_BCN, idx, weights)  # (B,C,Nd)
        return as_BNC(feat_interp)            # (B,Nd,C)

    # -------------------------------------------------
    # Forward: cost volume + coarse-to-fine 4 -> 1
    # (versione "solo radar", senza camera)
    # -------------------------------------------------
    def forward(self, xyz_t, feat_t, xyz_t1, feat_t1):
        """
        xyz_t,  feat_t:  (B, N, 3), (B, N, Cin) a tempo t
        xyz_t1, feat_t1: (B, N, 3), (B, N, Cin) a tempo t+1

        Ritorna:
          E: (B, N1, out_ch) sul livello più fine (lvl1) dopo coarse-to-fine.
        """
        xyz_t  = xyz_t.contiguous()
        feat_t = feat_t.contiguous()
        xyz_t1 = xyz_t1.contiguous()
        feat_t1 = feat_t1.contiguous()

        # 1) piramidi radar
        xyzs_t,  feats_t  = self.encode_pyramid(xyz_t,  feat_t,  which="t")
        xyzs_t1, feats_t1 = self.encode_pyramid(xyz_t1, feat_t1, which="t1")

        # 2) cost volume a tutti i livelli
        cvs = self.cost_pyramid(xyzs_t, feats_t, xyzs_t1, feats_t1)
        cv1, cv2, cv3, cv4 = cvs

        # 3) livello 4 (coarsest)
        x4 = torch.cat([feats_t[3], cv4], dim=-1)      # (B,N4,C+Cv)
        x4 = self.proj4(as_BCN(x4))
        x4 = as_BNC(x4)                                # (B,N4,width)

        # 4) lvl4 -> lvl3
        x4_up3 = self._interp(xyzs_t[3], x4, xyzs_t[2])  # (B,N3,width)
        x3 = torch.cat([feats_t[2], cv3, x4_up3], dim=-1)
        x3 = self.proj3(as_BCN(x3))
        x3 = as_BNC(x3)                                # (B,N3,width)

        # 5) lvl3 -> lvl2
        x3_up2 = self._interp(xyzs_t[2], x3, xyzs_t[1])  # (B,N2,width)
        x2 = torch.cat([feats_t[1], cv2, x3_up2], dim=-1)
        x2 = self.proj2(as_BCN(x2))
        x2 = as_BNC(x2)                                # (B,N2,width)

        # 6) lvl2 -> lvl1
        x2_up1 = self._interp(xyzs_t[1], x2, xyzs_t[0])  # (B,N1,width)
        x1 = torch.cat([feats_t[0], cv1, x2_up1], dim=-1)
        x1 = self.proj1(as_BCN(x1))
        x1 = as_BNC(x1)                                # (B,N1,width)

        # 7) head finale sul livello più fine
        E = self.out_head(as_BCN(x1))                  # (B,out_ch,N1)
        return as_BNC(E)                               # (B,N1,out_ch)
