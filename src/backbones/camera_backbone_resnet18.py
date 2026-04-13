# camera_backbone_resnet18.py
# -*- coding: utf-8 -*-
"""
ResNet18 tronca (torchvision) -> fmap monoscala per fusione A-RCFM.
Compatibile con PyTorch 2.1.2 e torchvision 0.16.2 (CUDA 11.8).

Funzionalità:
- Carica ResNet18 con pesi ImageNet.
- Restituisce UNA sola feature map 2D a stride 8 (layer2) o stride 16 (layer3).
- Proiezione 1x1 verso C_I configurabile (default 64) + GroupNorm.
- Normalizzazione immagini con statistiche ImageNet.
- Metodi utility per congelare/scongelare il backbone in fasi.

Uso tipico:
    from camera_backbone_resnet18 import ImageBackboneResNet18

    img_backbone = ImageBackboneResNet18(out_channels=64, use_stride16=False, freeze_at_start=True)
    fmap = img_backbone(img_bchw)  # (B, 64, H/8, W/8) se use_stride16=False

Note:
- L’encoder NON cambia il formato né la risoluzione in ingresso: accetta (B,3,H,W).
- Se l’input è in [0,255], converte in float e scala a [0,1] internamente.
- Per A-RCFM, tipicamente preferisci stride 8 (più denso).
"""

from typing import Iterable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# torchvision 0.16.2
from torchvision.models import resnet18, ResNet18_Weights


class ImageBackboneResNet18(nn.Module):
    """
    ResNet18 -> piramide di feature map 2D per A-RCFM multi-scala.

    - out_channels: canali delle fmap di uscita (uguali su tutti i livelli)
    - use_stride16: lasciato per compatibilità, ma il forward restituisce SEMPRE 3 scale:
        "l2": stride  8  (layer2)
        "l3": stride 16  (layer3)
        "l4": stride 32  (layer4)
    - freeze_at_start: se True congela tutto il backbone, poi sblocca solo i proiettori 1x1
                       e le GroupNorm (come nella versione mono-scala).
    """
    def __init__(
        self,
        out_channels: int = 64,
        use_stride16: bool = False,
        freeze_at_start: bool = True,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()

        # Carica ResNet18 con pesi ImageNet
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Stem iniziale (/4)
        self.stem = nn.Sequential(
            backbone.conv1,  # stride=2
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,  # totale /4
        )

        # Blocchi residuali
        self.layer1 = backbone.layer1  # /4,  C=64
        self.layer2 = backbone.layer2  # /8,  C=128
        self.layer3 = backbone.layer3  # /16, C=256
        self.layer4 = backbone.layer4  # /32, C=512

        # Manteniamo l'attributo per compatibilità, anche se ora usiamo sempre multi-scala
        self.use_stride16 = bool(use_stride16)

        # Proiezioni a C_I (1x1) + normalizzazione leggera, una per scala
        self.proj_l2 = nn.Conv2d(128, out_channels, kernel_size=1, bias=False)
        self.norm_l2 = nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels)

        self.proj_l3 = nn.Conv2d(256, out_channels, kernel_size=1, bias=False)
        self.norm_l3 = nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels)

        self.proj_l4 = nn.Conv2d(512, out_channels, kernel_size=1, bias=False)
        self.norm_l4 = nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels)

        # Buffers per normalizzazione ImageNet
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        # Congela all'inizio (sbloccando solo proj+norm) se richiesto
        if freeze_at_start:
            self.freeze_backbone()
            # ri-abilita solo i proiettori + norm a tutte le scale
            self.set_trainable(self.proj_l2, True)
            self.set_trainable(self.norm_l2, True)
            self.set_trainable(self.proj_l3, True)
            self.set_trainable(self.norm_l3, True)
            self.set_trainable(self.proj_l4, True)
            self.set_trainable(self.norm_l4, True)

    # -------------------------- Utility di (s)blocco pesi --------------------------

    @staticmethod
    def set_trainable(module: nn.Module, flag: bool) -> None:
        for p in module.parameters(recurse=True):
            p.requires_grad = flag

    def freeze_backbone(self) -> None:
        """
        Congela tutti i parametri del backbone (stem, layer1-4, e anche proj/norm).
        Poi in __init__ ri-abilitiamo i proiettori e le norm se serve.
        """
        self.set_trainable(self, False)

    def unfreeze_backbone(self, stages: Optional[Iterable[str]] = None, include_stem: bool = False) -> None:
        """
        Sblocca selettivamente parti del backbone.
        stages può contenere sottoinsiemi di {"layer1","layer2","layer3","layer4"}.
        Esempio: unfreeze_backbone(stages=("layer2","layer3"))
        """
        if include_stem:
            self.set_trainable(self.stem, True)

        # Se stages è None o vuoto, sblocca tutto
        if not stages:
            self.set_trainable(self, True)
            return

        if "layer1" in stages:
            self.set_trainable(self.layer1, True)
        if "layer2" in stages:
            self.set_trainable(self.layer2, True)
        if "layer3" in stages:
            self.set_trainable(self.layer3, True)
        if "layer4" in stages:
            self.set_trainable(self.layer4, True)

        # Di solito vuoi sempre tenere allenabili i proiettori e le norm
        self.set_trainable(self.proj_l2, True)
        self.set_trainable(self.norm_l2, True)
        self.set_trainable(self.proj_l3, True)
        self.set_trainable(self.norm_l3, True)
        self.set_trainable(self.proj_l4, True)
        self.set_trainable(self.norm_l4, True)

    # -------------------------- Info utili sulla scala --------------------------

    @property
    def feat_stride(self) -> int:
        """
        Per compatibilità: stride "di riferimento".
        Usiamo quello della scala più fine che esponiamo (layer2) => 8.
        """
        return 8

    # -------------------------- Forward --------------------------

    def forward(self, img_bchw: torch.Tensor):
        """
        img_bchw: (B,3,H,W), RGB in [0,1] o [0,255].

        Ritorna un dizionario di feature map normalizzate:
          {
            "l2": (B, C_I, H/8,  W/8),
            "l3": (B, C_I, H/16, W/16),
            "l4": (B, C_I, H/32, W/32),
          }
        """
        x = img_bchw
        if x.dtype != torch.float32:
            x = x.float()
        # porta a [0,1] se necessario
        if x.max() > 1.5:
            x = x / 255.0

        # Normalizzazione ImageNet
        x = (x - self.mean) / self.std

        # Passaggi ResNet
        x = self.stem(x)          # /4
        x = self.layer1(x)        # /4,  C=64
        x2 = self.layer2(x)       # /8,  C=128
        x3 = self.layer3(x2)      # /16, C=256
        x4 = self.layer4(x3)      # /32, C=512

        # Proiezioni e normalizzazione per ciascun livello
        f2 = self.norm_l2(self.proj_l2(x2))   # (B, C_I, H/8,  W/8)
        f3 = self.norm_l3(self.proj_l3(x3))   # (B, C_I, H/16, W/16)
        f4 = self.norm_l4(self.proj_l4(x4))   # (B, C_I, H/32, W/32)

        return {"l2": f2, "l3": f3, "l4": f4}
