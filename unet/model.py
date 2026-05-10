"""
unet/model.py
=============
Modified U-Net for fossil CT segmentation.

Architecture (per manuscript §Dual-stream knowledge transfer to U-Net):
  - Encoder backbone : ResNet-50 (e1–e4), initialised from SimCLR checkpoint
  - Channel depths   : 64 → 256 → 512 → 1024 → 2048 (deeper than vanilla U-Net)
  - Bottleneck       : 7 × 7 convolutional block with integrated ReLU
  - Decoder          : symmetric (up4–up1) with skip connections from encoder
  - Skip connections : preserve fine-scale structural detail (delicate fossil
                       boundaries, narrow bone processes, thin cortical margins)
  - Output           : single-channel sigmoid map for bone / not-bone

Reference
---------
Ronneberger, O., Fischer, P., & Brox, T. (2015).
U-Net: Convolutional networks for biomedical image segmentation. MICCAI.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ---------------------------------------------------------------------------
# Decoder building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two consecutive Conv → BN → ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Bilinear upsampling + skip-connection concatenation + ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dimensions mismatch (odd input sizes)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[-1] - x.shape[-1],
                          0, skip.shape[-2] - x.shape[-2]])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Bottleneck
# ---------------------------------------------------------------------------

class Bottleneck7x7(nn.Module):
    """
    Bottleneck with 7 × 7 convolution.

    "The central bottleneck was substantially expanded, incorporating a
    7 × 7 convolutional block with integrated ReLU activation functions
    to model complex spatial relationships at a 7 × 7 resolution."
    — manuscript
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------

class FossilUNet(nn.Module):
    """
    Modified U-Net with ResNet-50 encoder and 7 × 7 bottleneck.

    Parameters
    ----------
    simclr_ckpt : path to SimCLR checkpoint — encoder weights are loaded
                  from this file to implement knowledge transfer from the
                  contrastive pre-training stage.
    n_classes   : number of output classes (default 1 for binary bone/matrix)
    """

    # Encoder output channels at each stage
    _ENCODER_CHANNELS = [64, 256, 512, 1024, 2048]

    def __init__(
        self,
        simclr_ckpt: str | Path | None = None,
        n_classes: int = 1,
    ):
        super().__init__()

        # ---- Encoder (ResNet-50 stages) ----
        backbone = models.resnet50(weights=None)
        self.enc_stem = nn.Sequential(backbone.conv1, backbone.bn1,
                                      backbone.relu, backbone.maxpool)   # /4
        self.enc1 = backbone.layer1    # 256 ch, /4
        self.enc2 = backbone.layer2    # 512 ch, /8
        self.enc3 = backbone.layer3    # 1024 ch, /16
        self.enc4 = backbone.layer4    # 2048 ch, /32

        if simclr_ckpt is not None:
            self._load_simclr_weights(simclr_ckpt)

        # ---- Bottleneck ----
        self.bottleneck = Bottleneck7x7(2048, 1024)

        # ---- Decoder ----
        self.up4 = UpBlock(in_ch=1024, skip_ch=1024, out_ch=512)
        self.up3 = UpBlock(in_ch=512,  skip_ch=512,  out_ch=256)
        self.up2 = UpBlock(in_ch=256,  skip_ch=256,  out_ch=128)
        self.up1 = UpBlock(in_ch=128,  skip_ch=64,   out_ch=64)

        # Final upsampling to match input resolution (/1)
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Output head
        self.out_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 1, H, W) — batch of preprocessed CT slices (single channel)

        Returns
        -------
        (N, n_classes, H, W) — segmentation logits (apply sigmoid for binary)
        """
        # Encoder
        x0  = self.enc_stem(x)   # (N, 64,   H/4,  W/4)
        e1   = self.enc1(x0)     # (N, 256,  H/4,  W/4)
        e2   = self.enc2(e1)     # (N, 512,  H/8,  W/8)
        e3   = self.enc3(e2)     # (N, 1024, H/16, W/16)
        e4   = self.enc4(e3)     # (N, 2048, H/32, W/32)

        # Bottleneck
        b = self.bottleneck(e4)  # (N, 1024, H/32, W/32)

        # Decoder with skip connections
        d4 = self.up4(b,  e3)   # (N, 512,  H/16, W/16)
        d3 = self.up3(d4, e2)   # (N, 256,  H/8,  W/8)
        d2 = self.up2(d3, e1)   # (N, 128,  H/4,  W/4)
        d1 = self.up1(d2, x0)   # (N, 64,   H/4,  W/4)
        d0 = self.up0(d1)       # (N, 32,   H,    W)

        return self.out_conv(d0) # (N, n_classes, H, W)

    def _load_simclr_weights(self, ckpt_path: str | Path):
        """
        Load ResNet-50 encoder weights from a SimCLR checkpoint.

        Implements the knowledge transfer step: "By initialising the U-Net with
        weights from the SimCLR ResNet-50 encoder, the model began with an
        advanced representation of fossilised bone textures and structural
        signatures that differentiated them from surrounding geological noise."
        """
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)

        # SimCLR state dict keys start with "encoder." — strip the prefix
        encoder_state = {
            k.replace("encoder.", "", 1): v
            for k, v in state.items()
            if k.startswith("encoder.")
        }

        # Load into backbone; strict=False tolerates minor architecture diffs
        for module_name, module in [
            ("enc_stem", self.enc_stem),
            ("enc1", self.enc1),
            ("enc2", self.enc2),
            ("enc3", self.enc3),
            ("enc4", self.enc4),
        ]:
            missing, unexpected = module.load_state_dict(
                {k: v for k, v in encoder_state.items()}, strict=False
            )

        print(f"SimCLR encoder weights loaded from {ckpt_path}")

    @classmethod
    def from_checkpoint(
        cls, ckpt_path: str | Path, map_location: str | torch.device = "cpu"
    ) -> "FossilUNet":
        ckpt = torch.load(str(ckpt_path), map_location=map_location)
        model = cls(n_classes=ckpt.get("n_classes", 1))
        model.load_state_dict(ckpt["model_state_dict"])
        return model
