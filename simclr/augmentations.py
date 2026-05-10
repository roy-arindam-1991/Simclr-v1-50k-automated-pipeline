"""
simclr/augmentations.py
=======================
Domain-specific stochastic augmentation pipeline for SimCLR v1 pre-training.

Each CT slice is transformed twice to produce a positive pair (xi, xj).
Augmentations are designed to match the structural characteristics of
palaeontological µCT data (manuscript §SimCLR pre-training):

  - Geometric transforms (crop, flip, rotation) → invariance to arbitrary
    specimen orientation within the matrix.
  - Intensity / blur augmentations → invariance to scanner- and
    matrix-chemistry-dependent variation in CT attenuation values across
    multi-institutional fossil datasets.

All probabilities and ranges are taken from config/config.yaml.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import yaml


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class FossilCTAugmentation:
    """
    Generate a positive pair of augmented views from a single CT slice.

    Parameters are read from the *simclr.augmentations* section of config.yaml.

    Usage
    -----
    augment = FossilCTAugmentation("config/config.yaml")
    xi, xj = augment(slice_tensor)   # both are float tensors of shape (1, H, W)
    """

    def __init__(self, config_path: str | Path = "config/config.yaml"):
        cfg = load_config(config_path)["simclr"]["augmentations"]
        size = load_config(config_path)["preprocessing"]["image_size"]

        self.transform = T.Compose([
            # --- geometric transforms ---
            T.RandomResizedCrop(
                size=size,
                scale=cfg["random_resized_crop"]["scale"],
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.RandomVerticalFlip(p=cfg["vertical_flip"]["p"]),
            T.RandomHorizontalFlip(p=cfg["horizontal_flip"]["p"]),
            _RandomRotation(
                degrees=cfg["random_rotation"]["degrees"],
                p=cfg["random_rotation"]["p"],
            ),
            # --- intensity transforms ---
            _RandomGaussianBlur(p=cfg["gaussian_blur"]["p"]),
            _RandomIntensityJitter(
                brightness=cfg["color_jitter"]["brightness"],
                contrast=cfg["color_jitter"]["contrast"],
                p_brightness=cfg["color_jitter"]["p_brightness"],
                p_contrast=cfg["color_jitter"]["p_contrast"],
            ),
        ])

    def __call__(
        self, img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply two independent random augmentations to the same input.

        Parameters
        ----------
        img : float tensor, shape (1, H, W) or (H, W)

        Returns
        -------
        (xi, xj) : pair of float tensors, shape (1, H, W)
        """
        if img.ndim == 2:
            img = img.unsqueeze(0)
        return self.transform(img), self.transform(img)


# ---------------------------------------------------------------------------
# Custom transforms for greyscale CT data
# ---------------------------------------------------------------------------

class _RandomRotation:
    """Random rotation applied with probability p."""

    def __init__(self, degrees: float, p: float = 0.3):
        self.degrees = degrees
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees))
            return TF.rotate(img, angle)
        return img


class _RandomGaussianBlur:
    """
    Gaussian blur with a randomly sampled σ ∈ [0.1, 2.0].

    Enforces invariance to high-frequency noise and radiographic artefacts
    (beam hardening, salt-and-pepper noise) typical of µCT data.
    """

    def __init__(self, p: float = 0.3, sigma_range: tuple[float, float] = (0.1, 2.0)):
        self.p = p
        self.sigma_range = sigma_range

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            sigma = float(torch.empty(1).uniform_(*self.sigma_range))
            kernel_size = int(2 * round(3 * sigma) + 1)   # 3σ rule, must be odd
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            return TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size],
                                    sigma=[sigma, sigma])
        return img


class _RandomIntensityJitter:
    """
    Intensity (brightness / contrast) jitter for greyscale CT slices.

    Enforces invariance to the scanner- and matrix-chemistry-dependent
    variation in CT attenuation values across multi-institutional datasets.
    """

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        p_brightness: float = 0.8,
        p_contrast: float = 0.8,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.p_brightness = p_brightness
        self.p_contrast = p_contrast

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p_brightness:
            factor = float(torch.empty(1).uniform_(
                max(0, 1 - self.brightness), 1 + self.brightness
            ))
            img = TF.adjust_brightness(img, factor)

        if torch.rand(1).item() < self.p_contrast:
            factor = float(torch.empty(1).uniform_(
                max(0, 1 - self.contrast), 1 + self.contrast
            ))
            img = TF.adjust_contrast(img, factor)

        return img
