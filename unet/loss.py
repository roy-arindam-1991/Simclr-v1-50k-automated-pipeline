"""
unet/loss.py
============
Composite loss function for U-Net training.

"Training was conducted over 500 epochs using a composite Weighted
Cross-Entropy (WCE) and Dice loss function, designed to address the
pronounced class imbalance between bone and matrix voxels."
— manuscript §Knowledge fusion

Class weights [background=0.1, bone=0.9] reflect the severe imbalance
in fossil CT data (bone is a small fraction of each slice's area).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    Dice = 2 |P ∩ T| / (|P| + |T|)
    Loss = 1 − Dice

    Smooth factor prevents division by zero on empty masks (common in fossil CT
    slices where thin sections contain no bone).
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (N, 1, H, W) — raw model output (before sigmoid)
        targets : (N, 1, H, W) — binary ground truth {0, 1}
        """
        probs = torch.sigmoid(logits)
        probs_flat  = probs.view(probs.shape[0], -1)
        target_flat = targets.view(targets.shape[0], -1).float()

        intersection = (probs_flat * target_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class WeightedCrossEntropyLoss(nn.Module):
    """
    Pixel-wise binary cross-entropy with class weights.

    Weights [background, bone] = [0.1, 0.9] address the severe class imbalance
    between bone voxels and the surrounding matrix in fossil CT slices.
    """

    def __init__(self, class_weights: list[float] | None = None):
        super().__init__()
        weights = class_weights if class_weights is not None else [0.1, 0.9]
        # pos_weight = bone_weight / background_weight for BCEWithLogitsLoss
        self.pos_weight_ratio = weights[1] / weights[0]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = torch.tensor(
            [self.pos_weight_ratio], device=logits.device, dtype=logits.dtype
        )
        return F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=pos_weight
        )


class CompositeLoss(nn.Module):
    """
    Composite Weighted Cross-Entropy + Dice loss.

    L = wce_weight × L_WCE + dice_weight × L_Dice

    Default weighting: 0.5 / 0.5 (equal contribution, from config.yaml).
    """

    def __init__(
        self,
        wce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: list[float] | None = None,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.wce_weight  = wce_weight
        self.dice_weight = dice_weight
        self.wce  = WeightedCrossEntropyLoss(class_weights)
        self.dice = DiceLoss(smooth=smooth)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Parameters
        ----------
        logits  : (N, 1, H, W)
        targets : (N, 1, H, W)

        Returns
        -------
        total_loss : scalar tensor
        components : dict with individual loss values for logging
        """
        l_wce  = self.wce(logits, targets)
        l_dice = self.dice(logits, targets)
        total  = self.wce_weight * l_wce + self.dice_weight * l_dice
        return total, {"wce": l_wce.item(), "dice": l_dice.item()}
