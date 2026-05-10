"""
evaluation/metrics.py
=====================
Segmentation quality metrics reported in the manuscript.

Metrics
-------
- Dice-Sørensen Coefficient (DSC) — primary metric; reported as 0.9366 on
  held-out internal test specimen and 0.838–0.843 on five of six external
  specimens (Table 1).
- Intersection-over-Union (IoU / Jaccard Index) — companion metric; reported
  as 0.8242 on held-out test specimen and 0.723–0.730 on five external
  specimens.

Both metrics are computed on the underlying 2D segmentation masks.
Geometric mesh-level agreement is assessed separately via CloudCompare
(see manuscript §AI-generated meshes achieve sub-voxel agreement).
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Torch versions (used during training for GPU-accelerated batch evaluation)
# ---------------------------------------------------------------------------

def dice_coefficient(
    preds: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> float:
    """
    Compute mean Dice-Sørensen coefficient over a batch.

    Dice = 2|P ∩ T| / (|P| + |T|)

    Parameters
    ----------
    preds   : (N, 1, H, W) float tensor — binary predictions {0.0, 1.0}
    targets : (N, 1, H, W) float tensor — binary ground truth {0.0, 1.0}
    smooth  : Laplace smoothing to avoid division by zero

    Returns
    -------
    Mean Dice score as a Python float
    """
    preds_flat  = preds.view(preds.shape[0], -1)
    target_flat = targets.view(targets.shape[0], -1).float()
    intersection = (preds_flat * target_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        preds_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth
    )
    return dice.mean().item()


def iou_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> float:
    """
    Compute mean Intersection-over-Union (Jaccard Index) over a batch.

    IoU = |P ∩ T| / |P ∪ T|

    Parameters
    ----------
    preds   : (N, 1, H, W) float tensor — binary predictions {0.0, 1.0}
    targets : (N, 1, H, W) float tensor — binary ground truth {0.0, 1.0}
    smooth  : Laplace smoothing

    Returns
    -------
    Mean IoU score as a Python float
    """
    preds_flat  = preds.view(preds.shape[0], -1)
    target_flat = targets.view(targets.shape[0], -1).float()
    intersection = (preds_flat * target_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


# ---------------------------------------------------------------------------
# NumPy versions (used for offline evaluation on full volumes)
# ---------------------------------------------------------------------------

def dice_numpy(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """Binary Dice coefficient on numpy arrays."""
    pred_flat   = pred.flatten().astype(float)
    target_flat = target.flatten().astype(float)
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def iou_numpy(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """Binary IoU on numpy arrays."""
    pred_flat   = pred.flatten().astype(float)
    target_flat = target.flatten().astype(float)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


# ---------------------------------------------------------------------------
# Per-slice statistics (matches manuscript Table 1 reporting style)
# ---------------------------------------------------------------------------

def volume_metrics(
    pred_volume: np.ndarray,
    target_volume: np.ndarray,
) -> dict[str, float]:
    """
    Compute per-slice Dice and IoU then aggregate for a full CT volume.

    Parameters
    ----------
    pred_volume   : (Z, H, W) binary uint8
    target_volume : (Z, H, W) binary uint8

    Returns
    -------
    dict with keys: mean_dice, std_dice, mean_iou, std_iou, n_slices
    """
    assert pred_volume.shape == target_volume.shape, "Shape mismatch"
    dices, ious = [], []
    for z in range(pred_volume.shape[0]):
        p = pred_volume[z]
        t = target_volume[z]
        # Skip empty slices (no bone in either prediction or target)
        if p.sum() == 0 and t.sum() == 0:
            continue
        dices.append(dice_numpy(p, t))
        ious.append(iou_numpy(p, t))

    return {
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "std_dice":  float(np.std(dices))  if dices else 0.0,
        "mean_iou":  float(np.mean(ious))  if ious  else 0.0,
        "std_iou":   float(np.std(ious))   if ious  else 0.0,
        "n_slices":  len(dices),
    }
