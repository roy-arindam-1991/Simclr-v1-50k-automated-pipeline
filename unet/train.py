"""
unet/train.py
=============
Training loop for the modified U-Net segmentation model.

Key hyperparameters (manuscript §Dual-stream knowledge transfer):
  - Epochs        : 500
  - Optimiser     : AdamW (Loshchilov & Hutter, 2017)
  - Scheduler     : OneCycleLR (Smith & Topin, 2019) — rapid identification
                    of optimal learning rate during training
  - Loss          : Composite Weighted Cross-Entropy + Dice
  - Initialisation: SimCLR ResNet-50 encoder weights (knowledge transfer)
  - Training data : 2 specimens with rule-based deterministic masks
  - Test data     : 1 held-out specimen (same genus, different individual)
                    → Dice = 0.9366, IoU = 0.8242

Convergence: "Both training and test losses converged within 50 epochs
and plateaued thereafter without divergence."
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

from evaluation.metrics import dice_coefficient, iou_score
from unet.loss import CompositeLoss
from unet.model import FossilUNet

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    """
    Paired CT-slice + binary-mask dataset.

    Expects:
        slices : (N, H, W) float32 — preprocessed CT volume
        masks  : (N, H, W) uint8   — rule-based coarse masks (0/1)
    """

    def __init__(self, slices, masks):
        assert len(slices) == len(masks), "Slice and mask counts must match"
        self.slices = slices
        self.masks = masks

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int):
        import torch
        x = torch.tensor(self.slices[idx], dtype=torch.float32).unsqueeze(0)  # (1,H,W)
        y = torch.tensor(self.masks[idx],  dtype=torch.float32).unsqueeze(0)  # (1,H,W)
        return x, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_unet(
    train_slices,
    train_masks,
    test_slices,
    test_masks,
    simclr_ckpt: str | Path,
    config_path: str | Path = "config/config.yaml",
) -> FossilUNet:
    """
    Train the modified U-Net via dual-stream knowledge fusion.

    Parameters
    ----------
    train_slices  : (N_train, H, W) — CT slices for training
    train_masks   : (N_train, H, W) — corresponding rule-based masks
    test_slices   : (N_test, H, W)  — held-out specimen slices
    test_masks    : (N_test, H, W)  — held-out specimen reference masks
    simclr_ckpt   : path to best SimCLR checkpoint (provides encoder weights)
    config_path   : path to config.yaml

    Returns
    -------
    Trained FossilUNet model
    """
    cfg  = load_config(config_path)
    ucfg = cfg["unet"]
    paths = cfg["paths"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training U-Net on %s", device)

    # -- Model (knowledge transfer: SimCLR encoder weights) --
    model = FossilUNet(simclr_ckpt=simclr_ckpt, n_classes=1).to(device)

    # -- Loss --
    criterion = CompositeLoss(
        wce_weight=ucfg["loss"]["wce_weight"],
        dice_weight=ucfg["loss"]["dice_weight"],
        class_weights=ucfg["loss"]["class_weights"],
    )

    # -- Optimiser: AdamW --
    optimizer = optim.AdamW(
        model.parameters(),
        lr=ucfg["learning_rate"],
        weight_decay=ucfg["weight_decay"],
    )

    # -- Data --
    train_ds = SegmentationDataset(train_slices, train_masks)
    test_ds  = SegmentationDataset(test_slices,  test_masks)
    train_loader = DataLoader(train_ds, batch_size=ucfg["batch_size"],
                              shuffle=True,  num_workers=ucfg["num_workers"],
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=ucfg["batch_size"],
                              shuffle=False, num_workers=ucfg["num_workers"],
                              pin_memory=True)

    # -- Scheduler: OneCycleLR —
    # "applied to rapidly identify an optimal learning rate during training"
    scheduler = OneCycleLR(
        optimizer,
        max_lr=ucfg["learning_rate"],
        steps_per_epoch=len(train_loader),
        epochs=ucfg["epochs"],
    )

    ckpt_dir = Path(paths["unet_ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_dice = 0.0
    patience_counter = 0
    history = {"train_loss": [], "test_loss": [], "test_dice": [], "test_iou": []}

    for epoch in range(1, ucfg["epochs"] + 1):
        t0 = time.time()

        # ---- Train ----
        train_loss = _train_one_epoch(model, train_loader, criterion,
                                      optimizer, scheduler, device)

        # ---- Evaluate on held-out test specimen ----
        test_loss, test_dice, test_iou = _evaluate(
            model, test_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_dice"].append(test_dice)
        history["test_iou"].append(test_iou)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d | train_loss=%.4f test_loss=%.4f "
            "dice=%.4f iou=%.4f | %.1fs",
            epoch, ucfg["epochs"],
            train_loss, test_loss, test_dice, test_iou, elapsed,
        )

        # Best checkpoint
        if test_dice > best_dice:
            best_dice = test_dice
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_dice": test_dice,
                "test_iou": test_iou,
                "n_classes": 1,
            }, ckpt_dir / "unet_best.pt")
            logger.info("  ↑ New best dice=%.4f — checkpoint saved", test_dice)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= ucfg["early_stopping_patience"]:
            logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch, ucfg["early_stopping_patience"])
            break

    logger.info("U-Net training complete. Best test Dice=%.4f", best_dice)
    return model


def _train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss, _ = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_dice = total_iou = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss, _ = criterion(logits, y)
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_loss += loss.item()
            total_dice += dice_coefficient(preds, y)
            total_iou  += iou_score(preds, y)
    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    # Example usage — replace with real data loaders
    # train_unet(train_slices, train_masks, test_slices, test_masks,
    #            simclr_ckpt="checkpoints/simclr/simclr_best.pt")
