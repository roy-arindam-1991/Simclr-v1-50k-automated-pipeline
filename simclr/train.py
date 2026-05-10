"""
simclr/train.py
===============
Training loop for SimCLR v1 contrastive pre-training.

Hyperparameters (from manuscript and config/config.yaml):
  - Encoder  : ResNet-50
  - Epochs   : 250
  - Batch    : 64
  - LR       : 3 × 10⁻⁴ (Adam)
  - Weight decay : 1 × 10⁻⁴
  - Loss     : NT-Xent (τ = 0.07)
  - Hardware : single NVIDIA A100 GPU (~37–38 hrs total)

Training corpus: 39,037 unlabelled CT slices from 19 specimens
Validation set : 3,765 slices from 2 specimens (for checkpoint selection)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

from simclr.augmentations import FossilCTAugmentation
from simclr.loss import NTXentLoss, contrastive_accuracy
from simclr.model import SimCLRv1

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SimCLRDataset(Dataset):
    """
    Wraps a FossilHDF5Dataset to return positive pairs (xi, xj) via the
    FossilCTAugmentation pipeline.
    """

    def __init__(self, hdf5_dataset, augmentation: FossilCTAugmentation):
        self.dataset = hdf5_dataset
        self.augment = augmentation

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        import numpy as np
        import torch
        slice_np = self.dataset[idx]                             # (224, 224)
        img = torch.from_numpy(slice_np).unsqueeze(0).float()   # (1, 224, 224)
        xi, xj = self.augment(img)
        return xi, xj


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_simclr(config_path: str | Path = "config/config.yaml") -> SimCLRv1:
    """
    Run the SimCLR v1 pre-training loop.

    Checkpoints are saved every 10 epochs and on best validation accuracy
    to paths.simclr_ckpt_dir (config.yaml).

    Returns the trained SimCLRv1 model.
    """
    cfg = load_config(config_path)
    scfg = cfg["simclr"]
    paths = cfg["paths"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training SimCLR on %s", device)

    # -- Model --
    model = SimCLRv1(
        feature_dim=scfg["feature_dim"],
        hidden_dim=scfg["projection_hidden_dim"],
        projection_dim=scfg["projection_dim"],
        pretrained=False,
    ).to(device)

    # -- Loss --
    criterion = NTXentLoss(temperature=scfg["temperature"])

    # -- Optimiser --
    optimizer = optim.Adam(
        model.parameters(),
        lr=scfg["learning_rate"],
        weight_decay=scfg["weight_decay"],
    )

    # -- Data --
    # NOTE: Replace the placeholder imports below with your actual HDF5 paths.
    # Example:
    #   from data.hdf5_converter import FossilHDF5Dataset
    #   train_hdf5 = FossilHDF5Dataset([...])
    #   val_hdf5   = FossilHDF5Dataset([...])
    augmentation = FossilCTAugmentation(config_path)
    # train_dataset = SimCLRDataset(train_hdf5, augmentation)
    # val_dataset   = SimCLRDataset(val_hdf5,   augmentation)
    # train_loader  = DataLoader(train_dataset, batch_size=scfg["batch_size"],
    #                            shuffle=True,  num_workers=scfg["num_workers"],
    #                            pin_memory=True, drop_last=True)
    # val_loader    = DataLoader(val_dataset,   batch_size=scfg["batch_size"],
    #                            shuffle=False, num_workers=scfg["num_workers"],
    #                            pin_memory=True, drop_last=True)

    ckpt_dir = Path(paths["simclr_ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, scfg["epochs"] + 1):
        t0 = time.time()

        # ---- Training ----
        model.train()
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = _run_epoch(
                model, val_loader, criterion, None, device, training=False
            )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d | train_loss=%.4f val_loss=%.4f "
            "train_acc=%.4f val_acc=%.4f | %.1fs",
            epoch, scfg["epochs"],
            train_loss, val_loss, train_acc, val_acc, elapsed,
        )

        # Periodic checkpoint
        if epoch % 10 == 0:
            _save_checkpoint(model, optimizer, epoch, val_acc,
                             ckpt_dir / f"simclr_epoch{epoch:03d}.pt", scfg)

        # Best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _save_checkpoint(model, optimizer, epoch, val_acc,
                             ckpt_dir / "simclr_best.pt", scfg)
            logger.info("  ↑ New best val_acc=%.4f — checkpoint saved", val_acc)

    logger.info("SimCLR pre-training complete. Best val_acc=%.4f", best_val_acc)
    return model


def _run_epoch(
    model: SimCLRv1,
    loader: DataLoader,
    criterion: NTXentLoss,
    optimizer,
    device: torch.device,
    training: bool,
) -> tuple[float, float]:
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for xi, xj in loader:
        xi = xi.to(device)
        xj = xj.to(device)

        _, zi = model(xi)
        _, zj = model(xj)

        loss = criterion(zi, zj)
        acc = contrastive_accuracy(zi.detach(), zj.detach(),
                                   temperature=criterion.temperature)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def _save_checkpoint(model, optimizer, epoch, val_acc, path, hparams):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "hparams": hparams,
    }, path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    train_simclr()
