"""
evaluation/umap_viz.py
======================
UMAP dimensionality reduction for SimCLR feature space visualisation.

"UMAP dimensionality reduction was applied to the 2048-dimensional feature
vectors extracted by the trained ResNet-50 encoder, projecting them into
two-dimensional space. The projection is overlaid with a Gaussian Kernel
Density Estimate (KDE), mapping point density from dark purple (sparse)
through teal and green (moderate) to yellow (highest-density cores).
Crucially, no class labels are used; all visible structure has emerged
entirely from the contrastive objective."
— manuscript §Contrastive pre-training encodes fossil-matrix structure

Replicates the sanity-check visualisation in manuscript Fig. 2c.

Reference
---------
McInnes, L., Healy, J., & Melville, J. (2018).
UMAP: Uniform manifold approximation and projection for dimension reduction.
arXiv:1802.03426.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import gaussian_kde


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_features(
    model,
    slices: np.ndarray,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Extract 2048-d feature vectors from the SimCLR ResNet-50 encoder.

    Parameters
    ----------
    model      : trained SimCLRv1 model
    slices     : (N, H, W) float array — preprocessed CT slices
    batch_size : inference batch size

    Returns
    -------
    features : (N, 2048) float32 array
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    all_features = []

    with torch.no_grad():
        for i in range(0, len(slices), batch_size):
            batch = slices[i: i + batch_size]
            t = torch.tensor(batch, dtype=torch.float32).unsqueeze(1).to(device)
            h, _ = model(t)   # h: (B, 2048)
            all_features.append(h.cpu().numpy())

    return np.vstack(all_features)


def compute_umap(
    features: np.ndarray,
    config_path: str | Path = "config/config.yaml",
    random_state: int = 42,
) -> np.ndarray:
    """
    Project feature vectors into 2D using UMAP.

    Parameters
    ----------
    features    : (N, D) float array — encoder feature vectors
    config_path : path to config.yaml (reads umap hyperparameters)

    Returns
    -------
    embedding : (N, 2) float array — 2D UMAP coordinates
    """
    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn is required: pip install umap-learn"
        )

    cfg = load_config(config_path)["evaluation"]
    reducer = umap.UMAP(
        n_neighbors=cfg["umap_n_neighbours"],
        min_dist=cfg["umap_min_dist"],
        n_components=cfg["umap_n_components"],
        metric=cfg["umap_metric"],
        random_state=random_state,
        verbose=True,
    )
    embedding = reducer.fit_transform(features)
    return embedding


def plot_umap_kde(
    embedding: np.ndarray,
    output_path: Optional[str | Path] = None,
    title: str = "UMAP of SimCLR Feature Space (Validation Set)",
) -> plt.Figure:
    """
    Render UMAP embedding with Gaussian KDE overlay.

    Colour scale: dark purple (sparse) → teal/green (moderate) → yellow
    (highest-density cores), matching manuscript Fig. 2c.

    Parameters
    ----------
    embedding   : (N, 2) UMAP coordinates
    output_path : save path for the figure (optional)
    title       : figure title

    Returns
    -------
    matplotlib Figure
    """
    x, y = embedding[:, 0], embedding[:, 1]

    # Gaussian KDE over the 2D embedding
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    density = kde(xy)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(x, y, c=density, cmap="plasma", s=4, alpha=0.7, linewidths=0)
    plt.colorbar(sc, ax=ax, label="KDE density")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.text(0.02, 0.98,
            "No class labels used — structure\nemerges from contrastive objective",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def run_umap_pipeline(
    model,
    val_slices: np.ndarray,
    output_dir: str | Path = "outputs/",
    config_path: str | Path = "config/config.yaml",
) -> np.ndarray:
    """
    End-to-end UMAP sanity check: extract features → UMAP → plot.

    Parameters
    ----------
    model      : trained SimCLRv1 model
    val_slices : (N, H, W) validation CT slices
    output_dir : directory to save figure and embedding

    Returns
    -------
    embedding : (N, 2) UMAP coordinates
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting features …")
    features = extract_features(model, val_slices)
    np.save(output_dir / "simclr_features.npy", features)

    print("Computing UMAP …")
    embedding = compute_umap(features, config_path=config_path)
    np.save(output_dir / "umap_embedding.npy", embedding)

    print("Plotting …")
    plot_umap_kde(embedding, output_path=output_dir / "umap_kde.png")

    print(f"UMAP pipeline complete. Outputs in {output_dir}")
    return embedding
