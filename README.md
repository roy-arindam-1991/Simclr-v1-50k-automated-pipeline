# DEEPCTSEG: Simclr-v1-automated-pipeline (50k)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/HPC-BlueBear%20%7C%20A100-7B2FBE.svg)](https://www.birmingham.ac.uk/research/arc/bear)
[![Pipeline](https://img.shields.io/badge/Pipeline-SimCLR%20→%20U--Net-00B4D8.svg)]()
[![Build](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()

---

## Pipeline Developers

Arindam Roy, [Poulami Ghosh](https://github.com/g-Poulami)

---

## Overview

A fully automated, annotation-free pipeline for semantic segmentation of fossil CT data combining:

1. **SimCLR v1 contrastive pre-training** on unlabelled CT slices to learn fossil-matrix feature representations without any manual annotation
2. **Rule-based deterministic masking** to generate reproducible coarse spatial priors
3. **Knowledge fusion into a modified U-Net** to refine coarse masks into high-precision segmentation overlays
4. **3D mesh generation** via marching cubes for downstream morphometric analysis

The pipeline reduces per-specimen processing from ~100 person-hours to **1–3 minutes** at inference.

---

## Repository Structure

```bash
Simclr-v1-automated-pipeline/
│
├── config/
│   └── config.yaml                  # All hyperparameters and paths
│
├── data/
│   ├── __init__.py
│   ├── preprocessing.py             # Resize, normalise, z-standardise CT slices
│   ├── resize_and_normalise.py      # Resize to 224×224, normalise [0,1], z-standardise
│   ├── tiff_to_hdf5.py              # Convert TIFF stacks → HDF5 (.h5) volumes
│   ├── dataset_split.py             # Fixed-seed split-  train : val : U-Net train (eg. 80:10:10)
│   ├── hdf5_converter.py            # HDF5 I/O utilities
│   └── run_preprocessing.sh         # Runs all data scripts in order on BlueBear HPC
│
├── simclr/
│   ├── __init__.py
│   ├── model.py                     # ResNet-50 encoder + 2-layer MLP projection head
│   ├── augmentations.py             # Domain-specific stochastic augmentation pipeline
│   ├── loss.py                      # NT-Xent (normalised temperature-scaled cross-entropy)
│   └── train.py                     # Training loop: 250 epochs, batch 64, LR 3e-4
│
├── masking/
│   ├── __init__.py
│   └── rule_based_masking.py        # Otsu → region grow → morph ops → component filter
│
├── unet/
│   ├── __init__.py
│   ├── model.py                     # Deep encoder e1–e4 (64–2048 ch), 7×7 bottleneck, skips
│   ├── loss.py                      # Composite Weighted Cross-Entropy + Dice loss
│   └── train.py                     # 500 epochs, AdamW, OneCycleLR, best-checkpoint saving
│
├── mesh/
│   ├── __init__.py
│   └── generate_mesh.py             # Stack masks → 3D vol → marching cubes (iso=0.5) → STL
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                   # Dice-Sørensen coefficient, IoU (Jaccard Index)
│   ├── grad_cam.py                  # Grad-CAM heatmaps over CT slices
│   └── umap_viz.py                  # 2D UMAP of 2048-d features + KDE overlay
│
├── scripts/
│   ├── 01_preprocess.sh             # Stage 1: data preprocessing on SLURM
│   ├── 02_train_simclr.sh           # Stage 2: SimCLR pre-training (A100 GPU)
│   ├── 03_generate_masks.sh         # Stage 3: rule-based coarse masking
│   ├── 04_train_unet.sh             # Stage 4: U-Net training via knowledge fusion
│   └── 05_run_inference.sh          # Stage 5: end-to-end inference loop
│
├── inference.py                     # End-to-end inference on new CT specimens
├── requirements.txt
└── README.md
```

---

## Pipeline Workflow

```bash
 Raw CT TIFF Slices
        │
        ▼
┌───────────────────────────────────────────┐
│  STAGE 1 · Preprocessing                  │
│  resize → 224×224                         │
│  normalise [0,1] · z-standardise          │
│  TIFF stacks → HDF5 volumes               │
└───────────────┬───────────────────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
┌──────────────┐  ┌──────────────────────────┐
│  STAGE 2     │  │  STAGE 3                 │
│  SimCLR v1   │  │  Rule-Based Masking      │
│  Pre-training│  │                          │
│              │  │  Otsu thresholding       │
│  ResNet-50   │  │  → region growing        │
│  + MLP head  │  │  → morphological ops     │
│  NT-Xent loss│  │  → component filtering   │
│  250 epochs  │  │                          │
│  39,037 slices  │  Coarse binary masks     │
│  2048-d feats│  │                          │
└──────┬───────┘  └─────────────┬────────────┘
       |                        │
       └───────────|────────────┘
                   ▼
┌─────────────────────────────────────────┐
│  STAGE 4 · Knowledge Fusion → U-Net     │
│                                         │
│  SimCLR encoder weights → U-Net init    │
│  Deep encoder: e1–e4 (64 → 2048 ch)     │
│  7×7 bottleneck · skip connections      │
│  Loss: Weighted Cross-Entropy + Dice    │
│  500 epochs · AdamW · OneCycleLR        │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  STAGE 5 · 3D Mesh Generation           │
│                                         │
│  2D mask stack → 3D volume              │
│  Region growing · Otsu intensity filter │
│  Marching cubes (iso = 0.5)             │
│  Watertight STL · 1–3 min per specimen  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  STAGE 6 · Evaluation                   │
│                                         │
│  Dice / IoU per specimen                │
│  Grad-CAM encoder sanity checks         │
│  UMAP 2048-d feature visualisation      │
│  CloudCompare PPR → ICP → C2C distance  │
└─────────────────────────────────────────┘
```

---

## Installation

```bash
git clone https://github.com/roy-arindam-1991/Simclr-v1-automated-pipeline.git
cd Simclr-v1-automated-pipeline
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Stage 1 — preprocess raw TIFF stacks
bash scripts/01_preprocess.sh

# Stage 2 — pre-train SimCLR on unlabelled slices
bash scripts/02_train_simclr.sh

# Stage 3 — generate rule-based coarse masks
bash scripts/03_generate_masks.sh

# Stage 4 — train U-Net via knowledge fusion
bash scripts/04_train_unet.sh

# Stage 5 — run inference on new specimens
bash scripts/05_run_inference.sh
```

Or end-to-end via Python:

```python
from inference import FossilSegmentationPipeline

pipeline = FossilSegmentationPipeline.from_config("config/config.yaml")
pipeline.run(ct_dir="path/to/new/specimen/tiffs/", output_dir="outputs/")
```

---

## Data

Training data: 50,626 CT images from the Middle Jurassic Kilmaluag Formation, Skye, Scotland. Specimens housed at **National Museums Scotland (NMS)**. Data available on request.

---

## Hardware

Trained on a single **NVIDIA A100 GPU** via BlueBear HPC (University of Birmingham).

| Stage | Time |
|---|---|
| SimCLR pre-training (250 epochs) | ~37–38 hrs |
| U-Net training (500 epochs) | ~6 hrs 10 min |
| Peak GPU memory | ~3.26 GB |
| Mesh generation per specimen | 1–3 min |

---

## License

This project is licensed under the **GNU General Public License v3.0** — see [LICENSE](LICENSE) for full terms.  
All derivative works must remain open-source under GPL-3.0.
