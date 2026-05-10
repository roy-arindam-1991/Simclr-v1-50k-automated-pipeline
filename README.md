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

1. **SimCLR v1 contrastive pretraining** on unlabelled CT slices to learn fossil-matrix feature representations without any manual annotation
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
│   ├── README.md                    # Configuration guide and key parameter reference
│   └── config.yaml                  # All hyperparameters and paths
│
├── data/
│   ├── README.md                    # Data access, MorphoSource accession, NMS request
│   ├── __init__.py
│   ├── preprocessing.py             # Resize, normalise, z-standardise CT slices
│   ├── resize_and_normalise.py      # Resize and normalise pipeline
│   ├── tiff_to_hdf5.py              # Convert TIFF stacks → HDF5 (.h5) volumes
│   ├── dataset_split.py             # Fixed-seed train / val / U-Net partition split
│   ├── hdf5_converter.py            # HDF5 I/O utilities
│   └── run_preprocessing.sh         # Runs all data scripts in order on BlueBear HPC
│
├── simclr/
│   ├── README.md                    # SimCLR architecture, augmentation rationale, training
│   ├── __init__.py
│   ├── model.py                     # ResNet-50 encoder + 2-layer MLP projection head
│   ├── augmentations.py             # Domain-specific stochastic augmentation pipeline
│   ├── loss.py                      # NT-Xent (normalised temperature-scaled cross-entropy)
│   └── train.py                     # SimCLR pre-training loop
│
├── simclr_validation/
│   ├── README.md                    # How to run Grad-CAM and UMAP sanity checks
│   ├── __init__.py
│   ├── metrics.py                   # Dice-Sørensen coefficient, IoU (Jaccard Index)
│   ├── grad_cam.py                  # Grad-CAM heatmaps over CT slices (Fig. 2d)
│   └── umap_viz.py                  # 2D UMAP of learned features + KDE overlay (Fig. 2c)
│
├── masking/
│   ├── README.md                    # 5-stage deterministic pipeline, fixed params, usage
│   ├── __init__.py
│   └── rule_based_masking.py        # Otsu → region grow → morph ops → component filter
│
├── knowledge_fusion/
│   ├── README.md                    # Dual-stream transfer: SimCLR → U-Net encoder init
│   ├── __init__.py
│   ├── transfer_weights.py          # Loads SimCLR checkpoint, maps weights to U-Net encoder
│   └── run_transfer.sh              # Validates weight transfer before U-Net training starts
│
├── unet/
│   ├── README.md                    # Modified U-Net architecture, loss, training details
│   ├── __init__.py
│   ├── model.py                     # Deep encoder, bottleneck, skip connections
│   ├── loss.py                      # Composite Weighted Cross-Entropy + Dice loss
│   └── train.py                     # U-Net training loop, AdamW, OneCycleLR
│
├── mesh/
│   ├── README.md                    # 3D mesh generation pipeline and output metrics
│   ├── __init__.py
│   ├── generate_mesh.py             # Stack masks → 3D vol → marching cubes → STL
│   └── mesh_construction.py         # End-to-end mesh construction entry point
│
├── registration/
│   ├── README.md                    # CloudCompare PPR → ICP → C2C workflow
│   ├── __init__.py
│   ├── registration_stats.py        # Parses CloudCompare CSV export → Table 2 summary
│   └── run_registration.sh          # Batch-calls registration_stats.py for all specimens
│
├── scripts/
│   ├── README.md                    # SLURM job scripts, stage order, HPC details
│   ├── 01_preprocess.sh             # Stage 1: data preprocessing on SLURM
│   ├── 02_train_simclr.sh           # Stage 2: SimCLR pre-training
│   ├── 03_simclr_validation.sh      # Stage 3: Grad-CAM and UMAP sanity checks
│   ├── 04_generate_masks.sh         # Stage 4: rule-based coarse masking
│   ├── 05_transfer_weights.sh       # Stage 5: SimCLR → U-Net knowledge fusion
│   ├── 06_train_unet.sh             # Stage 6: U-Net training
│   └── 07_run_inference.sh          # Stage 7: inference and mesh generation
│
├── requirements.txt
└── README.md
```

---

## Pipeline Workflow

```bash
 Raw CT TIFF Slices
        │
        ▼
┌──────────────────────────────────────────┐
│  STAGE 1 · Data Preprocessing            │
│                                          │
│  Proportional sampling                   │
│  resize → 224×224 px                     │
│  16-bit → 8-bit conversion               │
│  normalise · z-standardise               │
│  TIFF stacks → HDF5 volumes              │
└───────────────┬──────────────────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
┌──────────────────┐  ┌──────────────────────────┐
│  STAGE 2         │  │  STAGE 4                 │
│  SimCLR v1       │  │  Rule-Based Masking      │
│  Pre-training    │  │                          │
│                  │  │  Intensity normalisation │
│  ResNet-50       │  │  → Gaussian/median blur  │
│  + MLP head      │  │  → Otsu thresholding     │
│  NT-Xent loss    │  │  → Region growing        │
│                  │  │  → Morphological ops     │
│                  │  │  → Component filtering   │
└──────┬───────────┘  │                          │
       │              │  Coarse binary masks     │
       ▼              │  Fixed parameters —      │
┌──────────────────┐  │  fully reproducible      │
│  STAGE 3         │  └─────────────┬────────────┘
│  SimCLR          │                │
│  Validation      │                │
│                  │                │
│  · Grad-CAM      │                │
│  · UMAP + KDE    │                │
└──────┬───────────┘                │
       │                            │
       └──────────────┬─────────────┘
                      ▼
┌─────────────────────────────────────────┐
│  STAGE 5 · Knowledge Fusion             │
│                                         │
│  SimCLR ResNet-50 encoder weights       │
│  → initialise U-Net encoder             │
│                                         │
│  Dual-stream transfer:                  │
│  · Textural features from SimCLR        │
│  · Spatial priors from coarse masks     │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  STAGE 6 · U-Net Segmentation           │
│                                         │
│  Deep encoder with skip connections     │
│  Expanded bottleneck                    │
│  Loss: Weighted Cross-Entropy + Dice    │
│  AdamW · OneCycleLR                     │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────────┐
│  STAGE 7 · 3D Mesh Generation                 │
│                                               │
│  Stack 2D masks → 3D volume                   │
│  Region growing · Otsu intensity filter       │
│  Morphological cleanup                        │
│  Marching cubes surface extraction            │
│  Laplacian smoothing                          │ 
│  Mesh simplification via quadratic decimation │
│  Watertight STL output                        │
└──────────────────┬────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  STAGE 8 · Geometric Registration       │
│           (CloudCompare v2)             │
│                                         │
│  PPR → manual landmarking               │
│  ICP → rigid alignment (scale = 1.0)    │
│  C2C → signed distance analysis         │
│                                         │
│  External specimens validated           │
│  against manual thresholding baseline   │
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

# Stage 3 — SimCLR sanity checks (Grad-CAM + UMAP)
bash scripts/03_simclr_validation.sh

# Stage 4 — generate rule-based coarse masks
bash scripts/04_generate_masks.sh

# Stage 5 — SimCLR → U-Net knowledge fusion
bash scripts/05_transfer_weights.sh

# Stage 6 — train U-Net
bash scripts/06_train_unet.sh

# Stage 7 — run inference and generate 3D meshes
bash scripts/07_run_inference.sh
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
