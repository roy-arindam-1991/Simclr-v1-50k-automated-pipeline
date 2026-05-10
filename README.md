# Breaking the Bottleneck: Self-Supervised Deep Learning for Fully Automated Fossil CT Segmentation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of the paper:

> **Breaking the bottleneck: self-supervised deep learning for fully automated fossil CT segmentation**  
> Arindam Roy, Poulami Ghosh, Roger Benson, Fraser Weston, Ben Scott, Arianna Salili-James, Sanson T.S. Poon, Stig Walsh, Susannah Maidment, Richard Butler  
> University of Birmingham · Natural History Museum · American Museum of Natural History · National Museums Scotland

---

## Overview

This repository provides a fully automated, annotation-free pipeline for semantic segmentation of fossil CT data. It combines:

1. **SimCLR v1 contrastive pre-training** on unlabelled CT slices to learn fossil-matrix feature representations without any manual annotation
2. **Rule-based deterministic masking** to generate reproducible coarse spatial priors
3. **Knowledge fusion into a modified U-Net** to refine coarse masks into high-precision segmentation overlays
4. **3D mesh generation** via marching cubes for downstream morphometric analysis

The pipeline reduces per-specimen processing from ~100 person-hours to **1–3 minutes** at inference.

---

## Repository Structure

```
Simclr-v1-automated-pipeline/
├── config/
│   └── config.yaml            # All hyperparameters and paths
├── data/
│   ├── preprocessing.py       # Resize, normalise, z-standardise CT slices
│   └── hdf5_converter.py      # Convert TIFF stacks → HDF5 for fast I/O
├── simclr/
│   ├── model.py               # ResNet-50 encoder + MLP projection head
│   ├── augmentations.py       # Domain-specific stochastic augmentation pipeline
│   ├── loss.py                # NT-Xent (Normalised Temperature-scaled Cross-Entropy) loss
│   └── train.py               # SimCLR pre-training loop (250 epochs)
├── masking/
│   └── rule_based_masking.py  # 5-stage deterministic bone-mask pipeline
├── unet/
│   ├── model.py               # Modified U-Net (ResNet-50 encoder, 7×7 bottleneck)
│   ├── loss.py                # Composite Weighted Cross-Entropy + Dice loss
│   └── train.py               # U-Net training loop (500 epochs)
├── mesh/
│   └── generate_mesh.py       # Region growing → marching cubes → watertight STL
├── evaluation/
│   ├── metrics.py             # Dice-Sørensen coefficient, IoU (Jaccard Index)
│   ├── grad_cam.py            # Grad-CAM heatmaps for encoder sanity checks
│   └── umap_viz.py            # UMAP of 2048-d ResNet-50 feature vectors
├── inference.py               # End-to-end inference on new CT datasets
└── scripts/
    ├── 01_preprocess.sh
    ├── 02_train_simclr.sh
    ├── 03_generate_masks.sh
    ├── 04_train_unet.sh
    └── 05_run_inference.sh
```

---

## Pipeline Workflow

```
Raw CT TIFF slices
       │
       ▼
[1] Preprocessing          resize → 224×224 | normalise [0,1] | z-standardise | → HDF5
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
[2] SimCLR Pre-training                  [3] Rule-Based Masking
    ResNet-50 + NT-Xent loss                 5-stage deterministic pipeline
    250 epochs | 39,037 unlabelled slices    Otsu → morphological refinement
    Feature vectors: 2048-d                  Coarse binary masks
       │                                          │
       └──────────────┬───────────────────────────┘
                      ▼
            [4] Knowledge Fusion → U-Net
                SimCLR weights initialise encoder
                Composite WCE + Dice loss | 500 epochs
                Fine segmentation masks
                      │
                      ▼
            [5] 3D Mesh Generation
                Stack 2D masks → region growing → marching cubes
                Watertight STL (1–3 min per specimen)
                      │
                      ▼
            [6] Evaluation
                Dice / IoU | Grad-CAM | UMAP | CloudCompare C2C
```

---

## Installation

```bash
git clone https://github.com/roy-arindam-1991/Simclr-v1-automated-pipeline.git
cd Simclr-v1-automated-pipeline
conda env create -f environment.yml
conda activate fossil-ct
```

Or with pip:

```bash
pip install -r requirements.txt
```

---

## Quick Start

```bash
# 1. Preprocess raw TIFF stacks
bash scripts/01_preprocess.sh

# 2. Pre-train SimCLR on unlabelled slices
bash scripts/02_train_simclr.sh

# 3. Generate rule-based coarse masks
bash scripts/03_generate_masks.sh

# 4. Train U-Net via knowledge fusion
bash scripts/04_train_unet.sh

# 5. Run inference on new specimens
bash scripts/05_run_inference.sh
```

Or run end-to-end via Python:

```python
from inference import FossilSegmentationPipeline

pipeline = FossilSegmentationPipeline.from_config("config/config.yaml")
pipeline.run(ct_dir="path/to/new/specimen/tiffs/", output_dir="outputs/")
```

---

## Data

Training data: 50,626 CT images from the Middle Jurassic Kilmaluag Formation, Skye, Scotland, spanning amphibians, archosaurs, pterosaurs, dinosaurs, and early mammals. Specimens are housed at **National Museums Scotland (NMS)**. CT volumes are available on request from MorphoSource ([www.morphosource.org](https://www.morphosource.org)).

External validation specimens (MorphoSource accession numbers):
| Specimen | Accession |
|---|---|
| Salamander A (Specimen 1) | 00084381 |
| Salamander A (Specimen 2) | 000071513 (rescan) |
| *M. wakei* (Specimen 1) | 000700518 |
| *M. wakei* (Specimen 2) | 000700519 |
| *Mammaliaformes* indet. | 000693738 |
| Docodonta (undescribed) | 000721884 |

---

## Hardware

Trained on a single **NVIDIA A100 GPU** via BlueBear HPC (University of Birmingham).

| Stage | Time |
|---|---|
| SimCLR pre-training (250 epochs) | ~37–38 hrs |
| U-Net training (500 epochs) | ~6 hrs 10 min |
| Peak GPU memory | ~3.26 GB |
| Inference per specimen | 1–3 min |

---

## Citation

```bibtex
@article{roy2025breakingbottleneck,
  title   = {Breaking the bottleneck: self-supervised deep learning for fully automated fossil {CT} segmentation},
  author  = {Roy, Arindam and Ghosh, Poulami and Benson, Roger and Weston, Fraser and Scott, Ben and Salili-James, Arianna and Poon, Sanson T.S. and Walsh, Stig and Maidment, Susannah and Butler, Richard},
  journal = {[journal to be confirmed upon acceptance]},
  year    = {2025}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
