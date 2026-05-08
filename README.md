# SimCLR v1 CT Segmentation Pipeline

![CI/CD Pipeline(https://github.com/roy-arindam-1991/Simclr-v1-automated-pipeline/actions/workflows/main.yml/badge.svg)
![Python(https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch(https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg)
![License(https://img.shields.io/badge/license-GPLv3-blue.svg)
![Accuracy(https://img.shields.io/badge/SSL_Accuracy-93.66%25-brightgreen.svg)
![Dice(https://img.shields.io/badge/Segmentation_Dice-0.93-blue.svg)

This repository provides the core technical implementation for the study: **"Breaking the bottleneck: self-supervised deep learning for fully automated fossil CT segmentation"**. The framework introduces a self-supervised, end-to-end pipeline combining SimCLR v1 contrastive pre-training with deterministic pseudo-label generation and U-Net refinement to fully automate fossil CT segmentation without manual annotation.

## Key Contributors
* **Arindam Roy**: Lead Researcher and Maintainer
* **[Poulami Ghosh(https://github.com/g-Poulami)**: Contributor and Pipeline Developer

## Pipeline Workflow and Structure
The framework integrates heterogeneous information sources into a unified model representation through a knowledge fusion step.

1. **Preprocessing and Normalisation**: Standardisation of CT slices into HDF5 format with intensity normalisation and z-standardisation to maintain consistency across the 50,626 image corpus.
2. **SimCLR v1 Self-Supervised Pre-training**: A ResNet-50 base encoder learns domain-specific feature representations from unlabelled CT data through a stochastic data augmentation pipeline.
3. **Post-hoc Validation**: Quality checks using UMAP dimensionality reduction and Grad-CAM heatmaps to ensure the learned representations encode biologically meaningful structures.
4. **U-Net Refinement (Knowledge Fusion)**: A modified U-Net architecture integrates SSL-derived features with standardised spatial guidance from deterministic rule-based masks to resolve ambiguous fossil-matrix boundaries.
5. **Volumetric 3D Mesh Generation**: Stacking 2D segmentation masks to reconstruct 3D volumes, followed by surface extraction using the marching cubes algorithm.

```text
.
├── .github/workflows/   # Automated code integrity validation (CI/CD)
├── scripts/
│   ├── python/          # Algorithmic implementations (Preprocessing to Meshing)
│   └── bash/            # HPC/Slurm execution templates (BlueBear HPC)
├── LICENSE              # GNU GPLv3 (Arindam Roy)
├── README.md            # Technical documentation
└── requirements.txt     # Python environment manifest (Python 3.11)
```

## Research Summary
* **Scale**: Implemented on a taxonomically diverse corpus representing a five-fold increase over the largest prior training sets in palaeontology.
* **Performance**: Achieved a Dice coefficient of 93.66% and IoU of 82.42% on held-out specimens.
* **Generalisation**: Validated geometrically on six external specimens, achieving sub-voxel mesh agreement with manually thresholded references.
* **Automation**: Reduces per-specimen processing from ~100 person-hours to 1–3 minutes of inference time.

## Data and Model Availability
In accordance with journal guidelines, raw CT data and generated 3D meshes are hosted via National Museums Scotland (NMS) and MorphoSource. Model weights and hyperparameter specifics are available from the corresponding author upon request.

## License
Licensed under the GNU General Public License v3.0. Copyright (c) 2026 Arindam Roy.
