# 40K Bone CT Segmentation Pipeline

![CI/CD Pipeline](https://github.com/roy-arindam-1991/40k-pipeline-dataprep/actions/workflows/main.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Accuracy](https://img.shields.io/badge/SSL_Accuracy-93.66%25-brightgreen.svg)
![Dice](https://img.shields.io/badge/Segmentation_Dice-0.93-blue.svg)

This repository provides a comprehensive computational framework for the automated analysis and 3D reconstruction of fossilized structures using volumetric CT imaging. The pipeline transitions from raw data standardization to self-supervised feature learning and high-fidelity mesh generation.

## Pipeline Workflow and Repository Structure

The workflow is designed as a linear sequence of five phases:

1. Phase 0: Data Preprocessing (Raw TIF to HDF5)
2. Phase 1: Self-Supervised Training (SimCLR ResNet50)
3. Phase 3: Validation (Manifold Analysis via UMAP)
4. Phase 3: Downstream Segmentation (ResNet50-UNet)
5. Phase 4: 3D Reconstruction (Mesh Generation and Volumetrics)

```text
.
├── .github/workflows/
│   └── main.yml              # CI/CD Pipeline configuration
├── scripts/
│   ├── python/
│   │   ├── split_tif.py       # Phase 0: Volumetric partitioning
│   │   ├── train_simclr_ct.py # Phase 1: SSL backbone training
│   │   ├── validate_simclr.py # Phase 2: Feature space evaluation
│   │   ├── train_unet_simclr.py # Phase 3: Bone segmentation
│   │   └── ct_mesh.py         # Phase 4: 3D STL generation
│   └── bash/
│       ├── split_tif.sh       # Slurm: Data prep template
│       ├── train_simclr_ct.sh # Slurm: Pre-training template
│       ├── validate_simclr.sh # Slurm: Validation template
│       ├── train_unet_simclr.sh # Slurm: Fine-tuning template
│       └── ct_mesh.sh         # Slurm: Reconstruction template
├── LICENSE                    # MIT License (Arindam Roy)
├── README.md                  # Project Documentation
└── requirements.txt           # Dependency Manifest
```

## Phase 0: Data Preprocessing
The initial phase converts raw volumetric CT scans into standardized HDF5 formats. This process includes intensity normalization and fixed scaling to ensure consistent density distributions across different specimens, providing high-speed I/O performance on HPC clusters.

## Phase 1: Training (Self-Supervised SimCLR)
A ResNet50 backbone is trained via contrastive learning to recognize morphological bone features without manual labels.
* Results: The model achieved a peak validation accuracy of 93.66% over 250 epochs.
* Convergence: Stable convergence was reached on NVIDIA A100 hardware with learning rate decay from 0.0003 to near zero.

## Phase 2: Validation (Manifold Analysis)
The learned latent space is evaluated using UMAP dimensionality reduction to ensure biologically meaningful feature clustering.
* Results: Analysis confirmed the successful clustering of 7,824 valid 2D slices, showing distinct morphological separation between bone structures and matrix types.

## Phase 3: UNet (Downstream Segmentation)
Pre-trained SSL weights are integrated into a UNet architecture for binary segmentation.
* Results: The pipeline reached a stable Dice Coefficient of 0.93 and an Intersection over Union (IoU) of 0.82.
* Optimization: Implementation of a Hybrid Fossil Loss (BCE + Tversky) effectively handled class imbalances and localized boundaries in low-contrast regions.

## Phase 4: 3D Mesh Generation and Volumetric Analysis
Segmented 2D volumes are reconstructed into 3D manifolds for quantitative analysis.
* Methodology: Employs Marching Cubes and Laplacian smoothing for high-fidelity STL generation.
* Output: Provides scientific volumetric reports, including voxel counts and triangle density for morphometric analysis.

## CI/CD and Automation
This repository utilizes GitHub Actions to automatically validate code integrity. The CI/CD pipeline runs on every push to the main branch, checking for syntax errors and dependency compatibility.

## License
Licensed under the MIT License. Copyright (c) 2026 Arindam Roy.
