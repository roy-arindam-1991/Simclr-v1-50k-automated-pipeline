# SimCLR v1 CT Segmentation Pipeline

![CI/CD Pipeline](https://github.com/roy-arindam-1991/Simclr-v1-automated-pipeline/actions/workflows/main.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg)
![License](https://img.shields.io/badge/license-GPLv3-blue.svg)
![Accuracy](https://img.shields.io/badge/SSL_Accuracy-93.66%25-brightgreen.svg)
![Dice](https://img.shields.io/badge/Segmentation_Dice-0.93-blue.svg)

This repository provides a comprehensive computational framework for the automated analysis and 3D reconstruction of fossilized structures using volumetric CT imaging. This pipeline represents the core technical implementation for the study targeted for Nature Machine Intelligence.

## Key Contributors
* **Arindam Roy**: Lead Researcher and Maintainer
* **[Poulami Ghosh](https://github.com/g-Poulami)**: Contributor and Pipeline Developer

## Pipeline Workflow and Structure
The workflow is a linear sequence of five research phases:

1. **Phase 0: Data Preprocessing**: Volumetric standardization (Raw TIF to HDF5).
2. **Phase 1: Training**: Self-Supervised SimCLR ResNet50 pre-training.
3. **Phase 2: Validation**: Manifold Analysis via UMAP feature clustering.
4. **Phase 3: UNet**: Downstream fine-tuning for bone segmentation.
5. **Phase 4: 3D Reconstruction**: Automated mesh generation and volumetrics.

```text
.
├── .github/workflows/   # Automated code integrity validation (CI/CD)
├── scripts/
│   ├── python/          # Phase 0-4 algorithmic implementations
│   └── bash/            # HPC/Slurm execution templates
├── LICENSE              # GNU GPLv3 (Arindam Roy)
├── README.md            # Technical documentation
└── requirements.txt     # Python environment manifest
```

## Research Summary

### Phase 1: Training (Self-Supervised SimCLR)
A ResNet50 backbone is trained via contrastive learning to recognize morphological bone features without manual labels. The model achieved a peak validation accuracy of 93.66% over 250 epochs on NVIDIA A100 hardware.

### Phase 2: Validation
UMAP manifold analysis confirmed the successful clustering of 7,824 valid 2D slices. This demonstrates the backbone's ability to distinguish complex biological morphologies in the latent space.

### Phase 3: UNet (Downstream Segmentation)
Utilizing a ResNet50-UNet architecture with Hybrid Fossil Loss (BCE + Tversky), the pipeline achieved a stable Dice Coefficient of 0.93 and an IoU of 0.82.

### Phase 4: 3D Reconstruction
Segmented 2D predictions are reconstructed into 3D manifolds using Marching Cubes and Laplacian smoothing. This phase provides automated morphometric reporting and high-resolution STL generation.

## CI/CD and Automation
This repository utilizes GitHub Actions to automatically validate code integrity on every push to the main branch.

## Data and Model Availability
In accordance with journal guidelines, the raw CT data and generated 3D meshes are not hosted in this repository. These assets will be made available through official supplementary data repositories upon publication.

## License
Licensed under the GNU General Public License v3.0. Copyright (c) 2026 Arindam Roy.
