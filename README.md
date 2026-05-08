# 40K Bone CT Segmentation Pipeline

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![CI/CD](https://github.com/roy-arindam-1991/40k-pipeline-dataprep/workflows/CI/CD%20Pipeline/badge.svg)

This repository documents a comprehensive deep learning framework for the automated analysis and segmentation of fossilized structures using 3D CT imaging. 

## Pipeline Workflow
The pipeline operates in a linear sequence to transform raw volumetric data into scientifically actionable 3D models:

1. **Preprocessing**: Standardization of raw TIF stacks into optimized HDF5 files.
2. **SSL Pre-training**: Feature learning via SimCLR to understand bone morphology without labels.
3. **Manifold Validation**: UMAP analysis to verify the integrity of the learned latent space.
4. **Segmentation**: Fine-tuning a ResNet50-UNet for precise bone extraction.
5. **3D Reconstruction**: Generating smoothed STL meshes and volumetric reports.

## Repository Structure
```text
.
├── .github/workflows/
│   └── main.yml              # CI/CD pipeline configuration
├── scripts/
│   ├── python/
│   │   ├── split_tif.py       # Phase 0: Data preparation
│   │   ├── train_simclr_ct.py # Phase 1: SSL Training
│   │   ├── validate_simclr.py # Phase 2: UMAP Validation
│   │   ├── train_unet_simclr.py # Phase 3: Segmentation
│   │   └── ct_mesh.py         # Phase 4: 3D Meshing
│   └── bash/
│       ├── split_tif.sh       # Slurm: Preprocessing
│       ├── train_simclr_ct.sh # Slurm: SSL Training
│       ├── validate_simclr.sh # Slurm: Validation
│       ├── train_unet_simclr.sh # Slurm: UNet fine-tuning
│       └── ct_mesh.sh         # Slurm: 3D Reconstruction
├── LICENSE                    # MIT License (Arindam Roy)
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

## Research Results
* **SSL Accuracy**: 93.66%
* **Segmentation Dice**: 0.93
* **Segmentation IoU**: 0.82

## CI/CD and Automation
This repository uses GitHub Actions to automatically lint code and verify dependency compatibility on every push to the main branch.

## License
Licensed under the MIT License. Copyright (c) 2026 Arindam Roy.
