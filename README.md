# SimCLR v1 CT Segmentation Pipeline

![CI/CD Pipeline](https://github.com/roy-arindam-1991/Simclr-v1-automated-pipeline/actions/workflows/main.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg)
![License](https://img.shields.io/badge/license-GPLv3-blue.svg)
![Accuracy](https://img.shields.io/badge/SSL_Accuracy-93.66%25-brightgreen.svg)
![Dice](https://img.shields.io/badge/Segmentation_Dice-0.93-blue.svg)

Technical implementation for: **"Breaking the bottleneck: self-supervised deep learning for fully automated fossil CT segmentation"**.

## Key Contributors
* **Arindam Roy**: Lead Researcher
* **Poulami Ghosh**: Pipeline Developer

## Pipeline Stages
1. **Data Prep**: HDF5 conversion and taxonomic partitioning of 50,626 images.
2. **Mask Prep**: Automated generation of binary spatial priors via deterministic processing.
3. **SimCLR Training**: SSL feature extraction using ResNet-50.
4. **U-Net Refinement**: Fine-tuning via knowledge fusion.
5. **3D Reconstruction**: STL mesh generation via marching cubes.
