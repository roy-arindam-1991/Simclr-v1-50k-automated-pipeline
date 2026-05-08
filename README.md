# 40K Bone CT Segmentation Pipeline

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Accuracy](https://img.shields.io/badge/SSL_Accuracy-93.66%25-brightgreen.svg)
![Dice](https://img.shields.io/badge/Segmentation_Dice-0.93-blue.svg)

This repository documents a comprehensive deep learning framework for the automated analysis and segmentation of fossilized structures using 3D CT imaging. The pipeline is organized into five distinct phases.

## Phase 0: Data Preprocessing
The initial phase focuses on converting high-resolution volumetric CT scans into a standardized format optimized for deep learning.
* **Normalization and Scaling**: Raw TIF stacks are processed to ensure consistent intensity distributions, applying a fixed scaling factor to normalize density values.
* **HDF5 Integration**: Processed slices are stored in HDF5 format to provide high-speed I/O performance on HPC clusters.

## Phase 1: Training (Self-Supervised SimCLR)
This phase involves training a ResNet50 backbone via contrastive learning to recognize morphological features without manual labels.
* **Accuracy**: The model achieved a peak validation accuracy of 93.66% over 250 epochs.
* **Convergence**: Training was executed on NVIDIA A100 hardware, with stable convergence of the NT-Xent loss.

## Phase 2: Validation (Feature Extraction and Manifold Analysis)
The learned feature space is evaluated to ensure a biologically meaningful understanding of the data.
* **Manifold Analysis**: UMAP dimensionality reduction successfully clustered 7,824 valid 2D slices.
* **Clustering Integrity**: Clear separation in the UMAP coordinates suggests the model effectively distinguished between bone morphologies and matrix types.

## Phase 3: UNet (Downstream Segmentation)
The pre-trained ResNet50 weights are integrated into a UNet architecture for binary segmentation of bone versus matrix.
* **Segmentation Quality**: The model reached a stable Dice Coefficient of 0.93 and an Intersection over Union (IoU) of 0.82.
* **Optimization**: A Hybrid Fossil Loss (BCE + Tversky) was utilized to handle class imbalances and improve boundary localization in low-contrast regions.

## Phase 4: 3D Mesh Generation and Volumetric Analysis
The final phase reconstructs 2D predictions into 3D manifolds and calculates scientific volumetric metrics.
* **Surface Reconstruction**: Employs Marching Cubes and Laplacian smoothing to generate high-fidelity STL meshes from segmented volumes.
* **Morphometric Reporting**: The pipeline calculates the total scientific volume and provides detailed reports on voxel counts and mesh triangle density.
* **Decimation**: Includes mesh decimation logic to maintain structural integrity while reducing the file size for smaller computational footprints.

## Repository Structure
* scripts/python/: Generalized scripts for preprocessing, SSL training, validation, UNet segmentation, and 3D meshing.
* scripts/bash/: Slurm templates for automated execution on HPC environments.

## Usage
Configure Slurm headers in the bash templates for your specific cluster environment before execution.

bash
# Phase 0 & 1: Pre-training
sbatch scripts/bash/train_simclr_ct.sh

# Phase 2: Validation
sbatch scripts/bash/validate_simclr.sh

# Phase 3: Segmentation
sbatch scripts/bash/train_unet_simclr.sh

# Phase 4: 3D Meshing
sbatch scripts/bash/ct_mesh.sh
