# 40K Bone CT Segmentation Pipeline

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Accuracy](https://img.shields.io/badge/SSL_Accuracy-93.66%25-brightgreen.svg)
![Dice](https://img.shields.io/badge/Segmentation_Dice-0.93-blue.svg)

This repository documents a comprehensive deep learning framework for the automated analysis and segmentation of fossilized structures using 3D CT imaging. The pipeline is organized into four distinct phases.

## Phase 0: Data Preprocessing
The initial phase focuses on converting high-resolution volumetric CT scans into a standardized format optimized for deep learning performance and hardware efficiency.
* **Normalization and Scaling**: Raw TIF stacks are processed to ensure consistent intensity distributions. A fixed scaling factor is applied to normalize the density values of fossilized bone across different specimens.
* **Volume Partitioning**: Volumetric data is partitioned into 2D slices. This allows for high-throughput training and ensures that the model learns from diverse anatomical cross-sections.
* **HDF5 Integration**: Processed slices are stored in HDF5 format. This provides high-speed I/O performance on HPC clusters and manages metadata for thousands of images within a single file structure.

## Phase 1: Training (Self-Supervised SimCLR)
The first phase involves training a ResNet50 backbone to learn rich morphological representations of fossil bone without manual labels.
* **Accuracy**: The model achieved a peak validation accuracy of 93.66% over 250 epochs.
* **Convergence**: Training was executed on NVIDIA A100 hardware. The learning rate decayed from 0.0003 to near zero, ensuring stable convergence of the NT-Xent contrastive loss.
* **Feature Learning**: The high accuracy indicates the model successfully captured complex internal structures and textures of bone without requiring ground-truth masks.

## Phase 2: Validation (Feature Extraction and Manifold Analysis)
The second phase evaluates the learned feature space to ensure the model has developed a biologically meaningful understanding of the data.
* **Manifold Analysis**: UMAP dimensionality reduction was applied to the latent space to visualize feature distributions.
* **Clustering Integrity**: Analysis confirmed the successful clustering of 7,824 valid 2D slices. Clear separation in the UMAP plot suggests the model effectively distinguished between different bone morphologies and matrix types.
* **Reliability**: This phase confirms the pre-trained weights provide a robust and representative foundation for downstream segmentation.

## Phase 3: UNet (Downstream Segmentation)
The final phase integrates the pre-trained ResNet50 weights into a UNet architecture for binary segmentation of bone versus the surrounding matrix.
* **Segmentation Quality**: The model achieved a stable Dice Coefficient of 0.93 and an Intersection over Union (IoU) of 0.82.
* **Optimization**: To mitigate metric volatility, a Hybrid Fossil Loss (BCE + Tversky) was utilized, which is particularly effective for the class imbalances found in fossil scans.
* **Conclusion**: The 0.93 Dice score proves that transferring self-supervised features significantly improves boundary localization. The pipeline accurately identifies bone structures in low-contrast regions where traditional segmentation methods often fail.

## Repository Structure
* scripts/python/: Generalized scripts for preprocessing, pre-training, validation, and UNet segmentation.
* scripts/bash/: Slurm templates for execution on HPC environments.
* requirements.txt: List of necessary dependencies.

## Usage
Configure Slurm headers in the bash templates for your specific cluster environment before execution.

bash
# Phase 0 & 1: Data Prep and Pre-training
sbatch scripts/bash/train_simclr_ct.sh

# Phase 2: Validation
sbatch scripts/bash/validate_simclr.sh

# Phase 3: Segmentation
sbatch scripts/bash/train_unet_simclr.sh
