#!/bin/bash
#SBATCH --qos gpus
#SBATCH --partition a100
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --time 8:0:0
#SBATCH --mem 32G

# Stage: U-Net Refinement (Knowledge Fusion)
# Objective: Fine-tune segmentation using SSL weights and spatial priors

module purge
module load bear-apps/2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

# Training uses composite Weighted Cross-Entropy and Dice loss
# Target: Dice coefficient of 0.9366 and IoU of 0.8242
python3 ../python/04_unet_refinement.py     --train_h5 "../../data/processed/unet_refinement.h5"     --epochs 500
