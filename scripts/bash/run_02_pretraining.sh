#!/bin/bash
#SBATCH --qos gpus
#SBATCH --partition a100
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --time 40:0:0
#SBATCH --mem 64G

# Stage: SimCLR v1 Self-Supervised Pre-training
# Objective: Learn feature representations without manual labels

module purge
module load bear-apps/2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

# Hyperparameters: 250 Epochs | Batch Size 64 | Learning Rate 3e-4
# Accuracy Target: 93.66% on the held-out validation set
python3 ../python/02_simclr_pretraining.py     --data_h5 "../../data/processed/simclr_train.h5"     --epochs 250     --lr 0.0003
