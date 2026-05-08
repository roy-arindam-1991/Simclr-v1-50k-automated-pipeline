#!/bin/bash
#SBATCH --qos gpus
#SBATCH --partition a100
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --time 40:0:0
#SBATCH --mem 64G

# SimCLR v1 Self-Supervised Pre-training. 
# Targets 93.66% accuracy over 250 epochs.
module purge
module load bear-apps/2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

python3 ../python/02_simclr_pretraining.py --epochs 250 --batch_size 64 --lr 0.0003
