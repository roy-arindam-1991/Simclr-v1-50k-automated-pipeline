#!/bin/bash
#SBATCH --qos gpus
#SBATCH --partition a100
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --time 8:0:0
#SBATCH --mem 32G

# U-Net Refinement: Knowledge fusion of SSL weights and deterministic masks.
module purge
module load bear-apps/2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

python3 ../python/04_unet_refinement.py --epochs 500 --loss hybrid
