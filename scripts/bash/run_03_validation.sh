#!/bin/bash
#SBATCH --qos gpus
#SBATCH --partition a100
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --time 1:0:0
#SBATCH --mem 32G

# Post-hoc Validation: UMAP and Grad-CAM generation.
module purge
module load bear-apps/2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

python3 ../python/03_manifold_validation.py
