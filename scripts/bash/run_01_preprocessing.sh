#!/bin/bash
#SBATCH --qos gpus
#SBATCH --partition a100
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --time 2:0:0
#SBATCH --mem 32G

# Preprocessing: Converts raw TIF stacks into standardized HDF5 format.
module purge
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0

python3 ../python/01_preprocessing_hdf5.py
