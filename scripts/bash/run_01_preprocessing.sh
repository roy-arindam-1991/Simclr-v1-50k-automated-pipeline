#!/bin/bash
#SBATCH --qos gpus
#SBATCH --partition a100
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --time 4:0:0
#SBATCH --mem 64G

# Pipeline Stage: Preprocessing and Normalisation
# Sequence: 1. Data Prep -> 2. Mask Prep -> 3. HDF5 Conversion

# Clear existing modules and load required software stack
module purge
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-foss-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

# Execute the integrated preprocessing script
# This script handles partitioning, deterministic masking, and final H5 packaging
python3 ../python/01_preprocessing_hdf5.py     --input_dir "../../data/raw"     --output_dir "../../data/processed"     --seed 42

# Log completion status
if [ $? -eq 0 ]; then
    echo "Preprocessing pipeline completed successfully: Data Prep, Mask Prep, and H5 Conversion finished."
else
    echo "Preprocessing pipeline failed. Check logs for details."
    exit 1
fi
