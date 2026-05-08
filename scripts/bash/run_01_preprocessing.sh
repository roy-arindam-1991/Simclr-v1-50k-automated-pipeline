#!/bin/bash
#SBATCH --qos gpus
#SBATCH --partition a100
#SBATCH --nodes 1
#SBATCH --gpus-per-node 1
#SBATCH --time 4:0:0
#SBATCH --mem 64G

# Integrated Preprocessing Pipeline
# Order of Execution: 1. Data Prep | 2. Mask Prep | 3. HDF5 Conversion

# Clear environment and load manuscript-specified software bundles
module purge
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-foss-2023a

# Execute Python framework with dynamic directory arguments
# This maintains the 19/2/3 specimen partitioning logic
python3 ../python/01_preprocessing_hdf5.py     --input "../../data/raw"     --output "../../data/processed"

# Verification check for downstream SimCLR phase
if [ $? -eq 0 ]; then
    echo "Success: Preprocessing, deterministic masking, and H5 packaging complete."
else
    echo "Pipeline failure in the preprocessing stage."
    exit 1
fi
