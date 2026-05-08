#!/bin/bash
#SBATCH --qos gpus
#SBATCH --partition a100
#SBATCH --time 4:0:0
#SBATCH --mem 64G

# Integrated Preprocessing Pipeline
# Order: 1. Data Prep | 2. Mask Prep | 3. H5 Conversion

module purge
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-foss-2023a

# Using dynamic path arguments instead of hard-coded values
python3 ../python/01_preprocessing_hdf5.py     --input "../../data/raw"     --output "../../data/processed"
