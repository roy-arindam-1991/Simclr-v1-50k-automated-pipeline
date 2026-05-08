#!/bin/bash
# Slurm job script for splitting TIFF data. 
# Configure #SBATCH directives here as per your cluster requirements.

# Use a project directory variable for portability
PROJ_DIR="${1:-./}"

# Run Python script with generalized arguments
python3 "${PROJ_DIR}/scripts/python/split_tif.py" \
  --input_dir "${PROJ_DIR}/input_data" \
  --output_dir "${PROJ_DIR}/results" \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1
