#!/bin/bash
#SBATCH --account=butlerry-deepctseg
#SBATCH --job-name=split_tif_data
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00

PROJ_DIR="${1:-./}"
python3 "${PROJ_DIR}/scripts/python/split_tif.py"   --input_dir "${PROJ_DIR}/input_data"   --output_dir "${PROJ_DIR}/results"   --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
