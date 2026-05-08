#!/bin/bash
#SBATCH --account=butlerry-deepctseg
#SBATCH --job-name=split_tif_data
#SBATCH --output=split_tif_%j.log
#SBATCH --error=split_tif_%j.err
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --qos=bbdefault
#SBATCH --mail-type=NONE
#SBATCH --mail-user=a.roy.2@bham.ac.uk,pxg491@alumni.bham.ac.uk

set -e

echo "Running split_tif job on $(hostname)"
echo "Start time: $(date)"

# ================= Configuration =================
SCRIPT_PATH="/rds/projects/b/butlerry-deepctseg/40K_pipeline/data_prep/split_tif.py"
INPUT_DIR="/rds/projects/b/butlerry-deepctseg/40K_pipeline/40k_data"
OUTPUT_DIR="/rds/projects/b/butlerry-deepctseg/40K_pipeline/data_prep_results/split_tif_files"

TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1
MODE="symlink"

# ================= Load modules =================
module purge
module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0

# Ensure dependencies for Excel output are installed in user space
pip install --user pandas openpyxl

# ================= Validation =================
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Python script not found at $SCRIPT_PATH"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found at $INPUT_DIR"
    exit 1
fi

# ================= Run Python script =================
echo "Splitting subfolders from: $INPUT_DIR"
echo "Results will be placed in: $OUTPUT_DIR"

python3 "$SCRIPT_PATH" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --train_ratio $TRAIN_RATIO \
  --val_ratio $VAL_RATIO \
  --test_ratio $TEST_RATIO \
  --mode "$MODE"

echo "Job complete at: $(date)"