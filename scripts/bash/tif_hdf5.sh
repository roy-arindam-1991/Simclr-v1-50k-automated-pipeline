#!/bin/bash
#SBATCH --account=butlerry-deepctseg
#SBATCH --job-name=TIFF_to_HDF5
#SBATCH --output=TIFF_to_HDF5_test_%j.log
#SBATCH --error=TIFF_to_HDF5_test_%j.err
#SBATCH --ntasks=16
#SBATCH --mem=128G 
#SBATCH --time=24:00:00
#SBATCH --qos=bbdefault
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.roy.2@bham.ac.uk,pxg491@alumni.bham.ac.uk

set -e

# ================= Configuration (All Paths Here) =================
SCRIPT_PATH="/rds/projects/b/butlerry-deepctseg/40K_pipeline/data_prep/tif_hdf5.py"
INPUT_DIR="/rds/projects/b/butlerry-deepctseg/40K_pipeline/data_masking/test_masks_tif"
OUTPUT_DIR="/rds/projects/b/butlerry-deepctseg/40K_pipeline/data_masking/test_masks_h5"

# Define filenames for the outputs
H5_NAME="val_data.h5"
MANIFEST_NAME="h5_val_manifest.log"
PARQUET_NAME="hu_val_values.parquet"
LOG_NAME="val_conversion_process.log"

# ================= Load modules =================
module purge 
module load bluebear
module load bear-apps/2023a
module load h5py/3.9.0-foss-2023a
module load scikit-image/0.22.0-foss-2023a
module load geopandas/0.14.2-foss-2023a
module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0
module load Arrow/14.0.1-gfbf-2023a

# ================= Execution =================
mkdir -p "$OUTPUT_DIR"

python3 "$SCRIPT_PATH" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --h5_name "$H5_NAME" \
    --manifest_name "$MANIFEST_NAME" \
    --parquet_name "$PARQUET_NAME" \
    --log_name "$LOG_NAME"

echo "Recursive TIFF → HDF5 job complete!"