#!/bin/bash
# Slurm job script for TIFF to HDF5 conversion.
# Configure #SBATCH directives here as per your cluster requirements.

PROJ_DIR="${1:-./}"

python3 "${PROJ_DIR}/scripts/python/tif_hdf5.py" \
    --input_dir "${PROJ_DIR}/input_data" \
    --output_dir "${PROJ_DIR}/output" \
    --h5_name "data.h5" \
    --manifest_name "manifest.log" \
    --parquet_name "values.parquet" \
    --log_name "conversion.log"
