#!/bin/bash
# Slurm template for SimCLR Training.
# Note: Manually add #SBATCH --account and --mail-user for your specific cluster.

PROJ_ROOT="${1:-./}"

module purge
module load bear-apps/2023a
module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load h5py/3.9.0-foss-2023a

python3 "${PROJ_ROOT}/scripts/python/train_simclr_ct.py" \
  --data_h5 "${PROJ_ROOT}/data/ct_data.h5" \
  --outdir "${PROJ_ROOT}/output/simclr_ct" \
  --epochs 100 \
  --batch_size 256 \
  --temperature 0.1 \
  --fp16
