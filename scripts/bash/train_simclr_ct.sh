#!/bin/bash
PROJ_ROOT="${1:-./}"
module load bear-apps/2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
python3 "${PROJ_ROOT}/scripts/python/train_simclr_ct.py" --data_h5 "${PROJ_ROOT}/data/train.h5" --out_dir "${PROJ_ROOT}/output"
