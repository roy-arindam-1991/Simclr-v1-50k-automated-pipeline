#!/bin/bash
PROJ_ROOT="${1:-./}"
module load bear-apps/2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
python3 "${PROJ_ROOT}/scripts/python/validate_simclr.py" \
    --data_h5 "${PROJ_ROOT}/data/val.h5" \
    --checkpoint "${PROJ_ROOT}/models/best.pth"
