#!/bin/bash
PROJ_ROOT="${1:-./}"
module load bear-apps/2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
python3 "${PROJ_ROOT}/scripts/python/train_unet_simclr.py" \
    --image_h5 "${PROJ_ROOT}/data/images.h5" \
    --mask_h5 "${PROJ_ROOT}/data/masks.h5" \
    --backbone "${PROJ_ROOT}/models/simclr.pth"
