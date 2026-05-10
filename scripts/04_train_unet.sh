#!/usr/bin/env bash
set -euo pipefail
echo "=== Stage 4: U-Net Training via Knowledge Fusion (500 epochs, ~6 hrs on A100) ==="
python unet/train.py
echo "Best checkpoint → checkpoints/unet/unet_best.pt"
