#!/usr/bin/env bash
set -euo pipefail
echo "=== Stage 2: SimCLR Pre-training (250 epochs, ~37-38 hrs on A100) ==="
python simclr/train.py
echo "Best checkpoint → checkpoints/simclr/simclr_best.pt"
