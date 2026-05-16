#!/usr/bin/env bash
set -euo pipefail
echo "=== Stage 1: Preprocessing TIFF stacks → HDF5 ==="
python - << 'PYEOF'
from data.hdf5_converter import batch_convert
batch_convert("data/raw/", "data/hdf5/", config_path="config/config.yaml")
PYEOF
echo "Done. HDF5 files written to data/hdf5/"
