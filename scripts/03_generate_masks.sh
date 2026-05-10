#!/usr/bin/env bash
set -euo pipefail
echo "=== Stage 3: Rule-Based Deterministic Masking ==="
python - << 'PYEOF'
import numpy as np
from pathlib import Path
from data.hdf5_converter import FossilHDF5Dataset
from masking.rule_based_masking import generate_masks_for_volume

for hdf5_path in sorted(Path("data/hdf5/").glob("*.h5")):
    print(f"Masking {hdf5_path.stem} ...")
    ds = FossilHDF5Dataset([hdf5_path])
    volume = np.stack([ds[i] for i in range(len(ds))], axis=0)
    masks = generate_masks_for_volume(volume)
    out = Path("data/masks/") / f"{hdf5_path.stem}_masks.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, masks)
    ds.close()
    print(f"  -> {out}")
PYEOF
echo "Masks saved to data/masks/"
