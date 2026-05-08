#!/bin/bash
PROJ_ROOT="${1:-./}"
module load bear-apps/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
python3 "${PROJ_ROOT}/scripts/python/ct_mesh.py" --input_h5 "${PROJ_ROOT}/data/results.h5" --output_stl "${PROJ_ROOT}/output/fossil.stl"
