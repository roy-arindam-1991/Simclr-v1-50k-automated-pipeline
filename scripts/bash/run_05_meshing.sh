#!/bin/bash
#SBATCH --nodes 1
#SBATCH --time 0:30:0
#SBATCH --mem 16G

# Volumetric 3D Mesh Generation: Marching Cubes and STL extraction.
module purge
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-foss-2023a

python3 ../python/05_mesh_reconstruction.py
