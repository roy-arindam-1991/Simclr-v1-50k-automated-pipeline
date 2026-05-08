#!/bin/bash
#SBATCH --nodes 1
#SBATCH --time 0:30:0
#SBATCH --mem 16G

# Stage: Volumetric 3D Mesh Generation
# Objective: Generate watertight STL meshes via marching cubes

module purge
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-foss-2023a

# Processes per-specimen 3D meshes in 1 to 3 minutes
python3 ../python/05_mesh_reconstruction.py     --mask_h5 "../../data/processed/unet_test_output.h5"     --output_dir "../../results/meshes"
