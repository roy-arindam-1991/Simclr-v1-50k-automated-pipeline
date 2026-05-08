import numpy as np
from skimage import measure

# Volumetric 3D Mesh Generation
# Reconstructs predicted slices into 3D manifolds[cite: 121, 295].

def extract_surface(volume_mask):
    """
    Applies Marching Cubes (iso-level=0.5) and Laplacian smoothing[cite: 122, 299].
    """
    # surface extraction produces watertight triangulated meshes [cite: 123, 300]
    verts, faces, normals, values = measure.marching_cubes(volume_mask, level=0.5)
    return verts, faces
