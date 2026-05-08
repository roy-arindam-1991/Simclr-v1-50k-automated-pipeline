import os
import argparse
import numpy as np
import h5py
from skimage import measure
import open3d as o3d

def generate_mesh(volume, threshold=0.5):
    verts, faces, normals, values = measure.marching_cubes(volume, threshold)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_h5", required=True)
    parser.add_argument("--output_stl", required=True)
    args = parser.parse_args()
    
    with h5py.File(args.input_h5, 'r') as f:
        data = np.stack([f['masks'][k][()] for k in sorted(f['masks'].keys())])
    
    mesh = generate_mesh(data)
    o3d.io.write_triangle_mesh(args.output_stl, mesh)
    print(f"Phase 4 Complete: 3D Mesh saved to {args.output_stl}")
