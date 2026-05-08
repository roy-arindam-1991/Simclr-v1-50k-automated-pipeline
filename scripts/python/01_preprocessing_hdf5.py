import h5py
import numpy as np
# Phase 0: Preprocessing and Normalisation
# Standardises raw CT volumes into optimized HDF5 containers.

def preprocess_ct_slice(img):
    # Resize to 224x224 px for ResNet-50 compatibility
    # Normalise intensity to [0,1] and apply z-standardisation (mu=0.5, sigma=0.5)
    pass

def convert_to_h5(tif_path, h5_output):
    # Consolidated TIFF slices into individual self-describing HDF5 containers
    # This mitigates I/O bottlenecks and directory latency during training
    pass
