import h5py
import numpy as np
import cv2
import os
from PIL import Image

# Preprocessing and Normalisation
# Standardises raw CT volumes into 224x224 px HDF5 containers[cite: 243].

def process_volume(tif_dir, output_h5):
    """
    Converts TIF slices to HDF5 to mitigate I/O bottlenecks and directory latency[cite: 248, 249].
    """
    files = sorted([f for f in os.listdir(tif_dir) if f.endswith('.tif')])
    with h5py.File(output_h5, 'w') as f:
        # Create structured 3D volume supporting data chunking [cite: 250]
        dataset = f.create_dataset('ct_slices', (len(files), 224, 224), dtype='f')
        for i, file in enumerate(files):
            img = Image.open(os.path.join(tif_dir, file)).resize((224, 224))
            img_array = np.array(img).astype(np.float32)
            
            # Normalisation [0,1] and z-standardisation (mu=0.5, sigma=0.5) [cite: 244]
            img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array) + 1e-6)
            img_array = (img_array - 0.5) / 0.5
            dataset[i] = img_array
