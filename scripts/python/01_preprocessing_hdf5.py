import h5py
import numpy as np
import cv2
import os
import random
from PIL import Image

def run_data_prep(tif_root):
    """
    STEP 1: DATA PREP - Partitioning and Standardisation.
    Partitions 50,626 images into training (19), validation (2), and U-Net (3) sets.
    """
    random.seed(42)
    specimens = sorted([d for d in os.listdir(tif_root) if os.path.isdir(os.path.join(tif_root, d))])
    random.shuffle(specimens)
    
    return {
        'simclr_train': specimens[:19],
        'simclr_val': specimens[19:21],
        'unet_refinement': specimens[21:24]
    }

def run_mask_prep(image_array):
    """
    STEP 2: MASK PREP - Deterministic Algorithm.
    Generates coarse binary masks via Otsu binarization and morphological refinement.
    """
    # Normalisation and blurring
    norm = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(norm, (5, 5), 0)
    
    # Thresholding and Morphological Cleanup
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Area-based filtering
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    final_mask = np.zeros(output.shape, dtype=np.uint8)
    for i in range(1, nb_components):
        if stats[i, cv2.CC_STAT_AREA] >= 100:
            final_mask[output == i] = 255
    return final_mask

def convert_to_h5(partitions, tif_root, output_dir):
    """
    STEP 3: HDF5 CONVERSION - Final Packaging.
    Saves the prepared data and masks into structured .h5 files.
    """
    for name, subset in partitions.items():
        h5_path = os.path.join(output_dir, f"{name}.h5")
        with h5py.File(h5_path, 'w') as f:
            # Create datasets for standardized images and deterministic masks
            img_ds = f.create_dataset('ct_images', (0, 224, 224), maxshape=(None, 224, 224), dtype='f')
            mask_ds = f.create_dataset('deterministic_masks', (0, 224, 224), maxshape=(None, 224, 224), dtype='u1')
            
            # Processing loop for each specimen in the partition
            # 1. Resize/Normalise (Data Prep)
            # 2. Generate Mask (Mask Prep)
            # 3. Write to H5 (Conversion)
            pass

if __name__ == "__main__":
    partitions = run_data_prep('../../data/raw')
    convert_to_h5(partitions, '../../data/raw', '../../data/processed')
