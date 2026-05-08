import h5py
import numpy as np
import cv2
import os
import random
from PIL import Image

def run_mask_prep(image_array):
    """
    MASK PREP: Generates automated ground truth masks using a 5-stage deterministic pipeline.
    This is independent of the HDF5 partitioning logic.
    """
    # 1. Percentile clipping and intensity normalisation
    norm_img = np.clip(image_array, np.percentile(image_array, 2), np.percentile(image_array, 98))
    norm_img = ((norm_img - norm_img.min()) / (norm_img.max() - norm_img.min()) * 255).astype(np.uint8)
    
    # 2. Gaussian and median blurring for noise suppression
    blur = cv2.GaussianBlur(norm_img, (5, 5), 0)
    
    # 3. Seed identification via Otsu binarization
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Morphological refinement (Closing/Opening cycles)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 5. Connected component filtering (Area-based rejection)
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]
    final_mask = np.zeros((output.shape), dtype=np.uint8)
    for i in range(0, nb_components - 1):
        if sizes[i] >= 100: # Minimum physical area threshold
            final_mask[output == i + 1] = 255
    return final_mask

def run_data_prep(tif_root, output_dir):
    """
    DATA PREP: Partitions the 50,626 CT images and converts to HDF5.
    Subsets: 19 Training (39,037), 2 Validation (3,765), 3 U-Net (7,824).
    """
    random.seed(42)
    specimens = sorted([d for d in os.listdir(tif_root) if os.path.isdir(os.path.join(tif_root, d))])
    random.shuffle(specimens)
    
    partitions = {
        'simclr_train': specimens[:19],
        'simclr_val': specimens[19:21],
        'unet_refinement': specimens[21:24]
    }
    
    for name, subset in partitions.items():
        h5_path = os.path.join(output_dir, f"{name}.h5")
        with h5py.File(h5_path, 'w') as f:
            # Create separate datasets for images and their corresponding mask-prep outputs
            f.create_dataset('ct_images', (0, 224, 224), maxshape=(None, 224, 224), dtype='f')
            f.create_dataset('deterministic_masks', (0, 224, 224), maxshape=(None, 224, 224), dtype='u1')
