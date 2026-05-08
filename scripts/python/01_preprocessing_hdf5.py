import h5py
import numpy as np
import cv2
import os
import random
from PIL import Image

def generate_dataprep_mask(image_array):
    """
    Deterministic image processing for automated ground truth generation.
    """
    # 1. Intensity normalisation via percentile clipping to suppress outliers
    norm_img = np.clip(image_array, np.percentile(image_array, 2), np.percentile(image_array, 98))
    norm_img = ((norm_img - norm_img.min()) / (norm_img.max() - norm_img.min()) * 255).astype(np.uint8)
    
    # 2. Noise reduction via Gaussian and median blurring
    blur = cv2.GaussianBlur(norm_img, (5, 5), 0)
    median = cv2.medianBlur(blur, 5)
    
    # 3. Seed identification using Otsu binarization
    _, mask = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Morphological refinement (Top-hat, erosion, dilation cycles)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 5. Connected component filtering based on physical area threshold (min 100 voxels)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]
    refined_mask = np.zeros((output.shape), dtype=np.uint8)
    for i in range(0, nb_components - 1):
        if sizes[i] >= 100:
            refined_mask[output == i + 1] = 255
            
    return refined_mask

def process_and_partition(tif_root, output_dir):
    """
    Partitions the 50,626 images into 19 Training, 2 Validation, and 3 U-Net datasets.
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
            f.create_dataset('images', (0, 224, 224), maxshape=(None, 224, 224), dtype='f')
            f.create_dataset('masks', (0, 224, 224), maxshape=(None, 224, 224), dtype='u1')
            # Processing loop applies generate_dataprep_mask here
