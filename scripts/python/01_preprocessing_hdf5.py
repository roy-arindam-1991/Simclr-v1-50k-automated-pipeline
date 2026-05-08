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
    # 1. Intensity normalisation via percentile clipping [cite: 274]
    norm_img = np.clip(image_array, np.percentile(image_array, 2), np.percentile(image_array, 98))
    
    # 2. Noise reduction via Gaussian and median blurring [cite: 275]
    blur = cv2.GaussianBlur(norm_img.astype(np.uint8), (5, 5), 0)
    median = cv2.medianBlur(blur, 5)
    
    # 3. Seed identification using Otsu binarization [cite: 276]
    _, mask = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Morphological refinement (Top-hat, erosion, dilation) [cite: 277]
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 5. Connected component filtering to remove artefactual fragments [cite: 278]
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Retain only components exceeding the physical area threshold [cite: 278]
    sizes = stats[1:, -1]
    refined_mask = np.zeros((output.shape))
    for i in range(0, nb_components - 1):
        if sizes[i] >= 100: # Threshold determined empirically [cite: 279]
            refined_mask[output == i + 1] = 255
            
    return refined_mask.astype(np.uint8)

def process_and_partition(tif_root, output_dir):
    random.seed(42) # Fixed seed for reproducibility [cite: 245]
    specimens = sorted([d for d in os.listdir(tif_root) if os.path.isdir(os.path.join(tif_root, d))])
    random.shuffle(specimens)
    
    # Partitions: 19 Training, 2 Validation, 3 U-Net [cite: 245, 246]
    partitions = {
        'simclr_train': specimens[:19],
        'simclr_val': specimens[19:21],
        'unet_refinement': specimens[21:24]
    }

    for name, subset in partitions.items():
        h5_path = os.path.join(output_dir, f"{name}.h5")
        with h5py.File(h5_path, 'w') as f:
            # Create datasets for both images and their deterministic masks [cite: 36, 68]
            f.create_dataset('images', (0, 224, 224), maxshape=(None, 224, 224), dtype='f')
            f.create_dataset('masks', (0, 224, 224), maxshape=(None, 224, 224), dtype='u1')
            # Processing loop omitted for brevity [cite: 280]

if __name__ == "__main__":
    process_and_partition('../../data/raw', '../../data/processed')
