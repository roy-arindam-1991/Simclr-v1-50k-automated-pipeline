#!/usr/bin/env python3
import h5py
import numpy as np
from PIL import Image

# ----------------------------
# Paths
# ----------------------------
OLD_H5 = "/rds/projects/b/butlerry-deepctseg/40K_pipeline/data_prep_results/split_h5_files/test/test_data.h5"
NEW_H5 = "/rds/projects/b/butlerry-deepctseg/40K_pipeline/data_prep_results/40k_images_fixed.h5"

TARGET_SIZE = (512, 512)

# Open input HDF5 and count total images
with h5py.File(OLD_H5, 'r') as f_old:
    image_keys = list(f_old['images'].keys())
    total_images = len(image_keys)
    print(f"Total images: {total_images}")

    # Create output HDF5
    with h5py.File(NEW_H5, 'w') as f_new:
        dset_new = f_new.create_dataset(
            'images',
            shape=(total_images, TARGET_SIZE[0], TARGET_SIZE[1]),
            dtype=np.uint8
        )

        # Process and resize each image
        for idx, key in enumerate(image_keys):
            img = f_old['images'][key][()]  # Load dataset as numpy array
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize(TARGET_SIZE, Image.BILINEAR)
            dset_new[idx] = np.array(pil_img, dtype=np.uint8)

            if idx % 100 == 0:
                print(f"Processed {idx}/{total_images} images")

print(f"All images saved to {NEW_H5} with size {TARGET_SIZE}")