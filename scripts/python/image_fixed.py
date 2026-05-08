import h5py
import numpy as np
from PIL import Image
import argparse

def resize_h5_dataset(input_h5, output_h5, target_size=(512, 512)):
    with h5py.File(input_h5, 'r') as f_old:
        image_keys = list(f_old['images'].keys())
        total_images = len(image_keys)
        with h5py.File(output_h5, 'w') as f_new:
            dset_new = f_new.create_dataset('images', shape=(total_images, target_size[0], target_size[1]), dtype=np.uint8)
            for idx, key in enumerate(image_keys):
                img = f_old['images'][key][()]
                pil_img = Image.fromarray(img).resize(target_size, Image.BILINEAR)
                dset_new[idx] = np.array(pil_img, dtype=np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize HDF5 image datasets.")
    parser.add_argument("--input", required=True, help="Path to input H5 file")
    parser.add_argument("--output", required=True, help="Path to save processed H5 file")
    args = parser.parse_args()
    resize_h5_dataset(args.input, args.output)
