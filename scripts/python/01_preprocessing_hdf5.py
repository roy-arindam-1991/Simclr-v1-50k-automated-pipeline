#!/usr/bin/env python3
import random
import argparse
from pathlib import Path
import logging
import sys
import os
import shutil
import pandas as pd  # Required for Excel generation

# =============================
# ARGUMENT PARSER
# =============================
parser = argparse.ArgumentParser(description="Split subfolders into train/val/test sets and count slices.")
parser.add_argument('--input_dir', type=str, required=True, help='Root directory containing subfolders.')
parser.add_argument('--output_dir', type=str, required=True, help='Folder where split folders will be created.')
parser.add_argument('--train_ratio', type=float, default=0.8)
parser.add_argument('--val_ratio', type=float, default=0.1)
parser.add_argument('--test_ratio', type=float, default=0.1)
parser.add_argument('--mode', type=str, choices=['symlink', 'copy', 'move'], default='symlink')
args = parser.parse_args()

input_root = Path(args.input_dir)
output_root = Path(args.output_dir)

# =============================
# LOGGING & SETUP
# =============================
output_root.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(output_root / "split_folders.log"), logging.StreamHandler(sys.stdout)]
)

# =============================
# DIRECTORY DISCOVERY
# =============================
subfolders = [f for f in input_root.iterdir() if f.is_dir()]
n_folders = len(subfolders)

if n_folders == 0:
    logging.error(f"No subdirectories found in {input_root}")
    sys.exit(1)

logging.info(f"Total subfolders found: {n_folders}")

# =============================
# SPLIT LOGIC
# =============================
random.seed(42)
random.shuffle(subfolders)

n_train = int(n_folders * args.train_ratio)
n_val   = int(n_folders * args.val_ratio)

splits = {
    "train": subfolders[:n_train],
    "val":   subfolders[n_train:n_train+n_val],
    "test":  subfolders[n_train+n_val:]
}

# Counters for total slices (tif files)
split_stats = {"train": 0, "val": 0, "test": 0}

# =============================
# DISTRIBUTION LOGIC
# =============================
for split_name, folders in splits.items():
    split_path = output_root / split_name
    split_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Processing {split_name} split ({len(folders)} folders)...")
    
    for folder in folders:
        # Count TIFF files in this specific folder
        tifs_in_folder = list(folder.rglob("*.tif")) + list(folder.rglob("*.tiff"))
        split_stats[split_name] += len(tifs_in_folder)
        
        dst = split_path / folder.name
        
        try:
            if args.mode == 'symlink':
                if dst.exists(): 
                    if dst.is_symlink() or dst.is_file(): dst.unlink()
                    else: shutil.rmtree(dst)
                os.symlink(folder.absolute(), dst, target_is_directory=True)
            elif args.mode == 'copy':
                shutil.copytree(folder, dst, dirs_exist_ok=True)
            elif args.mode == 'move':
                shutil.move(str(folder), str(dst))
        except Exception as e:
            logging.error(f"Failed to process folder {folder.name}: {e}")

# =============================
# SAVE SUMMARIES
# =============================
# 1. Text Summary
summary_file = output_root / "split_summary.txt"
with open(summary_file, "w") as f:
    f.write("=== Data Split Summary ===\n")
    f.write(f"Input Directory: {args.input_dir}\n\n")
    for name, count in split_stats.items():
        line = f"{name.upper()}: {len(splits[name])} folders, {count} total slices (TIFFs)\n"
        f.write(line)
        print(line, end="")

# 2. Excel Summary (NEW)
excel_file = output_root / "specimen_splits.xlsx"
logging.info(f"Creating Excel summary at {excel_file}...")

# Create a dictionary of the folder names for each split
# We use pd.Series to handle columns of different lengths automatically
data = {
    "Train": pd.Series([f.name for f in splits["train"]]),
    "Validation": pd.Series([f.name for f in splits["val"]]),
    "Test": pd.Series([f.name for f in splits["test"]])
}

df = pd.DataFrame(data)
df.to_excel(excel_file, index=False)

logging.info(f"Summary saved to {summary_file} and {excel_file}")