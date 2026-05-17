# scripts/

SLURM batch scripts to run each stage of the pipeline on BlueBear HPC (University of Birmingham).  
Scripts must be submitted in order — each stage depends on outputs from the previous.

## Scripts

| File | Stage | Description |
|---|---|---|
| `01_preprocess.sh` | 1 | Data preprocessing: resize, normalise, TIFF → HDF5 |
| `02_train_simclr.sh` | 2 | SimCLR v1 contrastive pre-training on A100 GPU |
| `03_generate_masks.sh` | 3 | Rule-based deterministic coarse mask generation |
| `04_train_unet.sh` | 4 | U-Net training via SimCLR knowledge fusion |
| `05_run_inference.sh` | 5 | End-to-end inference on new CT specimens |

## Usage

```bash
bash scripts/01_preprocess.sh
bash scripts/02_train_simclr.sh
bash scripts/03_generate_masks.sh
bash scripts/04_train_unet.sh
bash scripts/05_run_inference.sh
```

## HPC Details

- **Cluster**: BlueBear HPC, University of Birmingham
- **GPU**: NVIDIA A100
- **Scheduler**: SLURM
- All scripts activate the correct conda environment before execution
