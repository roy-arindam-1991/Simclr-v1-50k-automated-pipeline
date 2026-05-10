# unet/

Modified U-Net for fine segmentation of fossil CT slices, initialised with SimCLR pre-trained weights via knowledge fusion.

## Architecture

- **Encoder**: Deep ResNet-50 backbone e1–e4 (64 → 2048 channels), initialised from SimCLR checkpoint
- **Bottleneck**: 7×7 spatial resolution
- **Decoder**: Skip connections from encoder stages
- **Loss**: Composite Weighted Cross-Entropy + Dice loss

## Scripts

| File | Description |
|---|---|
| `model.py` | Full U-Net definition: encoder, bottleneck, decoder, skip connections |
| `loss.py` | Composite Weighted Cross-Entropy + Dice loss |
| `train.py` | 500 epochs, AdamW optimiser, OneCycleLR scheduler, best-checkpoint saving |
| `run_unet.sh` | SLURM job: loads SimCLR weights, runs train + evaluate |

## Training

```bash
bash scripts/04_train_unet.sh
```

## Notes

- SimCLR encoder weights must be transferred before training — see `knowledge_fusion/`
- Best checkpoint is saved automatically based on validation Dice score
