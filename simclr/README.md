# simclr/

Self-supervised contrastive pre-training using SimCLR v1 on unlabelled fossil CT slices.

## Architecture

- **Encoder**: ResNet-50 backbone producing 2048-d feature vectors
- **Projection head**: 2-layer MLP (2048 → 512 → 128)
- **Loss**: NT-Xent (Normalised Temperature-scaled Cross-Entropy)

## Scripts

| File | Description |
|---|---|
| `model.py` | ResNet-50 encoder + 2-layer MLP projection head |
| `augmentations.py` | Domain-specific stochastic augmentation pipeline |
| `loss.py` | NT-Xent loss implementation |
| `train.py` | Training loop: 250 epochs, batch size 64, LR 3e-4 |
| `run_simclr.sh` | SLURM job script: A100 GPU, env activation, train.py entry |

## Training

```bash
bash scripts/02_train_simclr.sh
```

## Augmentation Strategy

Augmentations are designed for CT domain specificity:
- Random resized crop
- Horizontal/vertical flip
- Gaussian blur
- Intensity jitter (adapted for greyscale CT slices)
