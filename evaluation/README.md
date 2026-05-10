# evaluation/

Quantitative and qualitative evaluation of segmentation quality and feature representations.

## Scripts

| File | Description |
|---|---|
| `metrics.py` | Dice-Sørensen coefficient and IoU (Jaccard Index) per specimen |
| `grad_cam.py` | Grad-CAM heatmaps overlaid on CT slices for encoder sanity checks |
| `umap_viz.py` | 2D UMAP projection of 2048-d ResNet-50 feature vectors with KDE overlay |

## Usage

```bash
# Run metrics on held-out test set
python evaluation/metrics.py

# Generate Grad-CAM heatmaps
python evaluation/grad_cam.py

# Plot UMAP of learned features
python evaluation/umap_viz.py
```

## Outputs

- Dice / IoU scores per specimen (Table in paper)
- Grad-CAM heatmap images (Fig. 2d)
- UMAP plot with KDE overlay (Fig. 2c)
- CloudCompare C2C distance statistics (Table 2)
