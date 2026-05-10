"""
evaluation/grad_cam.py
======================
Grad-CAM visualisation for SimCLR encoder sanity checks.

"Grad-CAM heatmaps provided spatially explicit, slice-level confirmation
of the UMAP findings. Heatmaps consistently localise strong activation
over fossilised skeletal material across both specimens […]. Activation
intensity tracks bone spatial extent and density, while matrix regions
with elevated radiodensity due to mineralogical inclusions or artefacts
do not elicit strong responses."
— manuscript §Contrastive pre-training encodes fossil-matrix structure

Used in the paper on two Cteniogenys sp. datasets as a visual sanity
check that the encoder captures biologically meaningful bone-matrix
structure rather than radiographic noise.

Reference
---------
Selvaraju, R. R. et al. (2017). Grad-CAM: Visual explanations from deep
networks via gradient-based localization. ICCV 2017, 618–626.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class GradCAM:
    """
    Grad-CAM implementation for the SimCLR ResNet-50 encoder.

    Registers forward and backward hooks on *target_layer* to capture
    feature maps and their gradients with respect to the class score.

    Parameters
    ----------
    model        : SimCLRv1 model (or any model with .encoder attribute)
    target_layer : name of the ResNet-50 layer to visualise
                   (default: "layer4" — manuscript target)
    """

    def __init__(self, model, target_layer: str = "layer4"):
        self.model = model
        self.target_layer = target_layer
        self._features: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        layer = dict(self.model.encoder.named_modules()).get(self.target_layer)
        if layer is None:
            raise ValueError(
                f"Target layer '{self.target_layer}' not found in encoder. "
                f"Available: {list(dict(self.model.encoder.named_modules()).keys())}"
            )
        self._hooks.append(
            layer.register_forward_hook(self._save_features)
        )
        self._hooks.append(
            layer.register_full_backward_hook(self._save_gradients)
        )

    def _save_features(self, module, input, output):
        self._features = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, img: torch.Tensor) -> np.ndarray:
        """
        Compute a Grad-CAM heatmap for a single CT slice.

        Parameters
        ----------
        img : (1, 1, H, W) float tensor — preprocessed CT slice

        Returns
        -------
        heatmap : (H, W) float array in [0, 1] — activation intensity map
        """
        self.model.eval()
        img.requires_grad_(True)

        # Forward pass
        h, z = self.model(img)
        # Use the L2-norm of feature vector as the scalar to back-propagate
        score = h.norm()
        self.model.zero_grad()
        score.backward()

        # Global average pool the gradients over spatial dims
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self._features).sum(dim=1, keepdim=True)   # (1, 1, H', W')
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=img.shape[-2:], mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def visualise_gradcam(
    model,
    ct_slices: list[np.ndarray],
    specimen_name: str = "",
    target_layer: str = "layer4",
    output_path: Optional[str | Path] = None,
    config_path: str | Path = "config/config.yaml",
) -> plt.Figure:
    """
    Generate a Grad-CAM panel for a list of CT slices.

    Replicates the sanity-check visualisation in manuscript Fig. 2d.

    Parameters
    ----------
    model        : trained SimCLRv1 model
    ct_slices    : list of (H, W) float arrays — representative slices
    specimen_name: label for plot title
    target_layer : ResNet-50 layer to hook (default "layer4")
    output_path  : save path for the figure (optional)

    Returns
    -------
    matplotlib Figure
    """
    device = next(model.parameters()).device
    gradcam = GradCAM(model, target_layer=target_layer)

    n = len(ct_slices)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, sl in enumerate(ct_slices):
        img_t = torch.tensor(sl, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        heatmap = gradcam(img_t)

        # Top row: original CT slice
        axes[0, i].imshow(sl, cmap="gray")
        axes[0, i].set_title(f"CT slice {i + 1}", fontsize=10)
        axes[0, i].axis("off")

        # Bottom row: heatmap overlay
        axes[1, i].imshow(sl, cmap="gray")
        axes[1, i].imshow(heatmap, cmap="jet", alpha=0.5)
        axes[1, i].set_title(f"Grad-CAM ({target_layer})", fontsize=10)
        axes[1, i].axis("off")

    fig.suptitle(f"Grad-CAM — {specimen_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    gradcam.remove_hooks()
    return fig
