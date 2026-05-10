"""
simclr/model.py
===============
SimCLR v1 model architecture.

Architecture (per manuscript):
  - Base encoder : ResNet-50 (He et al., 2016) — output feature dim = 2048
  - Projection head : MLP with two linear layers separated by ReLU activation
    (as in Chen et al., 2020). The projection head maps encoder features to a
    latent space where the NT-Xent contrastive loss is applied. This prevents
    loss of structural information that would occur if the contrastive objective
    were applied directly to the 2048-d feature vectors used for downstream
    segmentation.

Reference
---------
Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).
A simple framework for contrastive learning of visual representations.
ICML 2020, 1597–1607.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head with ReLU activation.

    Maps encoder features h ∈ R^{feature_dim} to a lower-dimensional
    latent space z ∈ R^{projection_dim} for contrastive loss computation.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        projection_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class SimCLRv1(nn.Module):
    """
    SimCLR v1 model: ResNet-50 encoder + projection head.

    During pre-training the projection head is active.
    For downstream segmentation, call `encoder_only()` to obtain the
    bare ResNet-50 feature extractor whose weights initialise the U-Net.

    Parameters
    ----------
    feature_dim     : encoder output dimension (2048 for ResNet-50)
    hidden_dim      : projection head hidden dimension
    projection_dim  : projection head output dimension (latent space)
    pretrained      : whether to initialise the encoder with ImageNet weights
                      (manuscript uses domain-specific pre-training; set False
                       to train from scratch on palaeontological CT data)
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        projection_dim: int = 128,
        pretrained: bool = False,
    ):
        super().__init__()

        # Base encoder — ResNet-50 without the classification head
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Remove the global average pool and FC so we get (N, 2048, H', W')
        # Then add our own global average pool → (N, 2048)
        self.encoder = nn.Sequential(
            *list(backbone.children())[:-1]   # drop avgpool + fc
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection_head = ProjectionHead(feature_dim, hidden_dim, projection_dim)

        self.feature_dim = feature_dim

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for one view of a positive pair.

        Parameters
        ----------
        x : (N, C, H, W) — batch of augmented CT slices

        Returns
        -------
        h : (N, feature_dim)     encoder features (used for UMAP, Grad-CAM)
        z : (N, projection_dim)  projected features (used for NT-Xent loss)
        """
        feat = self.encoder(x)          # (N, 2048, 7, 7)
        h = self.pool(feat).flatten(1)  # (N, 2048)
        z = self.projection_head(h)     # (N, projection_dim)
        return h, z

    def get_encoder(self) -> nn.Module:
        """Return the bare ResNet-50 encoder for U-Net weight initialisation."""
        return self.encoder

    @classmethod
    def from_checkpoint(
        cls, ckpt_path: str, map_location: str | torch.device = "cpu"
    ) -> "SimCLRv1":
        """Load a SimCLRv1 model from a saved checkpoint."""
        ckpt = torch.load(ckpt_path, map_location=map_location)
        hparams = ckpt.get("hparams", {})
        model = cls(
            feature_dim=hparams.get("feature_dim", 2048),
            hidden_dim=hparams.get("projection_hidden_dim", 512),
            projection_dim=hparams.get("projection_dim", 128),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model
