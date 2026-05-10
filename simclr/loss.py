"""
simclr/loss.py
==============
NT-Xent (Normalised Temperature-scaled Cross-Entropy) loss for SimCLR v1.

"The model training was assessed using the Normalized Temperature-scaled
Cross Entropy (NT-Xent) loss, which encourages the network to maximize the
agreement between the positive pair in the latent space while simultaneously
pushing them away from all other images in the training batch."
— manuscript §SimCLR pre-training

Reference
---------
Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).
A simple framework for contrastive learning of visual representations. ICML.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent loss for a batch of N positive pairs.

    For a batch of size N, the loss is computed over 2N representations
    (xi and xj for each sample). Each representation is attracted toward
    its positive pair and repelled from the 2(N−1) negatives.

    Parameters
    ----------
    temperature : τ — scales the similarity logits. Lower τ makes the
                  distribution sharper. Manuscript uses τ = 0.07.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss.

        Parameters
        ----------
        zi : (N, D) — projected features for the first view of each sample
        zj : (N, D) — projected features for the second view of each sample

        Returns
        -------
        Scalar loss tensor
        """
        N = zi.shape[0]
        device = zi.device

        # L2-normalise projections onto the unit hypersphere
        zi = F.normalize(zi, dim=1)
        zj = F.normalize(zj, dim=1)

        # Concatenate: (2N, D)
        z = torch.cat([zi, zj], dim=0)

        # Pairwise cosine similarity matrix: (2N, 2N)
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out self-similarity on the diagonal
        mask = torch.eye(2 * N, device=device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive pair indices:
        #   for row i (from zi), positive is row i+N (from zj)
        #   for row i+N (from zj), positive is row i (from zi)
        labels = torch.cat([
            torch.arange(N, 2 * N, device=device),   # positives for zi rows
            torch.arange(0, N,     device=device),   # positives for zj rows
        ])

        loss = F.cross_entropy(sim, labels)
        return loss


def contrastive_accuracy(zi: torch.Tensor, zj: torch.Tensor,
                         temperature: float = 0.07) -> float:
    """
    Top-1 contrastive accuracy: fraction of samples for which the true positive
    has the highest similarity among all negatives in the batch.

    Used in the manuscript to report training / validation accuracy converging
    to 93.89% / 93.66% over 250 epochs.
    """
    N = zi.shape[0]
    device = zi.device

    zi = F.normalize(zi, dim=1)
    zj = F.normalize(zj, dim=1)
    z = torch.cat([zi, zj], dim=0)
    sim = torch.mm(z, z.T) / temperature
    mask = torch.eye(2 * N, device=device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))

    labels = torch.cat([
        torch.arange(N, 2 * N, device=device),
        torch.arange(0, N,     device=device),
    ])
    preds = sim.argmax(dim=1)
    return (preds == labels).float().mean().item()
