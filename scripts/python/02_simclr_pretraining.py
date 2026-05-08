import torch
import torch.nn as nn
from torchvision import models

# SimCLR v1 Self-Supervised Pre-training
# Learns feature representations directly from unlabelled slices[cite: 67].

class SimCLR_Framework(nn.Module):
    """
    ResNet-50 backbone with modified first layer for single-channel CT data[cite: 84, 257].
    """
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet50(weights=None)
        # Modify for grayscale input [cite: 178]
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.fc = nn.Identity() # Feature representation h_i [cite: 265]
        
        # Projection head: MLP with two linear layers and ReLU [cite: 266]
        self.projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x_i, x_j):
        # Maximise agreement between positive pairs using NT-Xent loss [cite: 268]
        return self.projector(self.encoder(x_i)), self.projector(self.encoder(x_j))
