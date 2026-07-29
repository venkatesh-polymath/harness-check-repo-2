"""
Models: ResNet-18 backbone, SimCLR projection head, linear probe.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


def get_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """ResNet-18 adapted for CIFAR-32×32 (first conv 3×3 s1, no maxpool)."""
    model = tvm.resnet18(weights=None)
    # Adapt for 32×32 CIFAR images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


class SimCLRBackbone(nn.Module):
    """ResNet-18 backbone + projection head for SimCLR pretraining."""

    def __init__(self, proj_dim: int = 128):
        super().__init__()
        # Encoder (ResNet-18 without fc)
        base = tvm.resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        self.enc_dim = base.fc.in_features  # 512
        base.fc = nn.Identity()
        self.encoder = base

        # Projection head (2-layer MLP)
        self.projector = nn.Sequential(
            nn.Linear(self.enc_dim, self.enc_dim),
            nn.ReLU(),
            nn.Linear(self.enc_dim, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z

    def encode(self, x):
        return self.encoder(x)


class LinearProbe(nn.Module):
    """Single linear layer on top of a frozen backbone."""

    def __init__(self, feat_dim: int, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy loss (SimCLR)."""
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2N × D
    z = F.normalize(z, dim=1)

    # Cosine similarity matrix (2N × 2N)
    sim = torch.matmul(z, z.T) / temperature

    # Mask out diagonal (self-similarity)
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)

    # Positive pairs: (i, i+N) and (i+N, i)
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0, N, device=z.device),
    ])  # 2N

    loss = F.cross_entropy(sim, labels)
    return loss
