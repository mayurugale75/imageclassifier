"""ResNet-18 classifier with access to penultimate-layer embeddings."""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models

class ResNet18LandUse(nn.Module):
    embedding_dim = 512
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(self.embedding_dim, num_classes)
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.features(images), 1)
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(images))
