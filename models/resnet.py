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

def load_compatible_state(model: ResNet18LandUse, checkpoint) -> dict:
    """Load either this project's checkpoint format or the original ResNet state dict."""
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if "conv1.weight" in state:
        prefixes = {"conv1.": "features.0.", "bn1.": "features.1.", "layer1.": "features.4.", "layer2.": "features.5.", "layer3.": "features.6.", "layer4.": "features.7.", "fc.": "classifier."}
        state = {next((new + key[len(old):] for old, new in prefixes.items() if key.startswith(old)), key): value for key, value in state.items()}
    model.load_state_dict(state)
    return checkpoint if isinstance(checkpoint, dict) else {}
