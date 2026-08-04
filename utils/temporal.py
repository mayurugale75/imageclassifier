"""Embedding-based temporal change detection and patch heatmaps."""
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

def _patches(image, grid_size):
    image = image.convert("RGB"); width, height = image.size
    xs, ys = np.linspace(0, width, grid_size + 1, dtype=int), np.linspace(0, height, grid_size + 1, dtype=int)
    return [image.crop((xs[x], ys[y], xs[x+1], ys[y+1])) for y in range(grid_size) for x in range(grid_size)]
@torch.inference_mode()
def compare_images(model, before, after, transform, device, grid_size=8):
    model.eval()
    full = torch.stack([transform(before.convert("RGB")), transform(after.convert("RGB"))]).to(device)
    embeddings = F.normalize(model.encode(full), dim=1)
    similarity = float((embeddings[0] * embeddings[1]).sum().cpu())
    b = torch.stack([transform(p) for p in _patches(before, grid_size)]).to(device)
    a = torch.stack([transform(p) for p in _patches(after, grid_size)]).to(device)
    change = 1 - (F.normalize(model.encode(b), dim=1) * F.normalize(model.encode(a), dim=1)).sum(1)
    heatmap = change.reshape(grid_size, grid_size).clamp(0, 2).cpu()
    return similarity, TF.resize(heatmap[None], [after.height, after.width], antialias=True)[0].numpy()
