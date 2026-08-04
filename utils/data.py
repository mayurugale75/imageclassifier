"""Reproducible EuroSAT data loading."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import EuroSAT

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
def train_transform():
    return transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
def eval_transform():
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def make_loaders(data_dir, batch_size, val_fraction, seed, workers=0):
    train_base, eval_base = EuroSAT(root=data_dir, transform=train_transform(), download=False), EuroSAT(root=data_dir, transform=eval_transform(), download=False)
    indices = torch.randperm(len(train_base), generator=torch.Generator().manual_seed(seed)).tolist()
    split = int(len(indices) * (1 - val_fraction))
    args = {"batch_size": batch_size, "num_workers": workers, "pin_memory": torch.cuda.is_available()}
    return DataLoader(Subset(train_base, indices[:split]), shuffle=True, **args), DataLoader(Subset(eval_base, indices[split:]), shuffle=False, **args), train_base.classes
