# utils/data/fashion.py
# Version: 1.0

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def load_fashion_mnist(batch_size=64, selected_classes=[3, 8], root='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    train_indices = [i for i, t in enumerate(train_set.targets) if t in selected_classes]
    test_indices = [i for i, t in enumerate(test_set.targets) if t in selected_classes]

    train_subset = Subset(train_set, train_indices)
    test_subset = Subset(test_set, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size)

    return train_loader, test_loader
