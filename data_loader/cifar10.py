# data_loader/cifar10.py
# Version 2.1 – Support DataLoader, comme Fashion-MNIST

from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Subset

def load_cifar10(batch_size=64, binary_classes=(3, 5), grayscale=True, root='./data'):
    transform = transforms.Compose([
        transforms.Grayscale() if grayscale else transforms.Lambda(lambda x: x),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if grayscale else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    train_indices = [i for i, t in enumerate(train_set.targets) if t in binary_classes]
    test_indices = [i for i, t in enumerate(test_set.targets) if t in binary_classes]

    def relabel(subset):
        binary_targets = [1 if train_set.targets[i] == binary_classes[1] else 0 for i in subset.indices]
        for idx, label in zip(subset.indices, binary_targets):
            train_set.targets[idx] = label
        return subset

    train_subset = Subset(train_set, train_indices)
    test_subset = Subset(test_set, test_indices)

    train_loader = DataLoader(relabel(train_subset), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(relabel(test_subset), batch_size=batch_size)

    return train_loader, test_loader
