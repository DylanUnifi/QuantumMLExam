# utils/data_loader/utils.py
# Interface commune de chargement de datasets

from data_loader.fashion import load_fashion_mnist
from data_loader.cifar10 import load_cifar10
from data_loader.mnist import load_mnist
from data_loader.svhn import load_svhn
# Ajouter les autres loaders SVHN, MNIST, etc. ici

SUPPORTED_DATASETS = {
    'fashion_mnist': load_fashion_mnist,
    'cifar10': load_cifar10,
    'mnist': load_mnist,
    'svhn': load_svhn,
}

def load_dataset_by_name(name, **kwargs):
    name = name.lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Dataset '{name}' non supporté. Disponibles: {list(SUPPORTED_DATASETS.keys())}")
    return SUPPORTED_DATASETS[name](**kwargs)
