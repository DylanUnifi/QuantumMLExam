# main.py (Extrait pertinent pour Quantum Kernel)
# Version 3.2 – Intégration avec le dossier `data_loader`

import argparse
import numpy as np
import torch
import random
from config import load_config
from data_loader.utils import load_dataset  # Mise à jour ici
from train_qkernel import train_qkernel_model


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qkernel", help="Model name")
    parser.add_argument("--optimize", action="store_true", help="Enable Optuna optimization")
    args = parser.parse_args()

    config = load_config()
    seed = config.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Chargement du dataset via data_loader/utils.py
    dataset_name = config.get("dataset", "MNIST")
    selected_classes = config.get("selected_classes", [0, 1])
    X, y = load_dataset(config=config, dataset=dataset_name, classes=selected_classes)

    if args.model == "qkernel":
        method = config.get("qkernel_method", "fidelity")
        n_splits = config.get("n_splits", 5)
        train_qkernel_model(X, y, method=method, n_splits=n_splits, device=device, optimize=args.optimize)
    else:
        raise ValueError(f"Unsupported model: {args.model}")


if __name__ == "__main__":
    main()
