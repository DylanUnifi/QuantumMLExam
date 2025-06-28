# main.py
# Version: 4.0 – Unification de tous les modules d'entraînement

import argparse
import numpy as np
import torch
import random

from configs.config import load_config

from train_classical import run_train_classical
from train_cnn import run_train_cnn
from train_hybrid_qcnn import run_train_hybrid_qcnn
from train_qkernel import train_qkernel_model
from train_svm import run_train_svm


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name: classical, cnn, hybrid_qcnn, svm, qkernel")
    parser.add_argument("--config", type=str, default="config_train_quantum_mlp.yaml", help="Path to configuration YAML")
    parser.add_argument("--optimize", action="store_true", help="Enable Optuna optimization (if supported)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    if args.model == "classical":
        run_train_classical(config)
    elif args.model == "cnn":
        run_train_cnn(config)
    elif args.model == "hybrid_qcnn":
        run_train_hybrid_qcnn(config)
    elif args.model == "svm":
        config["svm"]["optimize"] = args.optimize
        run_train_svm(config)
    elif args.model == "qkernel":
        config["optimize"] = args.optimize
        train_qkernel_model(config)
    else:
        raise ValueError(f"Unsupported model: {args.model}")


if __name__ == "__main__":
    main()
