# main.py
# Version: 6.0 – Ajout d'une intégration unifiée avec Weights & Biases (wandb)

import argparse
import numpy as np
import torch
import random
import sys
import wandb
from train_hybrid_qcnn_svm import run_train_hybrid_qcnn_svm

from configs.config import load_config
from train_classical import run_train_classical
from train_cnn import run_train_cnn
from train_hybrid_qcnn import run_train_hybrid_qcnn
from train_svm import run_train_svm
from train_quantum_mlp import run_train_quantum_mlp


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Unified training script for classical, CNN, hybrid QCNN, Quantum MLP,"
                                                 "hybrid_qcnn_svm, SVM, and  models.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name: classical_mlp, cnn, svm, quantum_mlp, qcnn, svm_qkernel")
    parser.add_argument("--config", type=str, default="configs/config_train_quantum_mlp_fashion.yaml",
                        help="Path to configuration YAML file")
    parser.add_argument("--optimize", action="store_true",
                        help="Enable Optuna hyperparameter optimization (if supported by the model)")
    parser.add_argument("--project", type=str, default="qml_project",
                        help="Weights & Biases project name")

    args = parser.parse_args()
    available_models = {"classical_mlp", "cnn", "svm", "quantum_mlp", "qcnn", "svm_qkernel"}

    if args.model not in available_models:
        print(f"[ERROR] Unsupported model '{args.model}'. Choose from: {', '.join(available_models)}.")
        sys.exit(1)

    print(f"[INFO] Loading config: {args.config}")
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    print(f"[INFO] Starting training for model: {args.model.upper()}")
    print(f"[INFO] Hyperparameter optimization: {'ENABLED' if args.optimize else 'DISABLED'}")

    dataset_name = config["dataset"]["name"]
    base_exp_name = config.get("experiment_name", "default_exp")
    EXPERIMENT_NAME = f"{dataset_name}_{base_exp_name}"
    SAVE_DIR = os.path.join("engine/checkpoints", "classical", EXPERIMENT_NAME)
    CHECKPOINT_DIR = os.path.join(SAVE_DIR, "folds")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    wandb.init(
        project=args.project,
        name=f"{args.model}_{wandb.util.generate_id()}_{EXPERIMENT_NAME}",
        config=config,
        tags=[config["dataset"]["name"], args.model]
    )

    if args.model == "classical_mlp":
        run_train_classical_mlp(config)
        wandb.finish()
    elif args.model == "cnn":
        run_train_cnn(config)
        wandb.finish()
    elif args.model == "qcnn":
        run_train_qcnn(config)
        wandb.finish()
    elif args.model == "quantum_mlp":
        run_train_quantum_mlp(config)
        wandb.finish()
    elif args.model == "svm_qkernel":
        run_train_svm_qkernel(config)
        wandb.finish()
    elif args.model == "svm":
        config.setdefault("svm", {})
        config["svm"]["optimize"] = args.optimize
        run_train_svm(config)
        wandb.finish()
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    print(f"[INFO] Training for {args.model.upper()} completed successfully.")
    wandb.finish()


if __name__ == "__main__":
    main()
