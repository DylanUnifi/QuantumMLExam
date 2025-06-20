
# train_qkernel.py
# Version: 5.0 – Kernel Quantum SVM Training amélioré avec log, checkpointing, visualisation, optimisation Optuna

import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import optuna
import joblib
from data_loader.utils import load_dataset_by_name
from utils.visual import plot_quantum_circuit
from utils.metrics import log_metrics
from utils.checkpoint import save_checkpoint
from utils.logger import init_logger, write_log
from models.quantum_kernel import torch_quantum_kernel, torch_qks_features

def objective(trial, X, y, method, device, kf):
    kernel_type = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
    C = trial.suggest_float("C", 1e-3, 1e3, log=True)
    gamma = trial.suggest_float("gamma", 1e-4, 1e1, log=True)

    scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if method == "fidelity":
            K_train = torch_quantum_kernel(torch.tensor(X_train, device=device))
            K_test = torch_quantum_kernel(torch.tensor(X_test, device=device))
            clf = SVC(kernel="precomputed", C=C)
            clf.fit(K_train.cpu().numpy(), y_train)
            score = clf.score(K_test.cpu().numpy(), y_test)
        else:
            X_train_feat = torch_qks_features(torch.tensor(X_train, device=device)).cpu().numpy()
            X_test_feat = torch_qks_features(torch.tensor(X_test, device=device)).cpu().numpy()
            clf = SVC(kernel=kernel_type, C=C, gamma=gamma)
            clf.fit(X_train_feat, y_train)
            score = clf.score(X_test_feat, y_test)

        scores.append(score)

    return np.mean(scores)

def run_train_qkernel(config):
    dataset = config["dataset"]
    method = config.get("method", "fidelity")
    n_splits = config.get("kfold", 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimize = config.get("optimize", False)
    experiment_name = config.get("experiment_name", "qkernel_exp")
    SAVE_DIR = os.path.join("checkpoints", "qkernel", experiment_name)
    os.makedirs(SAVE_DIR, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, "tensorboard"))
    log_path, log_file = init_logger(os.path.join(SAVE_DIR, "logs"))

    write_log(log_file, f"[QKERNEL] Method: {method}, Dataset: {dataset}, Device: {device}\n")

    loader, _ = load_dataset_by_name(dataset, batch_size=1024, selected_classes=config.get("selected_classes", [3, 8]), return_tensor_dataset=True)
    X, y = loader.dataset.tensors
    X, y = X.numpy().reshape(len(X), -1), y.numpy()

    if optimize:
        print("Starting hyperparameter optimization...")
        study = optuna.create_study(direction="maximize")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        study.optimize(lambda trial: objective(trial, X, y, method, device, kf), n_trials=25)
        best_params = study.best_params
        write_log(log_file, f"Best Optuna Parameters: {best_params}\n")
        joblib.dump(study, os.path.join(SAVE_DIR, "optuna_study.pkl"))

        kernel_type = best_params.get("kernel", "rbf")
        C = best_params["C"]
        gamma = best_params.get("gamma", "scale")
    else:
        kernel_type = config.get("kernel", "rbf")
        C = config.get("C", 1.0)
        gamma = config.get("gamma", "scale")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        write_log(log_file, f"Fold {fold + 1}/{n_splits}\n")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if method == "fidelity":
            K_train = torch_quantum_kernel(torch.tensor(X_train, device=device))
            K_test = torch_quantum_kernel(torch.tensor(X_test, device=device))
            clf = SVC(kernel="precomputed", C=C)
            clf.fit(K_train.cpu().numpy(), y_train)
            y_pred = clf.predict(K_test.cpu().numpy())
        elif method == "qks":
            X_train_feat = torch_qks_features(torch.tensor(X_train, device=device)).cpu().numpy()
            X_test_feat = torch_qks_features(torch.tensor(X_test, device=device)).cpu().numpy()
            clf = SVC(kernel=kernel_type, C=C, gamma=gamma)
            clf.fit(X_train_feat, y_train)
            y_pred = clf.predict(X_test_feat)
        else:
            raise ValueError("Unsupported kernel method")

        acc, f1, prec, recall = log_metrics(y_test, y_pred)
        write_log(log_file, f"Fold {fold} - Acc: {acc:.4f}, F1: {f1:.4f}, Prec: {prec:.4f}, Rec: {recall:.4f}\n")
        writer.add_scalar("Accuracy/Fold", acc, fold)
        writer.add_scalar("F1/Fold", f1, fold)
        writer.add_scalar("Precision/Fold", prec, fold)
        writer.add_scalar("Recall/Fold", recall, fold)

        save_checkpoint({
            "model_state_dict": clf,
            "scaler": scaler
        }, os.path.join(SAVE_DIR, f"qkernel_fold_{fold}.pt"))

    plot_quantum_circuit(method=method, filename=os.path.join(SAVE_DIR, "circuit.png"))
    log_file.close()
    writer.close()

if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_train_qkernel(config)
