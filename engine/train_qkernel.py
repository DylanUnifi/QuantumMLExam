# train_qkernel.py
# Version 4.0 – Intégré dans main.py et piloté via config.yaml

import torch
import numpy as np
import optuna
import joblib
import os
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from models.quantum_kernel import torch_quantum_kernel, torch_qks_features
from utils.visual import plot_quantum_circuit
from utils.metrics import log_metrics
from utils.checkpoint import save_checkpoint
from config import load_config

def objective(trial, X, y, method, n_splits, device):
    kernel_type = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])
    C = trial.suggest_loguniform("C", 1e-3, 1e3)
    gamma = trial.suggest_loguniform("gamma", 1e-4, 1e1)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
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

def train_qkernel_model(X, y, config):
    method = config.get("method", "fidelity")
    n_splits = config.get("n_splits", 5)
    device = torch.device(config.get("device", "cpu"))
    optimize = config.get("optimize", False)
    log_dir = os.path.join("runs", f"qkernel_{method}")
    writer = SummaryWriter(log_dir)

    if optimize:
        print("\n[Optuna] Starting hyperparameter optimization...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X, y, method, n_splits, device), n_trials=20)
        best_params = study.best_params
        print(f"Best parameters: {best_params}")
        joblib.dump(study, os.path.join("checkpoints", f"optuna_qkernel_{method}.pkl"))
        kernel_type = best_params.get("kernel", "rbf")
        C = best_params["C"]
        gamma = best_params.get("gamma", "scale")
    else:
        kernel_type = config.get("kernel", "rbf")
        C = config.get("C", 1.0)
        gamma = config.get("gamma", "scale")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.get("seed", 42))
    results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")
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
            raise ValueError("Unsupported method. Use 'fidelity' or 'qks'")

        acc, f1, prec, recall = log_metrics(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}")
        writer.add_scalar("Accuracy/Fold", acc, fold)
        writer.add_scalar("F1/Fold", f1, fold)
        writer.add_scalar("Precision/Fold", prec, fold)
        writer.add_scalar("Recall/Fold", recall, fold)

        results.append((acc, f1, prec, recall))

        save_checkpoint({
            "model_state_dict": clf,
            "scaler": scaler
        }, f"checkpoints/qkernel_fold_{fold}.pt")

    plot_quantum_circuit(method=method, filename=os.path.join(log_dir, "circuit.png"))
    writer.close()
    return results
