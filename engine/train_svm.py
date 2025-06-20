# train_svm.py
# Version: 3.0 – Logging, Optuna, PCA, Metrics, Logs structurés par expérience

import os
import joblib
import optuna
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils.logger import init_logger, write_log
from utils.data_loader.utils import load_dataset_by_name
from models.svm_extension import EnhancedSVM
from utils.metrics import log_metrics


def objective(trial, X_train, y_train, X_val, y_val, log_file):
    C = trial.suggest_float('C', 1e-2, 10.0, log=True)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

    model = EnhancedSVM(C=C, kernel=kernel, gamma=gamma)
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_val, y_val)
    write_log(log_file, f"Trial: {trial.number} | Params: {trial.params} | F1: {metrics['f1']:.4f}")
    return 1.0 - metrics['f1']


def run_train_svm(config):
    EXPERIMENT_NAME = config.get("experiment_name", "default_svm")
    SAVE_DIR = os.path.join("checkpoints", "svm", EXPERIMENT_NAME)
    LOG_DIR = os.path.join(SAVE_DIR, "logs")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    dataset = config["dataset"]
    batch_size = config.get("batch_size", 1024)
    selected_classes = config.get("selected_classes", [3, 8])
    use_pca = config["svm"].get("use_pca", False)
    pca_components = config["svm"].get("pca_components", 50)

    log_path, log_file = init_logger(LOG_DIR, "svm")
    write_log(log_file, f"[SVM Training] Dataset: {dataset}, PCA: {use_pca} ({pca_components})\n")

    loader, _ = load_dataset_by_name(dataset, batch_size, selected_classes=selected_classes, return_tensor_dataset=True)
    X, y = loader.dataset.tensors
    X, y = X.numpy().reshape(len(X), -1), y.numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    pca = None
    if use_pca:
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, log_file), n_trials=30)

    best_params = study.best_params
    write_log(log_file, f"\nBest Params: {best_params}\n")

    final_model = EnhancedSVM(**best_params, use_pca=use_pca, pca_model=pca, save_path=SAVE_DIR)
    final_model.fit(X_train, y_train)
    final_model.save()

    metrics = final_model.evaluate(X_val, y_val)
    write_log(log_file, f"Final Evaluation Metrics: {metrics}\n")

    y_pred = final_model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
    plt.close()

    joblib.dump(study, os.path.join(SAVE_DIR, "optuna_study.pkl"))
    log_file.close()
    print("SVM training complete.")


if __name__ == "__main__":
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_train_svm(config)
