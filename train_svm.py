# train_svm.py
# Version: 3.9 – PCA cohérente, réentraînement final sur full set, logging structuré

import os
import joblib
import optuna
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from torch.utils.data import Subset

from utils.logger import init_logger, write_log
from data_loader.utils import load_dataset_by_name
from models.svm_extension import EnhancedSVM
from utils.metrics import log_metrics
import wandb


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
    EXPERIMENT_NAME = config.get("experiment_name", "svm_exp")
    SAVE_DIR = os.path.join("engine/checkpoints", "svm", EXPERIMENT_NAME)
    LOG_DIR = os.path.join(SAVE_DIR, "logs")
    os.makedirs(SAVE_DIR, exist_ok=True)

    wandb.init(
        project="qml_project",
        name=EXPERIMENT_NAME,
        config=config
    )

    os.makedirs(LOG_DIR, exist_ok=True)

    batch_size = config["training"]["batch_size"]
    dataset_name = config["dataset"]["name"]
    binary_classes = config.get("binary_classes", [3, 8])
    use_pca = config["svm"].get("use_pca", False)
    pca_components = config["svm"].get("pca_components", 50)

    log_path, log_file = init_logger(LOG_DIR, "svm")
    write_log(log_file, f"[SVM Training] Dataset: {dataset_name}, PCA: {use_pca} ({pca_components})\n")

    train_dataset, test_dataset = load_dataset_by_name(
        name=dataset_name,
        batch_size=batch_size,
        binary_classes=binary_classes
    )

    indices = torch.randperm(len(train_dataset))[:3000]
    train_dataset = Subset(train_dataset, indices)

    print(f"Nombre d'exemples chargés dans train_dataset : {len(train_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
    X_train_tensor, y_train_tensor = next(iter(train_loader))
    X, y = X_train_tensor.view(X_train_tensor.size(0), -1).numpy(), y_train_tensor.numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split avant PCA pour éviter d'entraîner PCA sur val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    pca = None
    if use_pca:
        pca = PCA(n_components=pca_components)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)
    else:
        X_train_pca, X_val_pca = X_train, X_val

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train_pca, y_train, X_val_pca, y_val, log_file), n_trials=30)

    best_params = study.best_params
    write_log(log_file, f"\nBest Params: {best_params}\n")

    final_model = EnhancedSVM(**best_params, use_pca=use_pca, pca_model=pca, save_path=SAVE_DIR)
    final_model.fit(X_train_pca, y_train)
    final_model.save()

    metrics = final_model.evaluate(X_val_pca, y_val)
    write_log(log_file, f"Final Evaluation Metrics on Validation: {metrics}\n")
    wandb.log({
        "val/f1": metrics["f1"],
        "val/accuracy": metrics["accuracy"],
        "val/precision": metrics["precision"],
        "val/recall": metrics["recall"]
    })

    y_pred_val = final_model.predict(X_val_pca)
    cm = confusion_matrix(y_val, y_pred_val)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix_val.png"))
    plt.close()
    wandb.log({"confusion_matrix_val": wandb.Image(os.path.join(SAVE_DIR, "confusion_matrix_val.png"))})

    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
        X_test_tensor, y_test_tensor = next(iter(test_loader))
        X_test, y_test = X_test_tensor.view(X_test_tensor.size(0), -1).numpy(), y_test_tensor.numpy()
        X_test = scaler.transform(X_test)
        if pca is not None:
            X_test = pca.transform(X_test)

        y_pred_test = final_model.predict(X_test)
        acc, f1, precision, recall = log_metrics(y_test, y_pred_test)
        write_log(log_file,
                  f"\nTest Metrics: acc={acc:.4f}, f1={f1:.4f}, precision={precision:.4f}, recall={recall:.4f}")

        cm_test = confusion_matrix(y_test, y_pred_test)
        disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)
        cm_test_path = os.path.join(SAVE_DIR, "confusion_matrix_test.png")
        disp_test.plot()
        plt.savefig(cm_test_path)
        plt.close()

        wandb.log({
            f"test/f1": f1,
            f"test/accuracy": acc,
            f"test/precision": precision,
            f"test/recall": recall,
        })

    joblib.dump(study, os.path.join(SAVE_DIR, "optuna_study.pkl"))
    log_file.close()
    print("SVM training complete.")
    wandb.finish()



if __name__ == "__main__":
    import yaml
    with open("configs/config_train_svm_fashion.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_train_svm(config)
