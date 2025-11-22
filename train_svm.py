import os
import joblib
import optuna
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, balanced_accuracy_score
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

    model = EnhancedSVM(C=C, kernel=kernel, gamma=gamma, probability=True)  # 👈 Active predict_proba
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_val, y_val)
    write_log(log_file, f"Trial: {trial.number} | Params: {trial.params} | F1: {metrics['f1']:.4f}")

    return 1.0 - metrics['f1']


def run_train_svm(config):
    dataset_name = config["dataset"]["name"]
    base_exp_name = config.get("experiment_name", "default_exp")
    EXPERIMENT_NAME = f"{dataset_name}_{base_exp_name}"

    SAVE_DIR = os.path.join("engine/checkpoints", "svm", EXPERIMENT_NAME)
    LOG_DIR = os.path.join(SAVE_DIR, "logs")
    os.makedirs(SAVE_DIR, exist_ok=True)

    os.makedirs(LOG_DIR, exist_ok=True)

    batch_size = config["training"]["batch_size"]
    dataset_cfg = config.get("dataset", {})
    binary_classes = dataset_cfg.get("binary_classes", [3, 8])
    use_pca = config["svm"].get("use_pca", False)
    pca_components = config["svm"].get("pca_components", 50)

    log_path, log_file = init_logger(LOG_DIR, "svm")
    write_log(log_file, f"[SVM Training] Dataset: {dataset_name}, PCA: {use_pca} ({pca_components})\n")

    train_dataset, test_dataset = load_dataset_by_name(
        name=dataset_name,
        batch_size=batch_size,
        binary_classes=binary_classes,
        grayscale=dataset_cfg.get("grayscale", config.get("model", {}).get("grayscale"))
    )

    indices = torch.randperm(len(train_dataset))[:3000]
    train_dataset = Subset(train_dataset, indices)

    print(f"Nombre d'exemples chargés dans train_dataset : {len(train_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
    X_train_tensor, y_train_tensor = next(iter(train_loader))
    X, y = X_train_tensor.view(X_train_tensor.size(0), -1).numpy(), y_train_tensor.numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

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

    final_model = EnhancedSVM(**best_params, use_pca=use_pca, pca_model=pca, save_path=SAVE_DIR, probability=True)
    final_model.fit(X_train_pca, y_train)
    final_model.save()

    metrics = final_model.evaluate(X_val_pca, y_val)
    try:
        y_pred_proba_val = final_model.predict_proba(X_val_pca)[:, 1]
        roc_auc = roc_auc_score(y_val, y_pred_proba_val)
    except Exception as e:
        print(f"[Warning] Could not compute ROC AUC on val set: {e}")
        roc_auc = float("nan")

    bal_acc = balanced_accuracy_score(y_val, final_model.predict(X_val_pca))

    write_log(log_file, f"Final Evaluation Metrics on Validation: {metrics}, Balanced Acc: {bal_acc:.4f}, ROC AUC: {roc_auc:.4f}\n")
    wandb.log({
        "val/f1": metrics["f1"],
        "val/accuracy": metrics["accuracy"],
        "val/precision": metrics["precision"],
        "val/recall": metrics["recall"],
        "val/balanced_accuracy": bal_acc,
        "val/auc": roc_auc
    })

    y_pred_val = final_model.predict(X_val_pca)
    cm = confusion_matrix(y_val, y_pred_val)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_path = os.path.join(SAVE_DIR, "confusion_matrix_val.png")
    disp.plot()
    plt.savefig(cm_path)
    plt.close()
    wandb.log({"confusion_matrix_val": wandb.Image(cm_path)})

    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
        X_test_tensor, y_test_tensor = next(iter(test_loader))
        X_test, y_test = X_test_tensor.view(X_test_tensor.size(0), -1).numpy(), y_test_tensor.numpy()
        X_test = scaler.transform(X_test)
        if pca is not None:
            X_test = pca.transform(X_test)

        y_pred_test = final_model.predict(X_test)
        acc, f1, precision, recall = log_metrics(y_test, y_pred_test)

        try:
            y_pred_proba_test = final_model.predict_proba(X_test)[:, 1]
            roc_auc_test = roc_auc_score(y_test, y_pred_proba_test)
        except Exception as e:
            print(f"[Warning] Could not compute ROC AUC on test set: {e}")
            roc_auc_test = float("nan")

        bal_acc_test = balanced_accuracy_score(y_test, y_pred_test)

        write_log(log_file, f"\nTest Metrics: acc={acc:.4f}, f1={f1:.4f}, precision={precision:.4f}, recall={recall:.4f}, Balanced Acc={bal_acc_test:.4f}, ROC AUC={roc_auc_test:.4f}")
        wandb.log({
            "test/f1": f1,
            "test/accuracy": acc,
            "test/precision": precision,
            "test/recall": recall,
            "test/balanced_accuracy": bal_acc_test,
            "test/auc": roc_auc_test
        })

        cm_test = confusion_matrix(y_test, y_pred_test)
        disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)
        cm_test_path = os.path.join(SAVE_DIR, "confusion_matrix_test.png")
        disp_test.plot()
        plt.savefig(cm_test_path)
        plt.close()
        wandb.log({"confusion_matrix_test": wandb.Image(cm_test_path)})

    joblib.dump(study, os.path.join(SAVE_DIR, "optuna_study.pkl"))
    log_file.close()
    print("SVM training complete.")