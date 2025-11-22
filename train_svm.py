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


def objective(trial, X_train, y_train, X_val, y_val, log_file, objective_cfg):
    kernel_options = objective_cfg.get("kernel_options", ["linear", "rbf", "poly"])
    gamma_choices = objective_cfg.get("gamma_options", ["scale", "auto"])
    c_range = objective_cfg.get("C_range", [1e-2, 10.0])
    gamma_range = objective_cfg.get("gamma_range", [1e-4, 10.0])
    allow_gamma_float = objective_cfg.get("allow_gamma_float", True)

    C = trial.suggest_float('C', c_range[0], c_range[1], log=True)
    kernel = trial.suggest_categorical('kernel', kernel_options)
    if allow_gamma_float and gamma_range:
        gamma = trial.suggest_float('gamma', gamma_range[0], gamma_range[1], log=True)
    else:
        gamma = trial.suggest_categorical('gamma', gamma_choices)

    model = EnhancedSVM(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=True,
        scaler=objective_cfg.get("scaler"),
        pca_model=objective_cfg.get("pca_model"),
        use_pca=objective_cfg.get("use_pca", False),
        auto_transform=True,
        use_gpu=objective_cfg.get("use_gpu", False),
    )
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_val, y_val)
    write_log(log_file, f"Trial: {trial.number} | Params: {trial.params} | F1: {metrics['f1']:.4f}")

    return 1.0 - metrics['f1']


def run_train_svm(config):
    dataset_name = config["dataset"]["name"]
    base_exp_name = config.get("experiment_name", "default_exp")
    EXPERIMENT_NAME = f"{dataset_name}_{base_exp_name}"

    checkpoint_cfg = config.get("checkpoint", {})
    checkpoint_root = checkpoint_cfg.get("save_dir", os.path.join("engine", "checkpoints"))
    checkpoint_subdir = checkpoint_cfg.get("subdir", os.path.join("svm", EXPERIMENT_NAME))

    SAVE_DIR = os.path.join(checkpoint_root, checkpoint_subdir)
    LOG_DIR = os.path.join(SAVE_DIR, "logs")
    os.makedirs(SAVE_DIR, exist_ok=True)

    wandb.init(
        project="qml_project",
        name=EXPERIMENT_NAME,
        config=config
    )

    os.makedirs(LOG_DIR, exist_ok=True)

    batch_size = config.get("training", {}).get("batch_size", 256)
    dataset_cfg = config.get("dataset", {})
    binary_classes = dataset_cfg.get("binary_classes", config.get("binary_classes", [3, 8]))
    grayscale = dataset_cfg.get("grayscale", config.get("model", {}).get("grayscale"))
    svm_cfg = config.get("svm", {})
    use_pca = svm_cfg.get("use_pca", False)
    pca_components = svm_cfg.get("pca_components", 50)
    optimize = svm_cfg.get("optimize", config.get("optuna", {}).get("optimize", False))
    n_trials = svm_cfg.get("n_trials", config.get("optuna", {}).get("n_trials", 30))
    use_gpu = svm_cfg.get("use_gpu", False)
    kernel_options = svm_cfg.get("kernel_options", ["rbf", "poly", "sigmoid"])
    gamma_choices = svm_cfg.get("gamma_options", ["scale", "auto"])
    c_range = svm_cfg.get("C_range", [1e-2, 10.0])
    gamma_range = svm_cfg.get("gamma_range", [1e-4, 10.0])
    default_C = svm_cfg.get("default_C", 1.0)
    default_kernel = svm_cfg.get("default_kernel", "rbf")
    default_gamma = svm_cfg.get("default_gamma", "scale")
    max_samples = svm_cfg.get("max_samples", 3000)

    log_path, log_file = init_logger(LOG_DIR, "svm")
    write_log(log_file, f"[SVM Training] Dataset: {dataset_name}, PCA: {use_pca} ({pca_components}), GPU: {use_gpu}\n")

    train_dataset, test_dataset = load_dataset_by_name(
        name=dataset_name,
        batch_size=batch_size,
        binary_classes=binary_classes,
        grayscale=grayscale,
    )

    indices = torch.randperm(len(train_dataset))[:max_samples]
    train_dataset = Subset(train_dataset, indices)

    print(f"Nombre d'exemples chargés dans train_dataset : {len(train_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def dataloader_to_numpy(loader):
        X_parts, y_parts = [], []
        for xb, yb in loader:
            X_parts.append(xb.view(xb.size(0), -1))
            y_parts.append(yb)
        X_cat = torch.cat(X_parts, dim=0)
        y_cat = torch.cat(y_parts, dim=0)
        return X_cat.numpy(), y_cat.numpy()

    X_raw, y = dataloader_to_numpy(train_loader)

    scaler = StandardScaler()
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_raw, y, test_size=0.2, random_state=42)
    scaler.fit(X_train_raw)
    pca = None
    if use_pca:
        pca = PCA(n_components=pca_components)
        pca.fit(scaler.transform(X_train_raw))

    study = None
    best_params = {"C": default_C, "kernel": default_kernel, "gamma": default_gamma}
    if optimize:
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(
                trial,
                X_train_raw,
                y_train,
                X_val_raw,
                y_val,
                log_file,
                {
                    "kernel_options": kernel_options,
                    "gamma_options": gamma_choices,
                    "C_range": c_range,
                    "gamma_range": gamma_range,
                    "allow_gamma_float": True,
                    "scaler": scaler,
                    "pca_model": pca,
                    "use_pca": use_pca,
                    "use_gpu": use_gpu,
                },
            ),
            n_trials=n_trials,
        )
        best_params = study.best_params
        write_log(log_file, f"\nBest Params: {best_params}\n")

    final_model = EnhancedSVM(
        **best_params,
        use_pca=use_pca,
        pca_model=pca,
        scaler=scaler,
        save_path=SAVE_DIR,
        probability=True,
        auto_transform=True,
        use_gpu=use_gpu,
    )
    final_model.fit(X_train_raw, y_train)
    final_model.save()

    metrics = final_model.evaluate(X_val_raw, y_val)
    try:
        y_pred_proba_val = final_model.predict_proba(X_val_raw)[:, 1]
        roc_auc = roc_auc_score(y_val, y_pred_proba_val)
    except Exception as e:
        print(f"[Warning] Could not compute ROC AUC on val set: {e}")
        roc_auc = float("nan")

    bal_acc = metrics.get("balanced_accuracy", balanced_accuracy_score(y_val, final_model.predict(X_val_raw)))

    write_log(log_file, f"Final Evaluation Metrics on Validation: {metrics}, Balanced Acc: {bal_acc:.4f}, ROC AUC: {roc_auc:.4f}\n")
    wandb.log({
        "val/f1": metrics["f1"],
        "val/accuracy": metrics["accuracy"],
        "val/precision": metrics["precision"],
        "val/recall": metrics["recall"],
        "val/balanced_accuracy": bal_acc,
        "val/auc": roc_auc
    })

    y_pred_val = final_model.predict(X_val_raw)
    cm = confusion_matrix(y_val, y_pred_val)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_path = os.path.join(SAVE_DIR, "confusion_matrix_val.png")
    disp.plot()
    plt.savefig(cm_path)
    plt.close()
    wandb.log({"confusion_matrix_val": wandb.Image(cm_path)})

    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        X_test_raw, y_test = dataloader_to_numpy(test_loader)

        y_pred_test = final_model.predict(X_test_raw)
        acc, f1, precision, recall = log_metrics(y_test, y_pred_test)

        try:
            y_pred_proba_test = final_model.predict_proba(X_test_raw)[:, 1]
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

    if study is not None:
        joblib.dump(study, os.path.join(SAVE_DIR, "optuna_study.pkl"))
    log_file.close()
    print("SVM training complete.")
    wandb.finish()

if __name__ == "__main__":
    import yaml
    with open("configs/config_train_svm_fashion.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_train_svm(config)