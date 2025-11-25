import os
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm
import torch
from torch.utils.data import Subset, DataLoader

from data_loader.utils import load_dataset_by_name
from utils.logger import init_logger, write_log
from utils.metrics import log_metrics
import wandb


def build_kernel_fn(n_wires: int, n_layers: int, rotation: str = "Y", device_name: str = "default.qubit"):
    register_a = list(range(n_wires))
    register_b = list(range(n_wires, 2 * n_wires))
    ancilla = 2 * n_wires
    dev = qml.device(device_name, wires=2 * n_wires + 1)

    def _embed_and_entangle(x, wires):
        qml.AngleEmbedding(x, wires=wires, rotation=rotation)
        for _ in range(n_layers):
            for i, w in enumerate(wires):
                qml.CZ(wires=[w, wires[(i + 1) % len(wires)]])
            qml.AngleEmbedding(x, wires=wires, rotation=rotation)

    @qml.qnode(dev)
    def swap_test_kernel(x, y):
        _embed_and_entangle(x, register_a)
        _embed_and_entangle(y, register_b)

        qml.Hadamard(wires=ancilla)
        for i in range(n_wires):
            qml.CSWAP(wires=[ancilla, register_a[i], register_b[i]])
        qml.Hadamard(wires=ancilla)

        # Probability of ancilla being |0> encodes fidelity via F = 2 * p0 - 1
        return qml.expval(qml.Projector([0], wires=ancilla))

    def fidelity_kernel(x, y):
        prob_zero = swap_test_kernel(x, y)
        return 2 * prob_zero - 1

    return fidelity_kernel


def select_device_name(qkernel_cfg, n_wires: int, total_wires: int):
    base_device = qkernel_cfg.get("device", "default.qubit")
    use_gpu = qkernel_cfg.get("use_gpu", False)
    gpu_device = qkernel_cfg.get("gpu_device", "lightning.gpu")
    kokkos_device = qkernel_cfg.get("kokkos_device", "lightning.kokkos")

    if not use_gpu:
        return base_device

    if torch.cuda.is_available():
        try:
            qml.device(gpu_device, wires=total_wires)
            print(f"[Info] Using GPU-backed PennyLane device: {gpu_device}")
            return gpu_device
        except Exception as exc:  # pragma: no cover - backend availability depends on environment
            print(f"[Warning] Could not create GPU device '{gpu_device}' ({exc}); trying kokkos fallback...")
    else:
        print("[Warning] GPU requested for qkernel but CUDA is not available; trying kokkos fallback...")

    try:
        qml.device(kokkos_device, wires=total_wires)
        print(f"[Info] Using CPU-accelerated PennyLane device: {kokkos_device}")
        return kokkos_device
    except Exception as exc:  # pragma: no cover - backend availability depends on environment
        print(f"[Warning] Could not create kokkos device '{kokkos_device}' ({exc}); falling back to {base_device}.")
        return base_device


def _compute_kernel_row(x_row, X_ref, kernel_fn):
    return [kernel_fn(x_row, x_ref) for x_ref in X_ref]


def compute_kernel_matrix_with_progress(X_left, X_right, kernel_fn, workers: int = 1, desc: str = ""):
    workers = max(1, int(workers or 1))
    iterator = range(len(X_left))
    if desc:
        iterator = tqdm(iterator, desc=desc)

    if workers == 1:
        rows = [_compute_kernel_row(X_left[i], X_right, kernel_fn) for i in iterator]
    else:
        try:
            rows = Parallel(n_jobs=workers, prefer="processes", backend="loky")(
                delayed(_compute_kernel_row)(X_left[i], X_right, kernel_fn) for i in iterator
            )
        except Exception as exc:
            print(f"[Warning] Parallel kernel computation failed ({exc}); falling back to single-worker mode.")
            rows = [_compute_kernel_row(X_left[i], X_right, kernel_fn) for i in tqdm(range(len(X_left)), desc=f"{desc} (fallback)")]

    return np.array(rows)


def prepare_features(dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    X_parts, y_parts = [], []
    for xb, yb in loader:
        X_parts.append(xb.view(xb.size(0), -1))
        y_parts.append(yb)
    X = torch.cat(X_parts, dim=0)
    y = torch.cat(y_parts, dim=0)
    return X.numpy(), y.numpy()


def run_train_svm_qkernel(config):
    dataset_cfg = config.get("dataset", {})
    dataset_name = dataset_cfg.get("name")
    base_exp_name = config.get("experiment_name", "default_exp")
    experiment_name = f"{dataset_name}_{base_exp_name}_qkernel"

    checkpoint_cfg = config.get("checkpoint", {})
    checkpoint_root = checkpoint_cfg.get("save_dir", os.path.join("engine", "checkpoints"))
    checkpoint_subdir = checkpoint_cfg.get("subdir", os.path.join("svm_qkernel", experiment_name))

    save_dir = os.path.join(checkpoint_root, checkpoint_subdir)
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    wandb.init(project="qml_project", name=experiment_name, config=config)

    training_cfg = config.get("training", {})
    batch_size = training_cfg.get("batch_size", 128)

    binary_classes = dataset_cfg.get("binary_classes", config.get("binary_classes", [3, 8]))
    grayscale = dataset_cfg.get("grayscale", config.get("model", {}).get("grayscale"))

    qkernel_cfg = config.get("qkernel", {})
    n_wires = qkernel_cfg.get("n_wires", 6)
    n_layers = qkernel_cfg.get("n_layers", 2)
    rotation = qkernel_cfg.get("rotation", "Y")
    total_wires = 2 * n_wires + 1
    device_name = select_device_name(qkernel_cfg, n_wires, total_wires)
    use_pca = qkernel_cfg.get("use_pca", True)
    pca_components = qkernel_cfg.get("pca_components", n_wires)
    max_samples = qkernel_cfg.get("max_samples", 1500)
    C = qkernel_cfg.get("C", 1.0)
    kernel_workers = qkernel_cfg.get("kernel_workers") or os.cpu_count() or 1

    log_path, log_file = init_logger(log_dir, "svm_qkernel")
    write_log(
        log_file,
        f"[QKernel SVM] Dataset: {dataset_name}, wires: {n_wires}, layers: {n_layers}, PCA: {use_pca} ({pca_components}), device: {device_name}, use_gpu: {qkernel_cfg.get('use_gpu', False)}, kernel_workers: {kernel_workers}\n",
    )

    train_dataset, test_dataset = load_dataset_by_name(
        name=dataset_name,
        batch_size=batch_size,
        binary_classes=binary_classes,
        grayscale=grayscale,
    )

    #indices = torch.randperm(len(train_dataset))[:max_samples]
    #train_dataset = Subset(train_dataset, indices)
    print(f"Nombre d'exemples chargés dans train_dataset : {len(train_dataset)}")

    X_raw, y_raw = prepare_features(train_dataset, batch_size=batch_size)

    scaler = StandardScaler()
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)
    scaler.fit(X_train_raw)

    from sklearn.decomposition import PCA

    pca = None
    if use_pca:
        pca = PCA(n_components=min(pca_components, n_wires))
        pca.fit(scaler.transform(X_train_raw))

    def transform_features(X):
        X_proc = scaler.transform(X)
        if pca is not None:
            X_proc = pca.transform(X_proc)
        else:
            X_proc = X_proc[:, :n_wires]
        if X_proc.shape[1] != n_wires:
            if X_proc.shape[1] < n_wires:
                pad_width = n_wires - X_proc.shape[1]
                X_proc = np.pad(X_proc, ((0, 0), (0, pad_width)), mode="constant")
            else:
                X_proc = X_proc[:, :n_wires]
        return X_proc

    X_train_proc = transform_features(X_train_raw)
    X_val_proc = transform_features(X_val_raw)

    kernel_fn = build_kernel_fn(n_wires=n_wires, n_layers=n_layers, rotation=rotation, device_name=device_name)

    write_log(log_file, f"Building training kernel matrix with {kernel_workers} worker(s)...\n")
    t_start = time.time()
    K_train = compute_kernel_matrix_with_progress(
        X_train_proc, X_train_proc, kernel_fn, workers=kernel_workers, desc="Train kernel rows"
    )
    write_log(log_file, f"Finished training kernel matrix in {time.time() - t_start:.2f}s\n")

    write_log(log_file, "Building validation kernel matrix...\n")
    K_val = compute_kernel_matrix_with_progress(
        X_val_proc, X_train_proc, kernel_fn, workers=kernel_workers, desc="Val kernel rows"
    )

    svm_model = SVC(kernel="precomputed", C=C, probability=True)
    svm_model.fit(K_train, y_train)

    metrics = log_metrics(y_val, svm_model.predict(K_val))
    try:
        y_val_proba = svm_model.predict_proba(K_val)[:, 1]
        roc_auc_val = roc_auc_score(y_val, y_val_proba)
    except Exception as exc:  # pragma: no cover - robustness for edge cases
        print(f"[Warning] Could not compute ROC AUC on val set: {exc}")
        roc_auc_val = float("nan")

    bal_acc_val = balanced_accuracy_score(y_val, svm_model.predict(K_val))

    write_log(log_file, f"Validation Metrics: acc={metrics[0]:.4f}, f1={metrics[1]:.4f}, precision={metrics[2]:.4f}, recall={metrics[3]:.4f}, Balanced Acc={bal_acc_val:.4f}, ROC AUC={roc_auc_val:.4f}\n")
    wandb.log({
        "val/accuracy": metrics[0],
        "val/f1": metrics[1],
        "val/precision": metrics[2],
        "val/recall": metrics[3],
        "val/balanced_accuracy": bal_acc_val,
        "val/auc": roc_auc_val,
    })

    cm = confusion_matrix(y_val, svm_model.predict(K_val))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    cm_path = os.path.join(save_dir, "confusion_matrix_val.png")
    disp.plot()
    plt.savefig(cm_path)
    plt.close()
    wandb.log({"confusion_matrix_val": wandb.Image(cm_path)})

    if test_dataset is not None:
        X_test_raw, y_test = prepare_features(test_dataset, batch_size=batch_size)
        X_test_proc = transform_features(X_test_raw)
        write_log(log_file, "Building test kernel matrix...\n")
        K_test = compute_kernel_matrix_with_progress(
            X_test_proc, X_train_proc, kernel_fn, workers=kernel_workers, desc="Test kernel rows"
        )

        y_pred_test = svm_model.predict(K_test)
        acc_test, f1_test, precision_test, recall_test = log_metrics(y_test, y_pred_test)

        try:
            y_test_proba = svm_model.predict_proba(K_test)[:, 1]
            roc_auc_test = roc_auc_score(y_test, y_test_proba)
        except Exception as exc:  # pragma: no cover - robustness for edge cases
            print(f"[Warning] Could not compute ROC AUC on test set: {exc}")
            roc_auc_test = float("nan")

        bal_acc_test = balanced_accuracy_score(y_test, y_pred_test)

        write_log(log_file, f"Test Metrics: acc={acc_test:.4f}, f1={f1_test:.4f}, precision={precision_test:.4f}, recall={recall_test:.4f}, Balanced Acc={bal_acc_test:.4f}, ROC AUC={roc_auc_test:.4f}\n")
        wandb.log({
            "test/accuracy": acc_test,
            "test/f1": f1_test,
            "test/precision": precision_test,
            "test/recall": recall_test,
            "test/balanced_accuracy": bal_acc_test,
            "test/auc": roc_auc_test,
        })

        cm_test = confusion_matrix(y_test, y_pred_test)
        disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)
        cm_test_path = os.path.join(save_dir, "confusion_matrix_test.png")
        disp_test.plot()
        plt.savefig(cm_test_path)
        plt.close()
        wandb.log({"confusion_matrix_test": wandb.Image(cm_test_path)})

    model_payload = {
        "svc": svm_model,
        "scaler": scaler,
        "pca": pca,
        "train_features": X_train_proc,
        "kernel_params": {
            "n_wires": n_wires,
            "n_layers": n_layers,
            "rotation": rotation,
            "device_name": device_name,
            "C": C,
        },
    }
    joblib.dump(model_payload, os.path.join(save_dir, "svm_qkernel_model.pkl"))

    log_file.close()
    print("Quantum kernel SVM training complete.")
    wandb.finish()


if __name__ == "__main__":
    import yaml
    with open("configs/config_train_svm_qkernel_fashion.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_train_svm_qkernel(config)
