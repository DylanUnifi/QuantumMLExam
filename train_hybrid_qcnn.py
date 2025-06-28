import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm, trange
import wandb

from models.hybrid_qcnn import HybridQCNNBinaryClassifier
from utils.checkpoint import load_checkpoint
from utils.early_stopping import EarlyStopping
from utils.scheduler import get_scheduler
from utils.logger import init_logger, write_log
from utils.metrics import log_metrics
from data_loader.utils import load_dataset_by_name


def run_train_hybrid_qcnn(config):
    EXPERIMENT_NAME = config.get("experiment_name", "hybrid_qcnn_exp")
    SAVE_DIR = os.path.join("engine/checkpoints", "hybrid_qcnn", EXPERIMENT_NAME)
    CHECKPOINT_DIR = os.path.join(SAVE_DIR, "folds")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    wandb.init(project="qml_project", name=EXPERIMENT_NAME, config=config)
    wandb.config.update({
        "dataset_name": config["dataset"]["name"],
        "binary_classes": config["dataset"].get("binary_classes", [3,8]),
        "n_qubits": config["quantum"]["n_qubits"],
        "quantum_backend": config["quantum"]["backend"],
        "learning_rate": config["training"]["learning_rate"],
        "epochs": config["training"]["epochs"],
        "batch_size": config["training"]["batch_size"],
        "scheduler": config.get("scheduler", None),
        "kfold": config["training"]["kfold"],
        "early_stopping": config["training"]["early_stopping"],
    })

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = config["training"]["batch_size"]
    EPOCHS = config["training"]["epochs"]
    LR = config["training"]["learning_rate"]
    KFOLD = config["training"]["kfold"]
    PATIENCE = config["training"]["early_stopping"]
    SCHEDULER_TYPE = config.get("scheduler", None)

    train_dataset, test_dataset = load_dataset_by_name(
        name=config["dataset"]["name"],
        batch_size=BATCH_SIZE,
        binary_classes=config.get("binary_classes", [3, 8])
    )

    indices = torch.randperm(len(train_dataset))[:2000]
    train_dataset = Subset(train_dataset, indices)

    print(f"Nombre d'exemples chargés dans train_dataset : {len(train_dataset)}")

    kfold = KFold(n_splits=KFOLD, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f"[Fold {fold}] Starting Hybrid QCNN training...")
        log_path, log_file = init_logger(os.path.join(SAVE_DIR, "logs"), fold)
        write_log(log_file, f"[Fold {fold}] Hybrid QCNN Training Log\n")

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

        sample_X, _ = train_dataset[0]
        input_size = sample_X.numel()

        model = HybridQCNNBinaryClassifier(input_size=input_size).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = get_scheduler(optimizer, SCHEDULER_TYPE)
        criterion = nn.BCELoss()

        start_epoch = 0
        try:
            model, optimizer, start_epoch = load_checkpoint(model, optimizer, CHECKPOINT_DIR, fold)
            print(f"Resuming from epoch {start_epoch}")
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch")

        loss_history, f1_history = [], []
        best_f1, best_epoch = 0, 0
        stopped_early = False

        for epoch in trange(start_epoch, EPOCHS, desc=f"[Fold {fold}] Hybrid QCNN Training"):
            model.train()
            total_loss = 0
            start_time = time.time()
            for batch_X, batch_y in tqdm(train_loader, desc=f"[Fold {fold}] Batches"):
                batch_X, batch_y = batch_X.view(batch_X.size(0), -1).to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(batch_X).view(-1)
                batch_y = batch_y.view(-1).float()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.view(batch_X.size(0), -1).to(DEVICE), batch_y.to(DEVICE)
                    preds = model(batch_X).view(-1)
                    preds = (preds >= 0.5).float()
                    y_true.extend(batch_y.tolist())
                    y_pred.extend(preds.cpu().tolist())

            acc, f1, precision, recall = log_metrics(y_true, y_pred)
            val_loss = total_loss / len(train_loader)
            duration = time.time() - start_time

            wandb.log({
                "train/loss": val_loss,
                "val/f1": f1,
                "val/accuracy": acc,
                "val/precision": precision,
                "val/recall": recall,
                "epoch": epoch,
                "fold": fold,
            })

            write_log(log_file,
                      f"[Epoch {epoch}] Time: {duration:.2f}s | Loss: {val_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")

            loss_history.append(val_loss)
            f1_history.append(f1)

            if f1 > best_f1:
                best_f1, best_epoch = f1, epoch
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_f1": best_f1
                }
                save_path = os.path.join(CHECKPOINT_DIR, f"hybrid_qcnn_fold_{fold}.pt")
                torch.save(checkpoint, save_path)
                print(f"✅ Checkpoint saved at {save_path}")
                write_log(log_file, f"[Epoch {epoch}] New best F1: {f1:.4f} (Saved model)")

                wandb.run.summary["best_f1"] = best_f1
                wandb.run.summary["best_epoch"] = best_epoch

            if EarlyStopping(patience=PATIENCE)(f1):
                print("Early stopping triggered.")
                write_log(log_file, f"Early stopping triggered at epoch {epoch}")
                stopped_early = True
                break

            if scheduler:
                scheduler.step()

        write_log(log_file, f"\n[Fold {fold}] Best F1: {best_f1:.4f} at epoch {best_epoch}")
        if stopped_early:
            write_log(log_file, f"Training stopped early before reaching max epochs ({EPOCHS})\n")
        else:
            write_log(log_file, f"Training completed full {EPOCHS} epochs\n")

        # Final evaluation on test set
        if test_dataset is not None:
            print(f"[Fold {fold}] Loading best model and evaluating on test set...")
            checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, f"hybrid_qcnn_fold_{fold}.pt"))
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            y_test_true, y_test_pred = [], []

            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.view(batch_X.size(0), -1).to(DEVICE), batch_y.to(DEVICE)
                    preds = model(batch_X).view(-1)
                    preds = (preds >= 0.5).float()
                    y_test_true.extend(batch_y.tolist())
                    y_test_pred.extend(preds.cpu().tolist())

            acc, f1, precision, recall = log_metrics(y_test_true, y_test_pred)
            print(f"[Fold {fold}] Test Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
            write_log(log_file,
                      f"\n[Fold {fold}] Test Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
            wandb.log({
                "test/accuracy": acc,
                "test/f1": f1,
                "test/precision": precision,
                "test/recall": recall,
                "fold": fold,
            })

            # Global log for final results
            global_log_path = os.path.join(SAVE_DIR, "logs", "test_evaluation.log")
            with open(global_log_path, "a") as global_log_file:
                write_log(global_log_file,
                          f"[Fold {fold}] Test Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        log_file.close()

    print("Hybrid QCNN training and evaluation complete.")


if __name__ == "__main__":
    import argparse, yaml
    parser = argparse.ArgumentParser(description="Train Hybrid QCNN")
    parser.add_argument("--config", type=str, default="configs/config_train_qcnn_cifar10.yaml", help="Path to YAML config")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    run_train_hybrid_qcnn(config)
