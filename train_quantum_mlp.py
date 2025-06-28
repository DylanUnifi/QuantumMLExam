# train_quantum_mlp.py
# Version: 1.1 – Aligne sur train_classical avec logger, early stopping, checkpoints, TensorBoard

import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from models.hybrid_qclassical import QuantumResidualMLP
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.early_stopping import EarlyStopping
from utils.scheduler import get_scheduler
from utils.logger import init_logger, write_log
from utils.metrics import log_metrics
from utils.visual import save_plots
from data_loader.utils import load_dataset_by_name
import wandb
from tqdm import tqdm, trange


def run_train_quantum_mlp(config):
    EXPERIMENT_NAME = config.get("experiment_name", "quantum_mlp_exp")

    wandb.init(project="qml_project", name=EXPERIMENT_NAME, config=config)
    wandb.config.update(config)

    SAVE_DIR = os.path.join("engine/checkpoints", "quantum_mlp", EXPERIMENT_NAME)
    CHECKPOINT_DIR = os.path.join(SAVE_DIR, "folds")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = config["training"]["batch_size"]
    EPOCHS = config["training"]["epochs"]
    LR = config["training"]["learning_rate"]
    KFOLD = config["training"]["kfold"]
    PATIENCE = config["training"]["early_stopping"]
    SCHEDULER_TYPE = config.get("scheduler", None)

    # Charger dataset en TensorDataset
    train_dataset, test_dataset = load_dataset_by_name(
        name=config["dataset"]["name"],
        batch_size=BATCH_SIZE,
        binary_classes=config.get("binary_classes", [0, 1])
    )

    indices = torch.randperm(len(train_dataset))[:2000]
    train_dataset = Subset(train_dataset, indices)

    print(f"Nombre d'exemples chargés dans train_dataset : {len(train_dataset)}")

    kfold = KFold(n_splits=KFOLD, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f"[Fold {fold}] Starting Quantum MLP training...")

        writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, f"fold_{fold}"))
        early_stopping = EarlyStopping(patience=PATIENCE)

        log_path, log_file = init_logger(os.path.join(SAVE_DIR, "logs"), fold)
        write_log(log_file, f"[Fold {fold}] Quantum MLP Training Log\n")

        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

        # Exemple pour input_size
        sample_X, _ = train_dataset[0]
        input_size = sample_X.numel()

        model = QuantumResidualMLP(input_size=input_size).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = get_scheduler(optimizer, SCHEDULER_TYPE)
        criterion = nn.BCELoss()

        start_epoch = 0
        try:
            model, optimizer, start_epoch = load_checkpoint(model, optimizer, CHECKPOINT_DIR, fold)
            print(f"Resuming from epoch {start_epoch}")
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch.")

        loss_history, f1_history = [], []
        best_f1, best_epoch = 0, 0
        stopped_early = False

        for epoch in trange(start_epoch, EPOCHS, desc=f"[Fold {fold}] Quantum MLP Training"):
            model.train()
            total_loss = 0
            for batch_X, batch_y in tqdm(train_loader, desc=f"[Fold {fold}] Batches"):
                batch_X, batch_y = batch_X.view(batch_X.size(0), -1).to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                batch_y = batch_y.view(-1).float()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.view(batch_X.size(0), -1).to(DEVICE)
                    preds = model(batch_X).squeeze()
                    preds = (preds >= 0.5).float()
                    y_true.extend(batch_y.tolist())
                    y_pred.extend(preds.cpu().tolist())

            acc, f1, precision, recall = log_metrics(y_true, y_pred)
            val_loss = total_loss / len(train_loader)

            writer.add_scalar("Loss/train", val_loss, epoch)
            writer.add_scalar("F1/val", f1, epoch)
            writer.add_scalar("Accuracy/val", acc, epoch)
            writer.add_scalar("Precision/val", precision, epoch)
            writer.add_scalar("Recall/val", recall, epoch)

            write_log(log_file, f"[Epoch {epoch}] Loss: {val_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")

            loss_history.append(val_loss)
            f1_history.append(f1)

            # 📊 Log des métriques à wandb
            wandb.log({
                "train/loss": val_loss,
                "val/f1": f1,
                "val/accuracy": acc,
                "val/precision": precision,
                "val/recall": recall,
                "epoch": epoch,
                "fold": fold,
            })

            print(f"[Fold {fold}][Epoch {epoch}] Loss: {val_loss:.4f} | F1: {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                save_checkpoint(model, optimizer, epoch, CHECKPOINT_DIR, fold, best_f1)
                write_log(log_file, f"[Epoch {epoch}] New best F1: {f1:.4f} (Saved model)")

                wandb.run.summary["best_f1"] = best_f1
                wandb.run.summary["best_epoch"] = best_epoch

            if early_stopping(f1):
                print("Early stopping triggered.")
                write_log(log_file, f"Early stopping triggered at epoch {epoch}")
                stopped_early = True
                break

            if scheduler:
                scheduler.step()

        wandb.run.summary[f"fold_{fold}/best_f1"] = best_f1
        wandb.run.summary[f"fold_{fold}/best_epoch"] = best_epoch

        save_plots(fold, loss_history, f1_history, os.path.join(SAVE_DIR, "plots"))
        writer.close()

        write_log(log_file, f"\n[Fold {fold}] Best F1: {best_f1:.4f} at epoch {best_epoch}")
        if stopped_early:
            write_log(log_file, f"Training stopped early before reaching max epochs ({EPOCHS})\n")
        else:
            write_log(log_file, f"Training completed full {EPOCHS} epochs\n")

        # Évaluation finale sur test set
        if test_dataset is not None:
            print(f"[Fold {fold}] Loading best model and evaluating on test set...")
            model, _, _ = load_checkpoint(model, optimizer, CHECKPOINT_DIR, fold)
            model.eval()
            y_test_true, y_test_pred = [], []

            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)  # AJOUT

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.view(batch_X.size(0), -1).to(DEVICE), batch_y.to(DEVICE)
                    preds = model(batch_X).squeeze()
                    preds = (preds >= 0.5).float()
                    y_test_true.extend(batch_y.tolist())
                    y_test_pred.extend(preds.cpu().tolist())

            acc, f1, precision, recall = log_metrics(y_test_true, y_test_pred)
            print(
                f"[Fold {fold}] Test Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
            write_log(log_file,
                      f"\n[Fold {fold}] Test Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

            wandb.log({
                "test/f1": f1,
                "test/accuracy": acc,
                "test/precision": precision,
                "test/recall": recall,
                "fold": fold
            })

            log_file.close()

    print("Quantum MLP training complete.")


if __name__ == "__main__":
    import yaml
    with open("configs/config_train_quantum_mlp_fashion.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_train_quantum_mlp(config)
