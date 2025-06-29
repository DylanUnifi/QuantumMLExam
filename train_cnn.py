# train_cnn.py
# Version: 4.1 – Ajout logs AUC et balanced accuracy sur le test set

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, balanced_accuracy_score  # 🔥 Ajout

from models.cnn import CNNBinaryClassifier
from utils.checkpoint import save_checkpoint, safe_load_checkpoint
from utils.early_stopping import EarlyStopping
from utils.metrics import log_metrics
from data_loader.utils import load_dataset_by_name
from utils.scheduler import get_scheduler
from utils.visual import save_plots
from utils.logger import init_logger, write_log
from tqdm import tqdm, trange
import wandb


def run_train_cnn(config):
    dataset_name = config["dataset"]["name"]  # Ex: "fashion-mnist", "cifar10", "svhn"
    base_exp_name = config.get("experiment_name", "default_exp")
    EXPERIMENT_NAME = f"{dataset_name}_{base_exp_name}"

    SAVE_DIR = os.path.join("engine/checkpoints", "cnn", EXPERIMENT_NAME)
    CHECKPOINT_DIR = os.path.join(SAVE_DIR, "folds")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    wandb.init(
        project="qml_project",
        name=EXPERIMENT_NAME,
        config=config
    )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = config["training"]["batch_size"]
    EPOCHS = config["training"]["epochs"]
    LR = config["training"]["learning_rate"]
    KFOLD = config["training"]["kfold"]
    PATIENCE = config["training"]["early_stopping"]
    SCHEDULER_TYPE = config.get("scheduler", None)
    IN_CHANNELS = config['model']['in_channels']

    train_dataset, test_dataset = load_dataset_by_name(
        name=config["dataset"]["name"],
        batch_size=BATCH_SIZE,
        binary_classes=config.get("binary_classes", [3, 8]),
        grayscale=config["dataset"].get("grayscale", None)
    )

    indices = torch.randperm(len(train_dataset))[:3000]
    train_dataset = Subset(train_dataset, indices)

    print(f"Nombre d'exemples chargés dans train_dataset : {len(train_dataset)}")

    kfold = KFold(n_splits=KFOLD, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f"[Fold {fold}] Starting training...")

        writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, f"fold_{fold}"))
        early_stopping = EarlyStopping(patience=PATIENCE)

        log_path, log_file = init_logger(os.path.join(SAVE_DIR, "logs"), fold)
        write_log(log_file, f"[Fold {fold}] CNN Training Log\n")

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

        model = CNNBinaryClassifier(in_channels=IN_CHANNELS).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = get_scheduler(optimizer, SCHEDULER_TYPE)
        criterion = nn.BCELoss()

        start_epoch = 0
        try:
            model, optimizer, start_epoch = safe_load_checkpoint(model, optimizer, CHECKPOINT_DIR, fold)
            print(f"Resuming from epoch {start_epoch}")
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch")

        loss_history, f1_history = [], []
        best_f1, best_epoch = 0, 0
        stopped_early = False

        for epoch in trange(start_epoch, EPOCHS, desc=f"[Fold {fold}] CNN Training"):
            model.train()
            total_loss = 0
            for batch_X, batch_y in tqdm(train_loader, desc=f"[Fold {fold}] Batches"):
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
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
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
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

            wandb.log({
                "val/loss": val_loss,
                "val/f1": f1,
                "val/accuracy": acc,
                "val/precision": precision,
                "val/recall": recall,
            })

            write_log(log_file, f"[Epoch {epoch}] Loss: {val_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")

            loss_history.append(val_loss)
            f1_history.append(f1)

            print(f"[Fold {fold}][Epoch {epoch}] Loss: {val_loss:.4f} | F1: {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                save_checkpoint(model, optimizer, epoch, CHECKPOINT_DIR, fold, best_f1)
                write_log(log_file, f"[Epoch {epoch}] New best F1: {f1:.4f} (Saved model)")

                wandb.run.summary[f"fold_{fold}/best_f1"] = best_f1
                wandb.run.summary[f"fold_{fold}/best_epoch"] = best_epoch

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
            try:
                model, _, _ = safe_load_checkpoint(model, optimizer, CHECKPOINT_DIR, fold)
            except FileNotFoundError:
                print(f"[Fold {fold}] Aucun checkpoint trouvé; évaluation du test set annulée pour ce fold.")

            model.eval()
            y_test_true, y_test_pred, y_test_probs = [], [], []

            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                    outputs = model(batch_X).squeeze()
                    preds = (outputs >= 0.5).float()
                    y_test_true.extend(batch_y.tolist())
                    y_test_pred.extend(preds.cpu().tolist())
                    y_test_probs.extend(outputs.cpu().tolist())

            acc, f1, precision, recall = log_metrics(y_test_true, y_test_pred)
            try:
                auc = roc_auc_score(y_test_true, y_test_probs)
            except ValueError:
                auc = float('nan')
            balanced_acc = balanced_accuracy_score(y_test_true, y_test_pred)

            print(f"[Fold {fold}] Test Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | AUC: {auc:.4f} | Balanced Acc: {balanced_acc:.4f}")
            write_log(log_file, f"\n[Fold {fold}] Test Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | AUC: {auc:.4f} | Balanced Acc: {balanced_acc:.4f}")

            wandb.log({
                f"test/f1": f1,
                f"test/accuracy": acc,
                f"test/precision": precision,
                f"test/recall": recall,
                f"test/auc": auc,
                f"test/balanced_accuracy": balanced_acc,
            })

            log_file.close()

    print("CNN Training and evaluation complete.")
    wandb.finish()


if __name__ == "__main__":
    import yaml
    with open("configs/config_train_cnn_fashion.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_train_cnn(config)
