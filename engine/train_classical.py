# train_classical.py
# Version: 3.4 – Logging intégré pour chaque fold

import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from models.classical import MLPBinaryClassifier
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.early_stopping import EarlyStopping
from utils.metrics import log_metrics
from data_loader.utils import load_dataset_by_name
from utils.scheduler import get_scheduler
from utils.visual import save_plots
import yaml

# Charger config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Configuration générale
SAVE_DIR = os.path.join("checkpoints", "classical")
CHECKPOINT_DIR = os.path.join(SAVE_DIR, "folds")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = config["batch_size"]
EPOCHS = config["epochs"]
LR = config["learning_rate"]
KFOLD = config["kfold"]
PATIENCE = config["early_stopping"]
SCHEDULER_TYPE = config.get("scheduler", None)

# Charger données en TensorDataset
train_loader, test_loader = load_dataset_by_name(
    name=config["dataset"],
    batch_size=BATCH_SIZE,
    selected_classes=config.get("selected_classes", [3, 8]),
    return_tensor_dataset=True
)
X = train_loader.dataset.tensors[0].view(train_loader.dataset.tensors[0].shape[0], -1)
y = train_loader.dataset.tensors[1].unsqueeze(1)

kfold = KFold(n_splits=KFOLD, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    print(f"[Fold {fold}] Starting training...")
    writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, f"fold_{fold}"))
    early_stopping = EarlyStopping(patience=PATIENCE)

    log_path = os.path.join(SAVE_DIR, f"fold_{fold}", "log.txt")
    with open(log_path, "w") as log_file:
        log_file.write(f"[Fold {fold}] Training Log\n\n")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_loader_fold = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = MLPBinaryClassifier(input_size=X.shape[1], hidden_size=config["hidden_size"]).to(DEVICE)
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

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader_fold:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE)
                preds = model(batch_X).squeeze()
                preds = (preds >= 0.5).float()
                y_true.extend(batch_y.tolist())
                y_pred.extend(preds.cpu().tolist())

        acc, f1, precision, recall = log_metrics(y_true, y_pred)
        val_loss = total_loss / len(train_loader_fold)

        writer.add_scalar("Loss/train", val_loss, epoch)
        writer.add_scalar("F1/val", f1, epoch)
        writer.add_scalar("Accuracy/val", acc, epoch)
        writer.add_scalar("Precision/val", precision, epoch)
        writer.add_scalar("Recall/val", recall, epoch)

        with open(log_path, "a") as log_file:
            log_file.write(f"[Epoch {epoch}] Loss: {val_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}\n")

        loss_history.append(val_loss)
        f1_history.append(f1)

        print(f"[Fold {fold}][Epoch {epoch}] Loss: {val_loss:.4f} | F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, CHECKPOINT_DIR, fold, best_f1)
            with open(log_path, "a") as log_file:
                log_file.write(f"[Epoch {epoch}] New best F1: {f1:.4f} (Saved model)\n")

        if early_stopping(f1):
            print("Early stopping triggered.")
            with open(log_path, "a") as log_file:
                log_file.write(f"Early stopping triggered at epoch {epoch}\n")
            break

        if scheduler:
            scheduler.step()

    save_plots(fold, loss_history, f1_history, SAVE_DIR)
    writer.close()

    # Évaluation finale sur test set
    if test_loader is not None:
        print(f"[Fold {fold}] Loading best model and evaluating on test set...")
        model, _, _ = load_checkpoint(model, optimizer, CHECKPOINT_DIR, fold)
        model.eval()
        y_test_true, y_test_pred = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.view(batch_X.size(0), -1).to(DEVICE)
                preds = model(batch_X).squeeze()
                preds = (preds >= 0.5).float()
                y_test_true.extend(batch_y.tolist())
                y_test_pred.extend(preds.cpu().tolist())

        acc, f1, precision, recall = log_metrics(y_test_true, y_test_pred)
        print(f"[Fold {fold}] Test Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        with open(log_path, "a") as log_file:
            log_file.write(f"\n[Fold {fold}] Test Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}\n")

print("Training and evaluation complete.")
