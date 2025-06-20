# train_hybrid_qcnn.py
# Version: 1.1 - Batch Quantum Circuit + Advanced Improvements

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from models.hybrid_qcnn import HybridQCNNBinaryClassifier
from trainer import train_one_epoch, evaluate_model
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.scheduler import get_scheduler
from utils.visual import save_plots, plot_quantum_circuit
from torch.utils.tensorboard import SummaryWriter
from utils.data import get_dataloaders
import numpy as np
import time

def main(config):
    SAVE_DIR = os.path.join(config['checkpoint']['save_dir'], 'hybrid_qcnn')
    CHECKPOINT_DIR = os.path.join(SAVE_DIR, config['checkpoint']['subdir'])
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = config['training']['batch_size']
    EPOCHS = config['training']['epochs']
    LR = config['training']['lr']
    KFOLD = config['training']['kfold']
    SCHEDULER_TYPE = config['training']['scheduler']
    EARLY_STOPPING = config['training']['early_stopping']
    INPUT_SIZE = config['model']['input_size']

    dataset_name = config['data']['name']
    selected_classes = config['data']['selected_classes']

    train_data, _ = get_dataloaders(dataset_name, BATCH_SIZE, selected_classes=selected_classes)
    X = train_data.dataset.tensors[0].view(train_data.dataset.tensors[0].shape[0], -1).numpy()
    y = train_data.dataset.tensors[1].numpy()

    kfold = KFold(n_splits=KFOLD, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"[Fold {fold}] Starting Hybrid QCNN training...")
        writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, f"fold_{fold}"))

        X_train = torch.tensor(X[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(1)

        train_loader = DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(torch.utils.data.TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

        model = HybridQCNNBinaryClassifier(INPUT_SIZE, device=DEVICE).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = get_scheduler(optimizer, SCHEDULER_TYPE)
        criterion = nn.BCELoss()

        start_epoch, best_f1, best_epoch = 0, 0, 0
        try:
            model, optimizer, start_epoch = load_checkpoint(model, optimizer, CHECKPOINT_DIR, fold)
        except FileNotFoundError:
            print(f"No checkpoint found for fold {fold}, starting from scratch.")

        loss_history, f1_history = [], []
        plot_quantum_circuit(model.qnn, path=os.path.join(SAVE_DIR, f"qnn_structure_fold_{fold}.png"))

        for epoch in range(start_epoch, EPOCHS):
            start_time = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, acc, f1, precision, recall = evaluate_model(model, val_loader, criterion, DEVICE)
            duration = time.time() - start_time

            print(f"[Fold {fold}][Epoch {epoch}] Time: {duration:.2f}s | Train Loss: {train_loss:.4f} | Val F1: {f1:.4f}")

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("F1/val", f1, epoch)
            writer.add_scalar("Accuracy/val", acc, epoch)
            writer.add_scalar("Precision/val", precision, epoch)
            writer.add_scalar("Recall/val", recall, epoch)

            loss_history.append(train_loss)
            f1_history.append(f1)

            if scheduler: scheduler.step()

            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                save_checkpoint(model, optimizer, epoch, CHECKPOINT_DIR, fold, best_f1)

            if epoch - best_epoch >= EARLY_STOPPING:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        save_plots(fold, loss_history, f1_history, SAVE_DIR)
        writer.close()

    print("Hybrid QCNN Training finished.")
