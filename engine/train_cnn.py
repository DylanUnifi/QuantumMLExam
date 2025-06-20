# train_cnn.py
# Version: 2.1

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from models.cnn import CNNBinaryClassifier
from trainer import train_one_epoch, evaluate_model
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.scheduler import get_scheduler
from utils.visual import save_plots
import numpy as np

def main(config=None):
    if config is None:
        raise ValueError("Configuration must be provided via YAML or dictionary")

    SAVE_DIR = os.path.join(config['checkpoint']['save_dir'], 'cnn')
    CHECKPOINT_DIR = os.path.join(SAVE_DIR, config['checkpoint']['subdir'])
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = config['training']['batch_size']
    EPOCHS = config['training']['epochs']
    LR = config['training']['lr']
    KFOLD = config['training']['kfold']
    SCHEDULER_TYPE = config['training']['scheduler']
    EARLY_STOPPING = config['training']['early_stopping']

    IN_CHANNELS = config['model']['in_channels']
    INPUT_SHAPE = (IN_CHANNELS, 28, 28)  # fix or dynamically load from dataset

    # Dummy dataset (to be replaced with loader)
    X = np.random.rand(500, *INPUT_SHAPE).astype(np.float32)
    y = np.random.randint(0, 2, size=(500,)).astype(np.float32)

    kfold = KFold(n_splits=KFOLD, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"[Fold {fold}] Starting CNN training...")

        writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, f"fold_{fold}"))

        train_dataset = TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y[train_idx]))
        val_dataset = TensorDataset(torch.tensor(X[val_idx]), torch.tensor(y[val_idx]))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        model = CNNBinaryClassifier(in_channels=IN_CHANNELS).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = get_scheduler(optimizer, SCHEDULER_TYPE)
        criterion = nn.BCELoss()

        start_epoch, best_f1, best_epoch = 0, 0, 0
        try:
            model, optimizer, start_epoch = load_checkpoint(model, optimizer, CHECKPOINT_DIR, fold)
        except FileNotFoundError:
            print(f"No checkpoint found for fold {fold}, starting from scratch.")

        loss_history, f1_history = [], []
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, acc, f1, precision, recall = evaluate_model(model, val_loader, criterion, DEVICE)

            print(f"[Fold {fold}][Epoch {epoch}] Train Loss: {train_loss:.4f} | Val F1: {f1:.4f}")

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("F1/val", f1, epoch)
            writer.add_scalar("Accuracy/val", acc, epoch)

            loss_history.append(train_loss)
            f1_history.append(f1)

            if scheduler: scheduler.step()

            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                save_checkpoint(model, optimizer, epoch, CHECKPOINT_DIR, fold)

            if epoch - best_epoch >= EARLY_STOPPING:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        save_plots(fold, loss_history, f1_history, SAVE_DIR)
        writer.close()

    print("CNN Training finished.")
