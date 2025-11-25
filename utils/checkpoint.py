# utils/checkpoint.py
# Version: 1.0

import os
import torch
from torch.serialization import add_safe_globals
import numpy as np

def save_checkpoint(model, optimizer, epoch, checkpoint_dir, fold, best_f1):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"fold_{fold}_best_model.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_f1': best_f1,
    }, checkpoint_path)


def load_checkpoint(model, optimizer, checkpoint_dir, fold):
    checkpoint_path = os.path.join(checkpoint_dir, f"fold_{fold}_best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    add_safe_globals([np.core.multiarray.scalar])

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']

import shutil

def safe_load_checkpoint(model, optimizer, checkpoint_dir, fold):
    """
    Charge un checkpoint pour le fold donné.
    En cas d'erreur de taille (size mismatch), supprime automatiquement le dossier de checkpoints du fold.
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"fold_{fold}.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}")
        return model, optimizer, start_epoch

    except RuntimeError as e:
        print(f"\n🚨 RuntimeError during checkpoint loading:\n{e}\n")
        print(f"🧹 Removing incompatible checkpoint folder: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        raise RuntimeError(f"Checkpoint at {checkpoint_path} was incompatible and has been deleted. Please re-run training.")
