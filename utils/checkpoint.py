# utils/checkpoint.py
# Version: 1.0

import os
import torch

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
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']