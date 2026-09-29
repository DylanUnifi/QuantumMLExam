# engine/evaluate.py
import torch
from utils.metrics import compute_metrics_dict
from sklearn.metrics import confusion_matrix


def evaluate_model(model, loader, criterion, device, final=False):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X).squeeze()
            loss = criterion(output, y)
            total_loss += loss.item()
            all_targets.append(y)
            all_outputs.append(output)

    all_targets = torch.cat(all_targets)
    all_outputs = torch.cat(all_outputs)
    model_metrics = compute_metrics_dict(all_outputs, all_targets)
    avg_loss = total_loss / len(loader)

    if final:
        cm = confusion_matrix(all_targets.cpu(), (all_outputs > 0.5).cpu())
        print("Confusion Matrix:")
        print(cm)

    return avg_loss, model_metrics