# train_hybrid_qcnn.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm, trange
import wandb

from models.hybrid_qcnn import HybridQCNNBinaryClassifier
from utils.checkpoint import save_checkpoint, safe_load_checkpoint
from utils.early_stopping import EarlyStopping
from utils.metrics import log_metrics
from data_loader.utils import load_dataset_by_name
from utils.scheduler import get_scheduler
from utils.visual import save_plots
from utils.logger import init_logger, write_log

def run_train_hybrid_qcnn(config):

    dataset_name = config["dataset"]["name"]
    base_exp_name = config.get("experiment_name", "default_exp")
    EXPERIMENT_NAME = f"{dataset_name}_{base_exp_name}"
    SAVE_DIR = os.path.join("engine/checkpoints", "hybrid_qcnn", EXPERIMENT_NAME)
    CHECKPOINT_DIR = os.path.join(SAVE_DIR, "folds")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = config["training"]["batch_size"]
    EPOCHS = config["training"]["epochs"]
    LR = config["training"]["learning_rate"]
    KFOLD = config["training"]["kfold"]
    PATIENCE = config["training"]["early_stopping"]
    SCHEDULER_TYPE = config.get("scheduler", None)
    IN_CHANNELS = config['model']['in_channels']
    MODEL_HIDDEN_SIZES = config['model'].get('hidden_sizes', [])
    MODEL_CONV_CHANNELS = config['model'].get('conv_channels', None)
    QUANTUM_CFG = config.get("quantum", {})
    N_QUBITS = QUANTUM_CFG.get("n_qubits", 4)
    Q_LAYERS = QUANTUM_CFG.get("layers", 1)
    Q_BACKEND = QUANTUM_CFG.get("backend", "lightning.qubit")
    Q_SHOTS = QUANTUM_CFG.get("shots", None)
    Q_USE_GPU = QUANTUM_CFG.get("use_gpu", False)

    dataset_cfg = config.get("dataset", {})
    train_dataset, test_dataset = load_dataset_by_name(
        name=dataset_name,
        batch_size=BATCH_SIZE,
        binary_classes=dataset_cfg.get("binary_classes", [3, 8]),
        grayscale=dataset_cfg.get("grayscale", config.get("model", {}).get("grayscale"))
    )

    print(f"Nombre d'exemples chargés dans train_dataset : {len(train_dataset)}")

    kfold = KFold(n_splits=KFOLD, shuffle=True, random_state=42)

    def build_loader(dataset, shuffle=False):
        training_cfg = config.get("training", {})
        num_workers = training_cfg.get("num_workers", 0)
        pin_memory = training_cfg.get("pin_memory", False)
        prefetch_factor = training_cfg.get("prefetch_factor", None)
        persistent_workers = training_cfg.get("persistent_workers", False) if num_workers > 0 else False

        loader_kwargs = {
            "batch_size": BATCH_SIZE,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
        }
        if prefetch_factor is not None and num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(dataset, **loader_kwargs)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f"[Fold {fold}] Starting Hybrid QCNN training...")

        writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, f"fold_{fold}"))
        early_stopping = EarlyStopping(patience=PATIENCE)
        log_path, log_file = init_logger(os.path.join(SAVE_DIR, "logs"), fold)
        write_log(log_file, f"[Fold {fold}] Hybrid QCNN Training Log\n")

        train_loader = build_loader(Subset(train_dataset, train_idx), shuffle=True)
        val_loader = build_loader(Subset(train_dataset, val_idx))

        model = HybridQCNNBinaryClassifier(
            input_channel=IN_CHANNELS,
            n_qubits=N_QUBITS,
            n_layers=Q_LAYERS,
            backend=Q_BACKEND,
            shots=Q_SHOTS,
            use_gpu=Q_USE_GPU,
            conv_channels=MODEL_CONV_CHANNELS,
            hidden_sizes=MODEL_HIDDEN_SIZES,
        ).to(DEVICE)
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

        for epoch in trange(start_epoch, EPOCHS, desc=f"[Fold {fold}] Hybrid QCNN Training"):
            model.train()
            total_loss = 0
            for batch_X, batch_y in tqdm(train_loader, desc=f"[Fold {fold}] Batches"):
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            y_true, y_pred, y_probs = [], [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(DEVICE)
                    preds_logits = model(batch_X).squeeze()
                    preds = (preds_logits >= 0.5).float()
                    y_true.extend(batch_y.tolist())
                    y_pred.extend(preds.cpu().tolist())
                    y_probs.extend(preds_logits.cpu().tolist())

            acc, f1, precision, recall = log_metrics(y_true, y_pred)
            try:
                auc = roc_auc_score(y_true, y_probs)
            except ValueError:
                auc = 0.0
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            val_loss = total_loss / len(train_loader)

            writer.add_scalar("Loss/train", val_loss, epoch)
            writer.add_scalar("F1/val", f1, epoch)
            writer.add_scalar("Accuracy/val", acc, epoch)
            writer.add_scalar("Precision/val", precision, epoch)
            writer.add_scalar("Recall/val", recall, epoch)
            writer.add_scalar("BalancedAccuracy/val", bal_acc, epoch)
            writer.add_scalar("AUC/val", auc, epoch)

            wandb.log({
                "val/loss": val_loss,
                "val/f1": f1,
                "val/accuracy": acc,
                "val/precision": precision,
                "val/recall": recall,
                "val/balanced_accuracy": bal_acc,
                "val/auc": auc,
            })

            write_log(
                log_file,
                f"[Epoch {epoch}] Loss: {val_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | "
                f"Balanced Accuracy: {bal_acc:.4f} | AUC: {auc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}"
            )

            loss_history.append(val_loss)
            f1_history.append(f1)

            print(
                f"[Fold {fold}][Epoch {epoch}] Loss: {val_loss:.4f} | F1: {f1:.4f} | "
                f"Acc: {acc:.4f}  | Balanced Accuracy: {bal_acc:.4f} | AUC: {auc:.4f}"
            )

            if f1 > best_f1:
                best_f1, best_epoch = f1, epoch
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

        save_plots(fold, loss_history, f1_history, os.path.join(SAVE_DIR, "plots"))
        writer.close()

        write_log(log_file, f"\n[Fold {fold}] Best F1: {best_f1:.4f} at epoch {best_epoch}")
        write_log(log_file, f"Training {'stopped early' if stopped_early else 'completed full'} {EPOCHS} epochs\n")

        if test_dataset is not None:
            print(f"[Fold {fold}] Loading best model and evaluating on test set...")
            try:
                model, _, _ = safe_load_checkpoint(model, optimizer, CHECKPOINT_DIR, fold)
            except FileNotFoundError:
                print(f"[Fold {fold}] Aucun checkpoint trouvé; évaluation du test set annulée pour ce fold.")
                continue

            model.eval()
            y_test_true, y_test_pred, y_test_probs = [], [], []
            test_loader = build_loader(test_dataset)
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(DEVICE)
                    preds_logits = model(batch_X).squeeze()
                    preds = (preds_logits >= 0.5).float()
                    y_test_true.extend(batch_y.tolist())
                    y_test_pred.extend(preds.cpu().tolist())
                    y_test_probs.extend(preds_logits.cpu().tolist())

            acc, f1, precision, recall = log_metrics(y_test_true, y_test_pred)
            try:
                auc = roc_auc_score(y_test_true, y_test_probs)
            except ValueError:
                auc = float('nan')
            bal_acc = balanced_accuracy_score(y_test_true, y_test_pred)

            print(
                f"[Fold {fold}] Test Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | "
                f"Recall: {recall:.4f} | AUC: {auc:.4f} | Balanced Accuracy: {bal_acc:.4f}"
            )
            write_log(
                log_file,
                f"\n[Fold {fold}] Test Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | "
                f"Recall: {recall:.4f} | AUC: {auc:.4f} | Balanced Accuracy: {bal_acc:.4f}"
            )

            wandb.log({
                f"test/f1": f1,
                f"test/accuracy": acc,
                f"test/precision": precision,
                f"test/recall": recall,
                f"test/auc": auc,
                f"test/balanced_accuracy": bal_acc,
            })

            log_file.close()

    print("Hybrid QCNN training and evaluation complete.")
