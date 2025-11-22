# train_quantum_mlp.py
# Version: 1.3 – Ajout balanced accuracy, AUC, logs complets cohérents avec train_classical.

import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
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
    dataset_name = config["dataset"]["name"]
    base_exp_name = config.get("experiment_name", "default_exp")
    EXPERIMENT_NAME = f"{dataset_name}_{base_exp_name}"

    wandb.init(
        project="qml_project",
        name=EXPERIMENT_NAME,
        config=config
    )

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

    dataset_cfg = config.get("dataset", {})
    train_dataset, test_dataset = load_dataset_by_name(
        name=dataset_cfg.get("name"),
        batch_size=BATCH_SIZE,
        binary_classes=dataset_cfg.get("binary_classes", [0, 1]),
        grayscale=dataset_cfg.get("grayscale", config.get("model", {}).get("grayscale"))
    )

    indices = torch.randperm(len(train_dataset))[:500]
    train_dataset = Subset(train_dataset, indices)

    print(f"Nombre d'exemples chargés dans train_dataset : {len(train_dataset)}")

    kfold = KFold(n_splits=KFOLD, shuffle=True, random_state=42)

    def build_loader(dataset, shuffle=False, drop_last=False):
        training_cfg = config.get("training", {})
        num_workers = training_cfg.get("num_workers", 0)
        pin_memory = training_cfg.get("pin_memory", False)
        prefetch_factor = training_cfg.get("prefetch_factor", None)
        persistent_workers = training_cfg.get("persistent_workers", False) if num_workers > 0 else False

        loader_kwargs = {
            "batch_size": BATCH_SIZE,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
        }
        if prefetch_factor is not None and num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(dataset, **loader_kwargs)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f"[Fold {fold}] Starting Quantum MLP training...")

        writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, f"fold_{fold}"))
        early_stopping = EarlyStopping(patience=PATIENCE)

        log_path, log_file = init_logger(os.path.join(SAVE_DIR, "logs"), fold)
        write_log(log_file, f"[Fold {fold}] Quantum MLP Training Log\n")

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = build_loader(train_subset, shuffle=True, drop_last=True)
        val_loader = build_loader(val_subset)

        sample_X, _ = train_dataset[0]
        input_size = sample_X.numel()

        model_hidden_sizes = config.get("model", {}).get("hidden_sizes", None)
        quantum_cfg = config.get("quantum", {})
        use_gpu = quantum_cfg.get("use_gpu", False)
        model = QuantumResidualMLP(
            input_size=input_size,
            hidden_sizes=model_hidden_sizes,
            n_qubits=quantum_cfg.get("n_qubits", 4),
            n_layers=quantum_cfg.get("layers", 2),
            backend=quantum_cfg.get("backend", "lightning.qubit"),
            shots=quantum_cfg.get("shots", None),
            use_gpu=use_gpu,
        ).to(DEVICE)
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
            y_true, y_pred, y_probs = [], [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.view(batch_X.size(0), -1).to(DEVICE)
                    preds = model(batch_X).squeeze()
                    probs = preds.detach().cpu().numpy()
                    preds = (preds >= 0.5).float()
                    y_true.extend(batch_y.tolist())
                    y_pred.extend(preds.cpu().tolist())
                    y_probs.extend(probs.tolist())

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

            write_log(
                log_file,
                f"[Epoch {epoch}] Loss: {val_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | "
                f"Balanced Accuracy: {bal_acc:.4f} | AUC: {auc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}"
            )

            loss_history.append(val_loss)
            f1_history.append(f1)

            wandb.log({
                "val/loss": val_loss,
                "val/f1": f1,
                "val/accuracy": acc,
                "val/precision": precision,
                "val/recall": recall,
                "val/balanced_accuracy": bal_acc,
                "val/auc": auc,
            })

            print(
                f"[Fold {fold}][Epoch {epoch}] Loss: {val_loss:.4f} | F1: {f1:.4f} | "
                f"Acc: {acc:.4f} | Balanced Accuracy: {bal_acc:.4f} | AUC: {auc:.4f}"
            )

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

        if test_dataset is not None:
            print(f"[Fold {fold}] Loading best model and evaluating on test set...")
            try:
                model, _, _ = load_checkpoint(model, optimizer, CHECKPOINT_DIR, fold)
            except FileNotFoundError:
                print(f"[Fold {fold}] Aucun checkpoint trouvé; évaluation du test set annulée pour ce fold.")
                continue

            model.eval()
            y_test_true, y_test_pred, y_test_probs = [], [], []

            test_loader = build_loader(test_dataset)
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.view(batch_X.size(0), -1).to(DEVICE), batch_y.to(DEVICE)
                    preds = model(batch_X).squeeze()
                    probs = preds.detach().cpu().numpy()
                    preds = (preds >= 0.5).float()
                    y_test_true.extend(batch_y.tolist())
                    y_test_pred.extend(preds.cpu().tolist())
                    y_test_probs.extend(probs.tolist())

            acc, f1, precision, recall = log_metrics(y_test_true, y_test_pred)
            try:
                auc = roc_auc_score(y_test_true, y_test_probs)
            except ValueError:
                auc = 0.0
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
                f"test/balanced_accuracy": bal_acc,
                f"test/auc": auc,
            })

            log_file.close()

    print("Quantum MLP training complete.")
    wandb.finish()


if __name__ == "__main__":
    import yaml
    with open("configs/config_train_quantum_mlp_fashion.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_train_quantum_mlp(config)
