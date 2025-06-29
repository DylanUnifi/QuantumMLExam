import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
import wandb

from data_loader.utils import load_dataset_by_name
from models.hybrid_qcnn import HybridQCNNFeatures
from utils.metrics import log_metrics
from utils.logger import init_logger, write_log
from utils.scheduler import get_scheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_feature_extractor(model, classifier, train_loader, optimizer, criterion):
    model.train()
    classifier.train()
    total_loss = 0
    for batch_X, batch_y in tqdm(train_loader, desc="Training QCNN"):
        batch_X, batch_y = batch_X.view(batch_X.size(0), -1).to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        feats = model(batch_X)
        outputs = classifier(feats).squeeze()  # Pas de sigmoïde ici !
        loss = criterion(outputs, batch_y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def extract_features(model, loader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in tqdm(loader, desc="Extracting features"):
            batch_X = batch_X.view(batch_X.size(0), -1).to(DEVICE)
            feats = model(batch_X).cpu().numpy()  # Features directement depuis le feature extractor
            features.append(feats)
            labels.append(batch_y.cpu().numpy())
    X = np.vstack(features)
    y = np.concatenate(labels)
    return X, y

def run_train_hybrid_qcnn_svm(config):
    EXPERIMENT_NAME = config.get("experiment_name", "hybrid_qcnn_svm_advanced_exp")
    SAVE_DIR = os.path.join("engine/checkpoints", "hybrid_qcnn_svm", EXPERIMENT_NAME)
    os.makedirs(SAVE_DIR, exist_ok=True)

    wandb.init(
        project="qml_project",
        name=EXPERIMENT_NAME,
        config=config
    )

    train_dataset, test_dataset = load_dataset_by_name(
        name=config["dataset"]["name"],
        batch_size=config["training"]["batch_size"],
        binary_classes=config.get("binary_classes", [3, 8])
    )

    indices = torch.randperm(len(train_dataset))[:500]
    train_dataset = Subset(train_dataset, indices)

    print(f"Nombre d'exemples chargés dans train_dataset : {len(train_dataset)}")

    kfold = KFold(n_splits=config["training"]["kfold"], shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f"[Fold {fold}] Starting Hybrid QCNN + SVM training...")
        log_path, log_file = init_logger(os.path.join(SAVE_DIR, "logs"), fold)

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config["training"]["batch_size"])

        sample_X, _ = train_dataset[0]
        input_size = sample_X.numel()

        feature_extractor = HybridQCNNFeatures(input_size=input_size).to(DEVICE)
        feats_dim = feature_extractor(torch.randn(1, input_size).to(DEVICE)).shape[1]
        classifier = nn.Linear(feats_dim, 1).to(DEVICE)

        params = list(feature_extractor.parameters()) + list(classifier.parameters())
        optimizer = optim.Adam(params, lr=config["training"]["learning_rate"])
        scheduler = get_scheduler(optimizer, config.get("scheduler", None))
        criterion = nn.BCEWithLogitsLoss()  # ✅ Utilisation de BCEWithLogitsLoss

        best_f1, best_epoch = 0, 0

        for epoch in range(config["training"]["epochs"]):
            val_loss = train_feature_extractor(feature_extractor, classifier, train_loader, optimizer, criterion)
            X_val, y_val = extract_features(feature_extractor, val_loader)

            scaler = StandardScaler()
            X_val_scaled = scaler.fit_transform(X_val)
            svm = SVC(C=1.0, kernel='rbf')
            svm.fit(X_val_scaled, y_val)
            y_pred = svm.predict(X_val_scaled)

            acc, f1, precision, recall = log_metrics(y_val, y_pred)

            wandb.log({
                "vak/loss": val_loss,
                "val/f1": f1,
                "val/accuracy": acc,
                "val/precision": precision,
                "val/recall": recall,
            })

            write_log(log_file,
                      f"[Epoch {epoch}] Loss: {val_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")
            print(f"[Fold {fold}][Epoch {epoch}] Loss: {val_loss:.4f} | F1: {f1:.4f}")

            if f1 > best_f1:
                best_f1, best_epoch = f1, epoch
                checkpoint = {
                    "feature_extractor_state_dict": feature_extractor.state_dict(),
                    "classifier_state_dict": classifier.state_dict(),
                    "scaler": scaler,
                    "svm": svm
                }
                save_path = os.path.join(SAVE_DIR, f"hybrid_qcnn_svm_fold_{fold}.pt")
                torch.save(checkpoint, save_path)
                print(f"✅ Checkpoint saved at {save_path}")
                write_log(log_file, f"[Epoch {epoch}] New best F1: {f1:.4f} (Saved model)")

            if scheduler:
                scheduler.step()

        wandb.run.summary[f"fold_{fold}/best_f1"] = best_f1
        wandb.run.summary[f"fold_{fold}/best_epoch"] = best_epoch

        write_log(log_file, f"\n[Fold {fold}] Best F1: {best_f1:.4f} at epoch {best_epoch}")
        log_file.close()

        # Évaluation finale sur test set
        if test_dataset is not None:
            print("\n🔎 Loading best feature extractor and evaluating on test set...")
            trainval_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"])
            X_trainval, y_trainval = extract_features(feature_extractor, trainval_loader)

            test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"])
            X_test, y_test = extract_features(feature_extractor, test_loader)

            scaler = StandardScaler()
            X_trainval_scaled = scaler.fit_transform(X_trainval)
            X_test_scaled = scaler.transform(X_test)

            best_svm = SVC(C=1.0, kernel='rbf')
            best_svm.fit(X_trainval_scaled, y_trainval)
            y_test_pred = best_svm.predict(X_test_scaled)

            acc, f1, precision, recall = log_metrics(y_test, y_test_pred)
            print(f"[Test Set] Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

            # 🔹 Rouvre un log global pour consigner l'évaluation du test set
            global_log_path = os.path.join(SAVE_DIR, "logs", "test_evaluation.log")
            with open(global_log_path, "a") as log_file:
                write_log(log_file,
                          f"\n[Test Set] Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

            wandb.log({
                f"test/svm_f1": f1,
                f"test/svm_accuracy": acc,
                f"test/svm_precision": precision,
                f"test/svm_recall": recall,
            })

            # 🔹 Sauvegarde le modèle SVM et le scaler sur le train complet
            final_checkpoint = {
                "feature_extractor_state_dict": feature_extractor.state_dict(),
                "scaler": scaler,
                "svm": best_svm
            }
            final_model_path = os.path.join(SAVE_DIR, "hybrid_qcnn_svm_final.pt")
            torch.save(final_checkpoint, final_model_path)
            print(f"✅ Final checkpoint saved at {final_model_path}")

            import pennylane as qml
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

            # Nombre de qubits et device (tu peux changer "lightning.qubit" dans ton YAML)
            n_qubits = config["quantum"]["n_qubits"]
            dev = qml.device(config["quantum"]["backend"], wires=n_qubits)

            # QNode pour définir le quantum kernel
            @qml.qnode(dev)
            def kernel_qnode(x1, x2):
                # Encoding de x1
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                    qml.RZ(x1[i], wires=i)
                # Encoding inverse de x2
                for i in range(n_qubits):
                    qml.RZ(-x2[i], wires=i)
                    qml.Hadamard(wires=i)
                return qml.probs(wires=range(n_qubits))

            # Fonction pour calculer le kernel quantique entre deux vecteurs
            def quantum_kernel(x1, x2):
                return kernel_qnode(x1, x2)[0]  # Prend la première probabilité

            # Fonction pour générer la matrice de kernel
            def compute_quantum_kernel_matrix(X1, X2):
                n1, n2 = len(X1), len(X2)
                K = np.zeros((n1, n2))
                for i in tqdm(range(n1), desc="Computing Quantum Kernel Matrix"):
                    for j in range(n2):
                        K[i, j] = quantum_kernel(X1[i], X2[j])
                return K

            # Limiter le nombre d'exemples pour le calcul du kernel QSVM
            max_kernel_samples = 100  # 👈 adapte si besoin

            n_samples = min(max_kernel_samples, len(X_trainval))
            indices_train = np.random.choice(len(X_trainval), n_samples, replace=False)
            indices_test = np.random.choice(len(X_test), n_samples, replace=False)

            X_trainval_q = X_trainval[indices_train]
            y_trainval_q = y_trainval[indices_train]
            X_test_q = X_test[indices_test]
            y_test_q = y_test[indices_test]

            print(f"\n🔎 Subsampled to {n_samples} examples for QSVM kernel computation.")
            print(f"🔎 Using {len(X_trainval_q)} train examples and {len(X_test_q)} test examples for QSVM kernel.")

            print("\n🔎 QSVM: Calculating quantum kernel matrices...")
            K_train = compute_quantum_kernel_matrix(X_trainval_q, X_trainval_q)
            K_test = compute_quantum_kernel_matrix(X_test_q, X_trainval_q)

            print("\n🔎 QSVM: Training final QSVM...")
            qsvm = SVC(kernel="precomputed")
            qsvm.fit(K_train, y_trainval_q)

            print("\n🔎 QSVM: Evaluating on test set...")
            y_test_pred = qsvm.predict(K_test)

            acc = accuracy_score(y_test_q, y_test_pred)
            f1 = f1_score(y_test_q, y_test_pred, average="binary")
            precision = precision_score(y_test_q, y_test_pred, average="binary")
            recall = recall_score(y_test_q, y_test_pred, average="binary")

            print(f"[QSVM Test Set] Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

            # 🔹 Logs dans wandb pour comparaison directe avec Hybrid QCNN + SVM
            wandb.log({
                "test/qsvm_accuracy": acc,
                "test/qsvm_f1": f1,
                "test/qsvm_precision": precision,
                "test/qsvm_recall": recall
            })
    wandb.finish()

    print("Hybrid QCNN + SVM training complete.")


if __name__ == "__main__":
    import yaml
    with open("configs/config_train_hybrid_qcnn_svm_fashion.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_train_hybrid_qcnn_svm(config)
