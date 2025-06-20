# train_svm.py
# Version: 3.7 – Ajout loggeur, visualisation, Optuna, gestion PCA et export checkpoints

import os
import joblib
import optuna
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from models.svm_extension import EnhancedSVM
from utils.logger import init_logger, write_log
from data_loader.utils import load_dataset_by_name
from utils.metrics import log_metrics

# Charger config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Configuration expérimentale
EXPERIMENT_NAME = config.get("experiment_name", "default_experiment")
SAVE_DIR = os.path.join("checkpoints", "svm", EXPERIMENT_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)
log_path, log_file = init_logger(os.path.join(SAVE_DIR, "logs"), fold="svm")
write_log(log_file, f"[SVM] Training Log for experiment: {EXPERIMENT_NAME}\n")

# Chargement des données
loader, _ = load_dataset_by_name(
    name=config["dataset"],
    batch_size=2048,
    selected_classes=config.get("selected_classes", [3, 8]),
    return_tensor_dataset=True
)
X, y = loader.dataset.tensors
X = X.view(len(X), -1).numpy()
y = y.numpy()

# Prétraitement
scaler = StandardScaler()
X = scaler.fit_transform(X)

use_pca = config["svm"].get("use_pca", False)
n_components = config["svm"].get("pca_components", 50)

if use_pca:
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X)
    write_log(log_file, f"PCA applied with {n_components} components\n")
else:
    pca = None
    write_log(log_file, "PCA not applied\n")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fonction objectif pour Optuna

def objective(trial):
    C = trial.suggest_float('C', 0.01, 10.0, log=True)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

    model = EnhancedSVM(C=C, kernel=kernel, gamma=gamma)
    model.fit(X_train, y_train)
    metrics = model.evaluate(y_val=y_val, X_val=X_val)
    return 1.0 - metrics['f1']

# Optimisation
study = optuna.create_study()
study.optimize(objective, n_trials=30)
best_params = study.best_params
write_log(log_file, f"Best params found: {best_params}\n")

# Entraînement et sauvegarde
model = EnhancedSVM(**best_params, use_pca=use_pca, pca_model=pca, save_path=SAVE_DIR)
model.fit(X_train, y_train)
model.save()
write_log(log_file, "Best SVM model trained and saved\n")

# Évaluation
metrics = model.evaluate(X_val, y_val)
write_log(log_file, f"Validation Metrics: {metrics}\n")

# Confusion Matrix
y_pred = model.predict(X_val)
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
plt.close()

write_log(log_file, "Confusion matrix saved\n")
log_file.close()
print("SVM training and evaluation complete.")
