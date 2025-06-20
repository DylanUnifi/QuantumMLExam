# train_svm.py
# Version: 2.0 - Intègre EnhancedSVM

import os
import numpy as np
import optuna
import joblib
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils.data import get_dataloaders
from models.svm_extension import EnhancedSVM


def objective(trial, X_train, y_train, X_val, y_val):
    C = trial.suggest_float('C', 0.01, 10.0, log=True)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

    model = EnhancedSVM(C=C, kernel=kernel, gamma=gamma)
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_val, y_val)
    return 1.0 - metrics['f1']


def main(config):
    dataset_name = config['data']['name']
    selected_classes = config['data']['selected_classes']
    use_pca = config['svm']['use_pca']
    n_components = config['svm']['pca_components']
    save_path = config['svm']['save_path']

    loader, _ = get_dataloaders(dataset_name, batch_size=1024, selected_classes=selected_classes)
    X, y = loader.dataset.tensors
    X, y = X.numpy().reshape(len(X), -1), y.numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if use_pca:
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)
    else:
        pca = None

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    study = optuna.create_study()
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=30)
    best_params = study.best_params

    print("Best params:", best_params)

    model = EnhancedSVM(**best_params, use_pca=use_pca, pca_model=pca, save_path=save_path)
    model.fit(X_train, y_train)
    model.save()

    metrics = model.evaluate(X_val, y_val)
    print("Validation Metrics:", metrics)

    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("confusion_matrix.png")
    plt.close()
