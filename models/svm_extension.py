# svm_extension.py
# Version: 1.0 - Extension personnalisée de sklearn.SVC avec hooks GPU/monitoring

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import torch
import numpy as np

class EnhancedSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', use_pca=False, pca_model=None, save_path=None):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.use_pca = use_pca
        self.pca_model = pca_model
        self.save_path = save_path or './enhanced_svm.pkl'
        self.model = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma)

    def fit(self, X, y):
        if self.use_pca and self.pca_model:
            X = self.pca_model.transform(X)

        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.use_pca and self.pca_model:
            X = self.pca_model.transform(X)
        return self.model.predict(X)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted')
        }

    def save(self):
        joblib.dump(self, self.save_path)

    @staticmethod
    def load(path):
        return joblib.load(path)

    def to_torch_tensor(self, X):
        return torch.tensor(X, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(X)

    def predict_torch(self, X):
        X_torch = self.to_torch_tensor(X)
        X_cpu = X_torch.cpu().numpy()
        return self.predict(X_cpu)
