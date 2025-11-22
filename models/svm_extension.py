import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import torch


class EnhancedSVM(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        C=1.0,
        kernel='rbf',
        gamma='scale',
        use_pca=False,
        pca_model=None,
        scaler=None,
        save_path=None,
        probability=False,
        auto_transform=True,
        use_gpu=False,
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.use_pca = use_pca
        self.pca_model = pca_model
        self.scaler = scaler
        self.save_path = save_path or './enhanced_svm.pkl'
        self.auto_transform = auto_transform
        self.use_gpu = use_gpu
        self.probability = probability
        self.xp = None
        self.model = self._build_model()

    def _build_model(self):
        if self.use_gpu:
            try:
                import cupy as cp  # type: ignore
                from cuml.svm import SVC as cuSVC  # type: ignore

                self.xp = cp
                return cuSVC(C=self.C, kernel=self.kernel, gamma=self.gamma, probability=self.probability)
            except Exception as e:
                print(f"[Warning] GPU SVM unavailable ({e}), falling back to CPU sklearn SVC.")

        import numpy as np
        self.xp = np
        return SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, probability=self.probability)

    def _transform_input(self, X):
        if not self.auto_transform:
            return X
        X_transformed = X
        if self.scaler is not None:
            X_transformed = self.scaler.transform(X_transformed)
        if self.use_pca and self.pca_model is not None:
            X_transformed = self.pca_model.transform(X_transformed)
        if self.use_gpu and self.xp is not None:
            try:
                X_transformed = self.xp.asarray(X_transformed)
            except Exception as e:
                print(f"[Warning] Could not move data to GPU ({e}); continuing on CPU.")
        return X_transformed

    def fit(self, X, y):
        X_transformed = self._transform_input(X)
        self.model.fit(X_transformed, y)
        return self

    def predict(self, X):
        X_transformed = self._transform_input(X)
        preds = self.model.predict(X_transformed)
        if hasattr(preds, "get"):
            preds = preds.get()
        return preds

    def predict_proba(self, X):
        """
        Retourne les probabilités (uniquement si probability=True au fit)
        """
        X_transformed = self._transform_input(X)
        probs = self.model.predict_proba(X_transformed)
        if hasattr(probs, "get"):
            probs = probs.get()
        return probs

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
        }

        try:
            y_proba = self.predict_proba(X)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except Exception as e:
            print(f"[Warning] Could not compute ROC AUC: {e}")
            metrics['roc_auc'] = float('nan')

        return metrics

    def save(self):
        os.makedirs(self.save_path, exist_ok=True)
        model_path = os.path.join(self.save_path, "svm_model.pkl")
        joblib.dump(self, model_path)
        print(f"✅ Modèle sauvegardé avec succès : {model_path}")

    @staticmethod
    def load(path):
        return joblib.load(path)

    def to_torch_tensor(self, X):
        return torch.tensor(X, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(X)

    def predict_torch(self, X):
        X_torch = self.to_torch_tensor(X)
        X_cpu = X_torch.cpu().numpy()
        return self.predict(X_cpu)
