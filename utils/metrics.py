# utils/metrics.py
# Version: 2.0

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def log_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return acc, f1, precision, recall


def compute_metrics_dict(y_true, y_pred):
    acc, f1, precision, recall = log_metrics(y_true, y_pred)
    return {
        'accuracy': acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
