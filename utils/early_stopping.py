# utils/early_stopping.py
# Version: 1.0

class EarlyStopping:
    def __init__(self, patience=10, mode='max', min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == 'max':
            improvement = current_score - self.best_score
        else:
            improvement = self.best_score - current_score

        if improvement > self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
