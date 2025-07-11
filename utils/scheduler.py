# utils/scheduler.py
# Version: 1.0

def get_scheduler(optimizer, name):
    if name:
        name= name.lower()
        if name == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(optimizer, T_max=50)
        elif name == "step":
            from torch.optim.lr_scheduler import StepLR
            return StepLR(optimizer, step_size=10, gamma=0.5)
        elif name == "exponential":
            from torch.optim.lr_scheduler import ExponentialLR
            return ExponentialLR(optimizer, gamma=0.95)
        elif name == "stepLR":
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            return ReduceLROnPlateau(
                                            optimizer,
                                            mode="max",          # car tu veux maximiser le F1
                                            factor=0.5,          # réduit LR de 50%
                                            patience=5,          # attend 5 epochs sans amélioration avant de réduire LR
                                            verbose=True,
                                            min_lr=1e-6          # LR minimal
                                        )
    return None