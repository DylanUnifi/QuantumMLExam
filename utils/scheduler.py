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
        return None
    else:
        return None