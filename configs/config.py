import yaml
import os

FIELD_RULES = {
    "training": {
        "epochs": lambda x: isinstance(x, int) and x >= 1,
        "batch_size": lambda x: isinstance(x, int) and x > 0,
        "learning_rate": lambda x: isinstance(x, float) and 0 < x <= 1,
        "kfold": lambda x: isinstance(x, int) and x >= 2,
        "early_stopping": lambda x: isinstance(x, int) and x >= 0,
    },
    "model": {
        "input_size": lambda x: isinstance(x, int) and x > 0,
    },
    "dataset": {
        "name": lambda x: isinstance(x, str),
        "binary_classes": lambda x: isinstance(x, list) and all(isinstance(c, int) for c in x),
        "in_channels": lambda x: isinstance(x, int) and x > 0,
    },
    "checkpoint": {
        "save_dir": lambda x: isinstance(x, str),
        "subdir": lambda x: isinstance(x, str),
    }
}


def load_config(path="config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier de configuration introuvable : {path}")

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Erreur de parsing YAML dans '{path}': {e}")

    if not isinstance(config, dict):
        raise ValueError("Le fichier YAML doit contenir un dictionnaire en racine.")

    return config
