import yaml
import os

REQUIRED_FIELDS = {
    "model": dict,
    "training": dict,
    "data": dict,
    "checkpoint": dict
}

FIELD_RULES = {
    "training": {
        "epochs": lambda x: isinstance(x, int) and x >= 1,
        "batch_size": lambda x: isinstance(x, int) and x > 0,
        "lr": lambda x: isinstance(x, float) and 0 < x <= 1,
        "kfold": lambda x: isinstance(x, int) and x >= 2,
        "early_stopping": lambda x: isinstance(x, int) and x >= 0,
    },
    "model": {
        "input_size": lambda x: isinstance(x, int) and x > 0,
    },
    "data": {
        "name": lambda x: isinstance(x, str),
        "selected_classes": lambda x: isinstance(x, list) and all(isinstance(c, int) for c in x),
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

    # Vérification des sections principales
    for field, field_type in REQUIRED_FIELDS.items():
        if field not in config:
            raise KeyError(f"Champ obligatoire manquant dans la configuration : '{field}'")
        if not isinstance(config[field], field_type):
            raise TypeError(f"Le champ '{field}' doit être de type {field_type.__name__}")

    # Règles spécifiques par section
    for section, rules in FIELD_RULES.items():
        for key, check in rules.items():
            value = config[section].get(key)
            if value is None or not check(value):
                raise ValueError(f"Valeur invalide ou manquante pour '{section}.{key}': {value}")

    return config
