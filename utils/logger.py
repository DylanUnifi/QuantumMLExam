# utils/logger.py
# Version: 1.0 — Centralisation du logging pour les expériences

import os
from datetime import datetime

def init_logger(log_dir, fold=None):
    """
    Initialise un fichier de log dans le répertoire donné.
    Si `fold` est précisé, crée un fichier spécifique à ce fold.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"log_fold_{fold}.txt" if fold is not None else f"log_{timestamp}.txt"
    log_path = os.path.join(log_dir, log_filename)
    log_file = open(log_path, "a")
    return log_path, log_file

def write_log(log_file, message):
    """
    Écrit un message dans le fichier de log avec un saut de ligne.
    """
    log_file.write(message + "\n")
    log_file.flush()

def close_logger(log_file):
    """
    Ferme proprement le fichier de log.
    """
    log_file.close()
