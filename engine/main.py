# main.py
# Version: 2.1 - Multi-model + HybridQCNN support

import argparse
import yaml
import datetime
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Quantum/Deep Learning Training Pipeline")
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    model_type = config['model']['type']

    # Créer run_id et dossiers
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config['run_id'] = run_id

    # Créer les répertoires nécessaires
    base_dirs = ['logs', 'checkpoints', 'runs']
    for d in base_dirs:
        os.makedirs(os.path.join(d, run_id), exist_ok=True)

    # Injection dans config (sous-objets aussi)
    if 'checkpoint' in config:
        config['checkpoint']['save_dir'] = os.path.join("checkpoints", run_id)
    if 'svm' in config:
        config['svm']['save_path'] = os.path.join("svm_models", run_id)


    if model_type == 'classical':
        from train_classical import main as train_classical
        train_classical(config)
    elif model_type == 'cnn':
        from train_cnn import main as train_cnn
        train_cnn(config)
    elif model_type == 'hybrid':
        from train_hybrid_qcnn import main as train_hybrid
        train_hybrid(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

if __name__ == '__main__':
    main()
