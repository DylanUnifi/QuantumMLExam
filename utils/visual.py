# utils/visual.py
# Version: 2.0

import matplotlib.pyplot as plt
import os

def save_plots(fold, loss_history, f1_history, save_dir):
    plt.figure()
    plt.plot(loss_history, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Train Loss - Fold {fold}')
    plt.savefig(os.path.join(save_dir, f'loss_fold_{fold}.png'))
    plt.close()

    plt.figure()
    plt.plot(f1_history, label='Val F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title(f'Validation F1 Score - Fold {fold}')
    plt.savefig(os.path.join(save_dir, f'f1_fold_{fold}.png'))
    plt.close()

def plot_quantum_circuit(qnode, filename='quantum_circuit.png'):
    try:
        drawer = qnode.qtape.draw(show_all_wires=True, decimals=2)
        with open(filename.replace('.png', '.txt'), 'w') as f:
            f.write(drawer)
        print(f"Quantum circuit saved to {filename.replace('.png', '.txt')}")
    except Exception as e:
        print(f"Failed to save quantum circuit: {e}")
