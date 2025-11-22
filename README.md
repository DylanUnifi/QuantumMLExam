# Quantum Machine Learning: Binary Image Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository compares classical and quantum approaches for **binary image classification (classes 3 vs 8)** on Fashion-MNIST, SVHN, and CIFAR-10. Models include classical MLPs and CNNs, hybrid quantum CNNs, quantum residual MLPs, and SVM baselines (standard and quantum-kernel variants). The goal is to study performance, trainability, and optimization challenges highlighted in *Challenges and Opportunities in Quantum Machine Learning* (Cerezo et al., 2023).

## 📂 Datasets
- **Fashion-MNIST**: 28×28 grayscale (Dress vs Bag)
- **SVHN**: 32×32 RGB (Digits 3 vs 8)
- **CIFAR-10**: 32×32 RGB (Cat vs Ship)

Datasets download automatically via `torchvision.datasets` during training.

## 🧩 Models & Features
- **Classical MLP / CNN**: configurable channel and hidden-layer depths, residual blocks, dropout, and metric logging (accuracy, balanced accuracy, precision/recall/F1, AUC).
- **Hybrid QCNN**: convolutional stem feeding a PennyLane quantum layer with configurable qubits, layers, and differentiable backends (CPU or GPU lightning).
- **Quantum Residual MLP**: classical embedding stack followed by a batch-aware quantum layer using PennyLane TorchLayer.
- **SVM & Quantum-Kernel SVM**: scikit-learn SVC with optional Optuna search plus a precomputed-kernel variant driven by PennyLane state kernels; both support optional GPU acceleration and PCA/scaling persistence.
- **Training infrastructure**: YAML-driven configs, k-fold support, TensorBoard and Weights & Biases logging, checkpointing under `engine/checkpoints/`.

## 🚀 How to Run
Clone and set up the environment:
```bash
git clone https://github.com/DylanUnifi/QuantumMLExam.git
cd QuantumMLExam
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Docker / Docker Compose
Build and run the project without installing dependencies locally. CPU-only image:
```bash
docker build -t quantum-ml-exam .
docker run --rm quantum-ml-exam
```

Use Docker Compose for repeatable runs and persisted outputs:
```bash
# CPU (default)
docker compose up --build trainer

# GPU (requires NVIDIA Container Toolkit) — uses the CUDA base image and cu-enabled torch by default
docker compose --profile gpu up --build trainer-gpu

# Override the command
docker compose run trainer python main.py --model hybrid_qcnn --config configs/config_train_hybrid_qcnn_fashion.yaml
```
Each YAML controls dataset settings (grayscale vs RGB, class subset), model hyperparameters (channels/hidden sizes, quantum qubits/layers, backend, shots), training knobs (epochs, early stopping), and data-loader performance flags (workers, pin memory, prefetch).

GPU images rely on the `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` base with CUDA wheels (`TORCH_SPEC`/`TORCH_INDEX_URL` build args). Customize these args in `docker-compose.yml` or via the command line to target different CUDA versions.

### Unified entrypoint
Use `main.py` to launch any model with its config:
```bash
python main.py --model classical_mlp --config configs/config_train_classical_mlp_fashion.yaml
python main.py --model cnn --config configs/config_train_cnn_svhn.yaml
python main.py --model hybrid_qcnn --config configs/config_train_hybrid_qcnn_cifar10.yaml
python main.py --model quantum_mlp --config configs/config_train_quantum_mlp_fashion.yaml
python main.py --model svm --config configs/config_train_svm_fashion.yaml --optimize
python main.py --model svm_qkernel --config configs/config_train_svm_qkernel_svhn.yaml
```
Each YAML controls dataset settings (grayscale vs RGB, class subset), model hyperparameters (channels/hidden sizes, quantum qubits/layers, backend, shots), training knobs (epochs, early stopping), and data-loader performance flags (workers, pin memory, prefetch).

### Direct script invocation
Individual trainers remain available, e.g.:
```bash
python train_hybrid_qcnn.py --config configs/config_train_hybrid_qcnn_svhn.yaml
python train_quantum_mlp.py --config configs/config_train_quantum_mlp_cifar10.yaml
```

## 📊 Example Results (mean over folds)
- **Classical MLP**: Fashion-MNIST F1 ≈ 0.987 / AUC ≈ 0.998; SVHN F1 ≈ 0.800 / AUC ≈ 0.924; CIFAR-10 F1 ≈ 0.798 / AUC ≈ 0.887.
- **Quantum MLP**: Fashion-MNIST F1 ≈ 0.639 (instability on one fold); SVHN F1 ≈ 0.852 (Recall ≈ 0.99, Precision ≈ 0.75); CIFAR-10 F1 ≈ 0.67–0.70 with higher variance.

These results illustrate the stability of classical models and the optimization sensitivity of quantum approaches on harder datasets.

## 📝 Outputs
Training logs, TensorBoard traces, checkpoints, confusion matrices, and (for quantum kernels) kernel heatmaps are stored under `outputs/` and `engine/checkpoints/`. GPU usage, backend selection, and data-loader performance settings are all logged for reproducibility.

## 📜 License
This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 🙋 Support
Open an issue on GitHub or email [dylan.fouepe@edu.unifi.it](mailto:dylan.fouepe@edu.unifi.it) for questions or contributions.
