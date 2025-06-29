#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Script : train_all_cifar10.sh
# But : entraîner tous les modèles pertinents sur CIFAR-10 avec main.py
# et enregistrer les logs dans le dossier logs_cifar10/

mkdir -p logs_cifar10

echo "=============================="
echo "Training Classical MLP..."
echo "=============================="
python main.py --model classical --config configs/config_train_classical_cifar10.yaml | tee logs_cifar10/classical.log

echo "=============================="
echo "Training CNN..."
echo "=============================="
python main.py --model cnn --config configs/config_train_cnn_cifar10.yaml | tee logs_cifar10/cnn.log

echo "=============================="
echo "Training Quantum MLP..."
echo "=============================="
python main.py --model quantum_mlp --config configs/config_train_quantum_mlp_cifar10.yaml | tee logs_cifar10/quantum_mlp.log

echo "=============================="
echo "Training Hybrid QCNN..."
echo "=============================="
python main.py --model hybrid_qcnn --config configs/config_train_qcnn_cifar10.yaml | tee logs_cifar10/hybrid_qcnn.log

echo "=============================="
echo "Training Hybrid QCNN + SVM..."
echo "=============================="
python main.py --model hybrid_qcnn_svm --config configs/config_train_hybrid_qcnn_svm_cifar10.yaml | tee logs_cifar10/hybrid_qcnn_svm.log

echo "=============================="
echo "Training Classical SVM..."
echo "=============================="
python main.py --model svm --config configs/config_train_svm_cifar10.yaml --optimize | tee logs_cifar10/svm.log

echo "=============================="
echo "All trainings on CIFAR-10 finished!"
