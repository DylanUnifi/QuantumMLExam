#!/bin/bash

# Script : train_all_fashion.sh
# But : entraîner tous les modèles pertinents sur Fashion-MNIST avec main.py
# et enregistrer les logs_fashion dans le dossier logs_fashion/

mkdir -p logs_fashion

echo "=============================="
echo "Training Classical MLP..."
echo "=============================="
python main.py --model classical_mlp --config configs/config_train_classical_mlp_fashion.yaml | tee logs_fashion/classical_mlp.log

echo "=============================="
echo "Training CNN..."
echo "=============================="
python main.py --model cnn --config configs/config_train_cnn_fashion.yaml | tee logs_fashion/cnn.log

echo "=============================="
echo "Training Quantum MLP..."
echo "=============================="
python main.py --model quantum_mlp --config configs/config_train_quantum_mlp_fashion.yaml | tee logs_fashion/quantum_mlp.log

echo "=============================="
echo "Training Hybrid QCNN..."
echo "=============================="
python main.py --model hybrid_qcnn --config configs/config_train_qcnn_fashion.yaml | tee logs_fashion/hybrid_qcnn.log

echo "=============================="
echo "Training Hybrid QCNN + SVM..."
echo "=============================="
python main.py --model svm_qkernel --config configs/config_train_hybrid_qcnn_svm_fashion.yaml | tee logs_fashion/svm_qkernel.log

echo "=============================="
echo "Training Classical SVM..."
echo "=============================="
python main.py --model svm --config configs/config_train_svm_fashion.yaml --optimize | tee logs_fashion/svm.log

echo "=============================="
echo "All trainings finished!"
