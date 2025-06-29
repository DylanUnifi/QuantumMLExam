#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# Script : train_all_svhn.sh
# But : entraîner tous les modèles pertinents sur SVHN avec main.py
# et enregistrer les logs dans le dossier logs_svhn/

mkdir -p logs_svhn

echo "=============================="
echo "Training Classical MLP..."
echo "=============================="
python main.py --model classical --config configs/config_train_classical_svhn.yaml | tee logs_svhn/classical.log

echo "=============================="
echo "Training CNN..."
echo "=============================="
python main.py --model cnn --config configs/config_train_cnn_svhn.yaml | tee logs_svhn/cnn.log

echo "=============================="
echo "Training Quantum MLP..."
echo "=============================="
python main.py --model quantum_mlp --config configs/config_train_quantum_mlp_svhn.yaml | tee logs_svhn/quantum_mlp.log

echo "=============================="
echo "Training Hybrid QCNN..."
echo "=============================="
python main.py --model hybrid_qcnn --config configs/config_train_qcnn_svhn.yaml | tee logs_svhn/hybrid_qcnn.log

echo "=============================="
echo "Training Hybrid QCNN + SVM..."
echo "=============================="
python main.py --model hybrid_qcnn_svm --config configs/config_train_hybrid_qcnn_svm_svhn.yaml | tee logs_svhn/hybrid_qcnn_svm.log

echo "=============================="
echo "Training Classical SVM..."
echo "=============================="
python main.py --model svm --config configs/config_train_svm_svhn.yaml --optimize | tee logs_svhn/svm.log

echo "=============================="
echo "All trainings on SVHN finished!"
