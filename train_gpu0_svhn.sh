#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs_svhn

echo "=============================="
echo "Training Classical MLP..."
echo "=============================="
python main.py --model classical --config configs/config_train_classical_svhn.yaml | tee logs_svhn/classical.log

echo "=============================="
echo "Training Quantum MLP..."
echo "=============================="
python main.py --model quantum_mlp --config configs/config_train_quantum_mlp_svhn.yaml | tee logs_svhn/quantum_mlp.log

echo "=============================="
echo "Training Classical SVM..."
echo "=============================="
python main.py --model svm --config configs/config_train_svm_svhn.yaml --optimize | tee logs_svhn/svm.log
