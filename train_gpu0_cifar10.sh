#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

mkdir -p logs_cifar10

echo "=============================="
echo "Training Classical MLP..."
echo "=============================="
python main.py --model classical --config configs/config_train_classical_cifar10.yaml | tee logs_cifar10/classical.log

echo "=============================="
echo "Training Quantum MLP..."
echo "=============================="
python main.py --model quantum_mlp --config configs/config_train_quantum_mlp_cifar10.yaml | tee logs_cifar10/quantum_mlp.log

echo "=============================="
echo "Training Classical SVM..."
echo "=============================="
python main.py --model svm --config configs/config_train_svm_cifar10.yaml --optimize | tee logs_cifar10/svm.log
