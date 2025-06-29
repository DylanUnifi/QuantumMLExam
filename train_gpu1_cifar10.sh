#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

mkdir -p logs_cifar10

echo "=============================="
echo "Training CNN..."
echo "=============================="
python main.py --model cnn --config configs/config_train_cnn_cifar10.yaml | tee logs_cifar10/cnn.log

echo "=============================="
echo "Training Hybrid QCNN..."
echo "=============================="
python main.py --model hybrid_qcnn --config configs/config_train_qcnn_cifar10.yaml | tee logs_cifar10/hybrid_qcnn.log

echo "=============================="
echo "Training Hybrid QCNN + SVM..."
echo "=============================="
python main.py --model hybrid_qcnn_svm --config configs/config_train_hybrid_qcnn_svm_cifar10.yaml | tee logs_cifar10/hybrid_qcnn_svm.log
