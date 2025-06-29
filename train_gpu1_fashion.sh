#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

mkdir -p logs_fashion

echo "=============================="
echo "Training CNN..."
echo "=============================="
python main.py --model cnn --config configs/config_train_cnn_fashion.yaml | tee logs_fashion/cnn.log

echo "=============================="
echo "Training Hybrid QCNN..."
echo "=============================="
python main.py --model hybrid_qcnn --config configs/config_train_qcnn_fashion.yaml | tee logs_fashion/hybrid_qcnn.log

echo "=============================="
echo "Training Hybrid QCNN + SVM..."
echo "=============================="
python main.py --model hybrid_qcnn_svm --config configs/config_train_hybrid_qcnn_svm_fashion.yaml | tee logs_fashion/hybrid_qcnn_svm.log
