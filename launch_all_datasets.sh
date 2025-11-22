#!/bin/bash

# Script : launch_all_datasets.sh
# Objectif : lancer en parallèle l'entraînement sur Fashion-MNIST, CIFAR-10 et SVHN,
# chacun dans une session tmux séparée, en forçant l'utilisation d'un GPU.

echo "Lancement des entraînements sur Fashion-MNIST dans tmux session 'fashion' sur GPU 0..."
tmux new-session -d -s fashion "CUDA_VISIBLE_DEVICES=0 bash train_all_fashion.sh"

echo "Lancement des entraînements sur CIFAR-10 dans tmux session 'cifar10' sur GPU 1..."
tmux new-session -d -s cifar10 "CUDA_VISIBLE_DEVICES=1 bash train_all_cifar10.sh"

echo "Lancement des entraînements sur SVHN dans tmux session 'svhn' sur GPU 0..."
tmux new-session -d -s svhn "CUDA_VISIBLE_DEVICES=0 bash train_all_svhn.sh"

echo "Tous les entraînements sont lancés. Utilisez 'tmux attach -t <session>' pour suivre une session."
