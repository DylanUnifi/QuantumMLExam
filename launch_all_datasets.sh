#!/bin/bash

set -euo pipefail

# Script : launch_all_datasets.sh
# Objectif : lancer l'entraînement sur Fashion-MNIST, CIFAR-10 et SVHN,
# chacun dans une session tmux séparée lorsque tmux est disponible. Les GPU
# peuvent être choisis via les variables d'environnement GPU_FASHION,
# GPU_CIFAR10 et GPU_SVHN (défaut : 0).

GPU_FASHION=${GPU_FASHION:-0}
GPU_CIFAR10=${GPU_CIFAR10:-0}
GPU_SVHN=${GPU_SVHN:-0}

if command -v tmux >/dev/null 2>&1; then
  echo "Lancement des entraînements sur Fashion-MNIST dans la session tmux 'fashion' (GPU ${GPU_FASHION})..."
  tmux new-session -d -s fashion "CUDA_VISIBLE_DEVICES=${GPU_FASHION} bash train_all_fashion.sh"

  echo "Lancement des entraînements sur CIFAR-10 dans la session tmux 'cifar10' (GPU ${GPU_CIFAR10})..."
  tmux new-session -d -s cifar10 "CUDA_VISIBLE_DEVICES=${GPU_CIFAR10} bash train_all_cifar10.sh"

  echo "Lancement des entraînements sur SVHN dans la session tmux 'svhn' (GPU ${GPU_SVHN})..."
  tmux new-session -d -s svhn "CUDA_VISIBLE_DEVICES=${GPU_SVHN} bash train_all_svhn.sh"

  echo "Tous les entraînements sont lancés. Utilisez 'tmux attach -t <session>' pour suivre une session."
else
  echo "[WARN] tmux introuvable. Exécution séquentielle sur le GPU par défaut (${GPU_FASHION})."
  CUDA_VISIBLE_DEVICES=${GPU_FASHION} bash train_all_fashion.sh
  CUDA_VISIBLE_DEVICES=${GPU_CIFAR10} bash train_all_cifar10.sh
  CUDA_VISIBLE_DEVICES=${GPU_SVHN} bash train_all_svhn.sh
fi
