#!/bin/bash

echo "===================================="
echo "Launching all trainings in tmux..."
echo "===================================="

# Fashion-MNIST GPU 0
tmux new-session -d -s gpu0_fashion "bash train_gpu0_fashion.sh"

# Fashion-MNIST GPU 1
tmux new-session -d -s gpu1_fashion "bash train_gpu1_fashion.sh"

# CIFAR-10 GPU 0
tmux new-session -d -s gpu0_cifar10 "bash train_gpu0_cifar10.sh"

# CIFAR-10 GPU 1
tmux new-session -d -s gpu1_cifar10 "bash train_gpu1_cifar10.sh"

# SVHN GPU 0
tmux new-session -d -s gpu0_svhn "bash train_gpu0_svhn.sh"

# SVHN GPU 1
tmux new-session -d -s gpu1_svhn "bash train_gpu1_svhn.sh"

echo "===================================="
echo "All tmux sessions started!"
echo "Use 'tmux attach -t <session_name>' to view logs."
echo "Sessions launched:"
echo "  - gpu0_fashion"
echo "  - gpu1_fashion"
echo "  - gpu0_cifar10"
echo "  - gpu1_cifar10"
echo "  - gpu0_svhn"
echo "  - gpu1_svhn"
echo "===================================="
