# utils/kernels.py
# Version: 1.0

import numpy as np


def center_kernel_matrix(K):
    """
    Centre la matrice de noyau selon la méthode de centering classique (centrage double).
    """
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n


def normalize_kernel_matrix(K):
    """
    Normalise la matrice de noyau par rapport à sa diagonale :
    K_ij / sqrt(K_ii * K_jj)
    """
    diag = np.sqrt(np.diag(K))
    return K / (diag[:, None] @ diag[None, :])
