# models/quantum_kernel.py
# Version: 3.0 – Optimisé (vitesse, qualité du noyau, visualisation)

import pennylane as qml
import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

# Device setup
_qdev_cache = {}
def get_qdevice(n_qubits, use_gpu=False):
    """
    Returns a cached PennyLane device. Uses lightning.gpu if use_gpu=True, else lightning.qubit (CPU).
    """
    key = (n_qubits, use_gpu)
    if key not in _qdev_cache:
        backend = "lightning.gpu" if use_gpu else "lightning.qubit"
        _qdev_cache[key] = qml.device(backend, wires=n_qubits, shots=None)
    return _qdev_cache[key]


# Hybrid Quantum Kernel (Amplitude Encoding + Fidelity)
def hybrid_kernel_circuit(x, wires):
    qml.AmplitudeEmbedding(x, wires=wires, normalize=True)
    qml.Barrier(wires=wires)

def fidelity_kernel(x1, x2, circuit, wires, dev):
    @qml.qnode(dev)
    def kernel_fn(x1_, x2_):
        circuit(x1_, wires)
        qml.adjoint(circuit)(x2_, wires)
        return qml.probs(wires=wires)

    probs = kernel_fn(x1, x2)
    fidelity = np.sum(np.sqrt(probs))
    return fidelity

def compute_fidelity_kernel_matrix(X, verbose=True):
    n_samples = len(X)
    K = np.zeros((n_samples, n_samples))
    wires = list(range(int(np.log2(X.shape[1]))))
    dev = get_qdevice(len(wires), use_gpu=True)
    print(f"[INFO] Using PennyLane backend: {dev.__class__.__name__}")

    iterator = tqdm(range(n_samples), desc="Computing QKernel") if verbose else range(n_samples)
    for i in iterator:
        for j in range(i, n_samples):
            val = fidelity_kernel(X[i], X[j], hybrid_kernel_circuit, wires, dev)
            K[i, j] = val
            K[j, i] = val
    return K


# Quantum Kitchen Sinks (QKS)
def random_qks_features(X, n_qubits=4, n_layers=1, seed=42, return_weights=False, n_jobs=8):
    """
    Compute Quantum Kitchen Sinks features in parallel using joblib.
    """
    np.random.seed(seed)
    dev = get_qdevice(n_qubits, use_gpu=True)
    print(f"[INFO] Using PennyLane backend: {dev.__class__.__name__}")

    @qml.qnode(dev)
    def qks_circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
        for i in range(n_layers):
            for j in range(n_qubits):
                qml.RY(weights[i, j], wires=j)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weights = np.random.uniform(0, 2 * np.pi, size=(n_layers, n_qubits))

    print(f"Computing QKS in parallel with n_jobs={n_jobs if n_jobs != -1 else 'all available cores'}...")

    # Parallélisation avec joblib
    features = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(qks_circuit)(x, weights) for x in tqdm(X, desc="Computing QKS")
        )
    )
    return (features, weights) if return_weights else features


# Torch wrappers (for hybrid kernel + SVM compatibility)
def torch_quantum_kernel(X):
    X_np = X.detach().cpu().numpy()
    K = compute_fidelity_kernel_matrix(X_np)
    return torch.tensor(K, dtype=torch.float32)

def torch_qks_features(X):
    X_np = X.detach().cpu().numpy()
    return torch.tensor(random_qks_features(X_np), dtype=torch.float32)

# Diagnostic : visualisation des vecteurs (debugging kernel collapse)
def inspect_quantum_vectors(X, limit=5):
    print("\nSample Quantum Vectors:")
    for i, vec in enumerate(X[:limit]):
        print(f"Vec {i}: {vec}")
