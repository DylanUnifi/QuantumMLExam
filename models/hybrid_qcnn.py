# models/hybrid_qcnn.py
# Version: 1.1 - Hybrid QCNN with Residual MLP blocks, multi-dataset support, circuit visualization

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class ResidualMLPBlock(nn.Module):
    def __init__(self, in_features, out_features, downsample=False):
        super(ResidualMLPBlock, self).__init__()
        self.downsample = None
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        if downsample or in_features != out_features:
            self.downsample = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += identity
        return F.relu(out)

# Define quantum device
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# Quantum circuit layer definition
def qnode(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# QNode wrapper
weight_shapes = {"weights": (2, n_qubits)}
qnode_wrapped = qml.QNode(qnode, dev, interface="torch")
quantum_layer = qml.qnn.TorchLayer(qnode_wrapped, weight_shapes)

# Optional: Visualize the quantum circuit (PDF or stdout)
def visualize_quantum_circuit():
    from pennylane import drawer
    dummy_input = torch.tensor([0.0] * n_qubits)
    dummy_weights = torch.zeros((2, n_qubits))
    circuit_draw = drawer.draw(qnode)(dummy_input, dummy_weights)
    print("\n--- Quantum Circuit ---\n")
    print(circuit_draw)

class HybridQCNNBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64]):
        super(HybridQCNNBinaryClassifier, self).__init__()
        self.block1 = ResidualMLPBlock(input_size, hidden_sizes[0], downsample=True)
        self.block2 = ResidualMLPBlock(hidden_sizes[0], hidden_sizes[1], downsample=True)
        self.block3 = ResidualMLPBlock(hidden_sizes[1], hidden_sizes[2], downsample=True)
        self.quantum_fc_input = nn.Linear(hidden_sizes[2], n_qubits)
        self.quantum_layer = quantum_layer
        self.dropout = nn.Dropout(0.3)
        self.final_fc = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dropout(x)
        x = torch.tanh(self.quantum_fc_input(x))  # bounded input for quantum
        x = self.quantum_layer(x)
        x = self.final_fc(x)
        return torch.sigmoid(x)

# Auto-visualize when module is imported standalone
if __name__ == '__main__':
    visualize_quantum_circuit()
