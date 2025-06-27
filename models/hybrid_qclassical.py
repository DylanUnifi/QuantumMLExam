# models/hybrid_qclassical.py
# Version: 1.0 – Quantum Residual MLP Inspired

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers, input_dim):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_dim = input_dim

        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qnn = qml.qnn.TorchLayer(self.circuit, weight_shapes)

    def forward(self, x):
        # suppose x est de dimension [batch_size, input_dim]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x[:, :self.n_qubits] * np.pi  # mapping inputs
        return self.qnn(x)

class QuantumResidualMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[8, 4], n_layers=2):
        super().__init__()
        self.fc_in = nn.Linear(input_size, hidden_sizes[0])
        self.bn_in = nn.BatchNorm1d(hidden_sizes[0])
        self.quantum1 = QuantumLayer(n_qubits=hidden_sizes[0], n_layers=n_layers, input_dim=input_size)

        self.fc_mid = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn_mid = nn.BatchNorm1d(hidden_sizes[1])
        self.quantum2 = QuantumLayer(n_qubits=hidden_sizes[1], n_layers=n_layers, input_dim=hidden_sizes[0])

        self.fc_out = nn.Linear(hidden_sizes[1], 1)

    def forward(self, x):
        x = torch.relu(self.bn_in(self.fc_in(x)))
        x = self.quantum1(x)
        x = torch.relu(self.bn_mid(self.fc_mid(x)))
        x = self.quantum2(x)
        x = self.fc_out(x)
        return torch.sigmoid(x)
