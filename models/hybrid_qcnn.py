import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np

class ResidualMLPBlock(nn.Module):
    def __init__(self, in_features, out_features, downsample=False, dropout=0.3):
        super().__init__()
        self.downsample = None
        self.fc1 = nn.Linear(in_features, out_features)
        self.ln1 = nn.LayerNorm(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.ln2 = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

        if downsample or in_features != out_features:
            self.downsample = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features)
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.ln1(self.fc1(x)))
        out = self.dropout(out)
        out = self.ln2(self.fc2(out))
        out += identity
        return F.relu(out)

def create_quantum_layer(n_qubits, n_layers=2):
    dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)  # backend différentiable

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        inputs = inputs.flatten()
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
            qml.RZ(inputs[i], wires=i)  # double rotation pour enrichir l’encoding
        qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits)}
    layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    # initialisation aléatoire des poids quantiques
    for name, param in layer.named_parameters():
        if "weights" in name:
            nn.init.normal_(param, mean=0.0, std=0.01)

    return layer

class HybridQCNNBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[32, 16, 8], n_qubits=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.block1 = ResidualMLPBlock(input_size, hidden_sizes[0], downsample=True, dropout=dropout)
        self.block2 = ResidualMLPBlock(hidden_sizes[0], hidden_sizes[1], downsample=True, dropout=dropout)
        self.block3 = ResidualMLPBlock(hidden_sizes[1], hidden_sizes[2], downsample=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.quantum_fc_input = nn.Linear(hidden_sizes[2], n_qubits)
        self.quantum_layer = create_quantum_layer(n_qubits, n_layers)
        self.bn_q = nn.LayerNorm(n_qubits)
        self.final_fc = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dropout(x)
        x = torch.tanh(self.quantum_fc_input(x)) * np.pi  # mapping [-π, π]

        outputs = []
        for sample in x:
            q_out = self.quantum_layer(sample.unsqueeze(0))
            outputs.append(q_out)
        x = torch.cat(outputs, dim=0)

        x = self.bn_q(x)
        x = self.final_fc(x)
        return torch.sigmoid(x)

class HybridQCNNFeatures(nn.Module):
    def __init__(self, input_size, hidden_sizes=[32, 16, 8], n_qubits=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.block1 = ResidualMLPBlock(input_size, hidden_sizes[0], downsample=True, dropout=dropout)
        self.block2 = ResidualMLPBlock(hidden_sizes[0], hidden_sizes[1], downsample=True, dropout=dropout)
        self.block3 = ResidualMLPBlock(hidden_sizes[1], hidden_sizes[2], downsample=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.quantum_fc_input = nn.Linear(hidden_sizes[2], n_qubits)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dropout(x)
        return torch.tanh(self.quantum_fc_input(x)) * np.pi  # mapping enrichi
