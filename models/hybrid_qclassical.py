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

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers, backend="lightning.qubit", shots=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.dev = qml.device(backend, wires=n_qubits, shots=shots)  # backend différentiable

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            inputs = inputs.flatten()
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
                qml.RZ(inputs[i], wires=i)
            qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qnn = qml.qnn.TorchLayer(self.circuit, weight_shapes)

        # initialisation aléatoire des poids quantiques
        for name, param in self.qnn.named_parameters():
            if "weights" in name:
                nn.init.normal_(param, mean=0.0, std=0.01)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = torch.tanh(x[:, :self.n_qubits]) * np.pi

        outputs = []
        for sample in x:
            q_out = self.qnn(sample.unsqueeze(0))  # shape [1, n_qubits]
            outputs.append(q_out)
        return torch.cat(outputs, dim=0)  # shape [batch_size, n_qubits]

class QuantumResidualMLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes=None,
        n_qubits=4,
        n_layers=2,
        dropout=0.3,
        backend="lightning.qubit",
        shots=None,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]

        blocks = []
        prev_dim = input_size
        for hidden_dim in hidden_sizes:
            blocks.append(ResidualMLPBlock(prev_dim, hidden_dim, downsample=True, dropout=dropout))
            prev_dim = hidden_dim
        self.blocks = nn.ModuleList(blocks)

        self.dropout = nn.Dropout(dropout)
        self.quantum = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers, backend=backend, shots=shots)
        self.bn_q = nn.LayerNorm(n_qubits)  # stabilisation de la sortie quantique
        self.fc = nn.Linear(n_qubits, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x)
        x = self.quantum(x)
        x = self.bn_q(x)
        return torch.sigmoid(self.fc(x))
