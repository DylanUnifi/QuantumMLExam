import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
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
    def __init__(self, input_channel = 1, dropout=0.3, n_qubits=4, n_layers=1):
        super().__init__()
        self.layer1 = ResidualBlock(input_channel, 32)
        self.layer2 = ResidualBlock(32, 64, downsample=True)
        self.layer3 = ResidualBlock(64, 128, downsample=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.quantum_fc_input = nn.Linear(128, n_qubits)
        self.quantum_layer = create_quantum_layer(n_qubits, n_layers)
        self.bn_q = nn.LayerNorm(n_qubits)
        self.final_fc = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
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
