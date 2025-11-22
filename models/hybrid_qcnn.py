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

def create_quantum_layer(
    n_qubits,
    n_layers=2,
    backend="lightning.qubit",
    shots=None,
    use_gpu=False,
):
    device_kwargs = {"wires": n_qubits, "shots": shots}
    selected_backend = backend

    if backend.startswith("lightning"):
        if use_gpu and torch.cuda.is_available():
            selected_backend = "lightning.gpu"
        elif use_gpu and backend != "lightning.kokkos":
            # Prefer Kokkos acceleration when GPU execution is requested but unavailable
            selected_backend = "lightning.kokkos"

    dev = qml.device(selected_backend, **device_kwargs)  # backend différentiable

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Z")
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
    def __init__(
        self,
        input_channel=1,
        dropout=0.3,
        n_qubits=4,
        n_layers=1,
        backend="lightning.qubit",
        shots=None,
        use_gpu=False,
        conv_channels=None,
        hidden_sizes=None,
    ):
        super().__init__()
        self.input_channel = input_channel

        if conv_channels is None:
            conv_channels = [32, 64, 128]

        self.conv_blocks = nn.ModuleList()
        in_ch = input_channel
        for idx, out_ch in enumerate(conv_channels):
            downsample = idx > 0  # reduce spatial dimensions after the first block
            self.conv_blocks.append(ResidualBlock(in_ch, out_ch, downsample=downsample))
            in_ch = out_ch

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        if hidden_sizes is None:
            hidden_sizes = []

        fc_layers = []
        prev_dim = in_ch
        for hidden_dim in hidden_sizes:
            fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.classical_head = nn.Sequential(*fc_layers)

        self.quantum_fc_input = nn.Linear(prev_dim, n_qubits)
        self.quantum_layer = create_quantum_layer(
            n_qubits,
            n_layers,
            backend=backend,
            shots=shots,
            use_gpu=use_gpu,
        )
        self.bn_q = nn.LayerNorm(n_qubits)
        self.final_fc = nn.Linear(n_qubits, 1)

    def forward(self, x):
        if x.dim() == 2:
            side = int((x.size(1) / self.input_channel) ** 0.5)
            if side * side * self.input_channel != x.size(1):
                raise ValueError(
                    f"Impossible de remodeler l'entrée de taille {x.size(1)} en image (canaux={self.input_channel})."
                )
            x = x.view(x.size(0), self.input_channel, side, side)
        elif x.dim() != 4:
            raise ValueError("L'entrée du HybridQCNNBinaryClassifier doit être 4D (N, C, H, W).")

        for block in self.conv_blocks:
            x = block(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classical_head(x)
        x = torch.tanh(self.quantum_fc_input(x)) * np.pi  # mapping [-π, π]
        x = self.quantum_layer(x)
        x = self.bn_q(x)
        x = self.final_fc(x)
        return torch.sigmoid(x)
