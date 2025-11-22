# models/cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class CNNBinaryClassifier(nn.Module):
    def __init__(self, in_channels=1, dropout=0.3, conv_channels=None, hidden_sizes=None):
        super(CNNBinaryClassifier, self).__init__()

        if conv_channels is None or len(conv_channels) == 0:
            conv_channels = [32, 64, 128]

        self.conv_blocks = nn.ModuleList()
        prev_channels = in_channels
        for idx, out_channels in enumerate(conv_channels):
            downsample = idx > 0
            self.conv_blocks.append(ResidualBlock(prev_channels, out_channels, downsample=downsample))
            prev_channels = out_channels

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        if hidden_sizes is None:
            hidden_sizes = []

        fc_layers = []
        prev_dim = prev_channels
        for hidden_dim in hidden_sizes:
            fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.classical_head = nn.Sequential(*fc_layers)

        self.fc = nn.Linear(prev_dim, 1)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classical_head(x)
        x = torch.sigmoid(self.fc(x))
        return x
