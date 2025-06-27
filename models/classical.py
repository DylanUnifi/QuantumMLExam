# models/classical.py
# Version: 3.0 - MLP Residual Inspired

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=None):
        super(MLPBinaryClassifier, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]
        self.block1 = ResidualMLPBlock(input_size, hidden_sizes[0], downsample=True)
        self.block2 = ResidualMLPBlock(hidden_sizes[0], hidden_sizes[1], downsample=True)
        self.block3 = ResidualMLPBlock(hidden_sizes[1], hidden_sizes[2], downsample=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_sizes[2], 1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))
