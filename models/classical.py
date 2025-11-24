# models/classical.py

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
    def __init__(self, input_size, hidden_sizes=None, dropout=0.3):
        super(MLPBinaryClassifier, self).__init__()
        if hidden_sizes is None or len(hidden_sizes) == 0:
            hidden_sizes = [256, 128, 64]

        blocks = []
        prev_dim = input_size
        for hidden_dim in hidden_sizes:
            blocks.append(ResidualMLPBlock(prev_dim, hidden_dim, downsample=True))
            prev_dim = hidden_dim

        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(prev_dim, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))
