# models.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_shape, num_classes: int, h1: int, h2: int, dropout: float):
        super().__init__()
        c, h, w = input_shape
        in_dim = c * h * w
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, num_classes)
        self.dropout = float(dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc3(x)

class SmallCNN(nn.Module):
    """
    Simple CNN for CIFAR-10 (still fast).
    """
    def __init__(self, num_classes: int, base_channels: int = 32, dropout: float = 0.3):
        super().__init__()
        ch = base_channels
        self.conv1 = nn.Conv2d(3, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv3 = nn.Conv2d(ch, 2 * ch, 3, padding=1)
        self.conv4 = nn.Conv2d(2 * ch, 2 * ch, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = float(dropout)
        self.fc1 = nn.Linear((2 * ch) * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)          # 16x16
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)          # 8x8
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)
