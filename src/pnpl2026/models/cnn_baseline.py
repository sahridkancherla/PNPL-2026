import torch
import torch.nn as nn


class SimpleWordCNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128), nn.GELU(), nn.MaxPool1d(4),
            nn.Conv1d(128, 256, kernel_size=9, padding=4),
            nn.BatchNorm1d(256), nn.GELU(), nn.MaxPool1d(4),
            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.GELU(), nn.AdaptiveAvgPool1d(1),   # -> (batch, 256, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.1), nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))
