"""
deep_mlp.py

Architecture of the sound-quality classifier provided by the team.
Input  : 7 scaled features (6 audio features + 1 pitch)
Output : 2-class logits  (0 = Bad, 1 = Good)
"""

import torch.nn as nn


class DeepMLP(nn.Module):
    def __init__(self, input_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)
