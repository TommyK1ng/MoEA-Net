"""Pooling and classification layers used by the retinal models."""

import torch
from torch import nn


class AttentionPooling(nn.Module):
    """Diagnostic-aware attention pooling over image patches."""

    def __init__(self, emb):
        super().__init__()
        self.score = nn.Linear(emb, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return (x * weights.unsqueeze(-1)).sum(dim=-2)


class VisionClassification(nn.Module):
    def __init__(self, hidden_dim, vision_improvement=1):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, vision_improvement)

    def forward(self, x):
        return self.fc(x)


class SubretinalFluidClassification(nn.Module):
    def __init__(self, hidden_dim, fluid_obsorbed=3):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, fluid_obsorbed)

    def forward(self, x):
        return self.fc(x)
