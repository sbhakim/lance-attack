"""MLP link decoder: scores an ordered (source, destination) embedding pair."""
from __future__ import annotations

import torch
import torch.nn as nn


class LinkPredictor(nn.Module):
    def __init__(self, embedding_dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        """Return a logit per pair. Inputs are [*, D]; output is [*]."""
        return self.net(torch.cat([z_src, z_dst], dim=-1)).squeeze(-1)
