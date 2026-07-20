"""Time encoding used by the temporal model (Time2Vec / Bochner style)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """Map a scalar time delta to a ``dim``-vector via fixed log-spaced cosines.

    Follows the functional time encoding of TGAT/TGN: features are ``cos(w_k dt)``
    with geometrically spaced frequencies. The frequencies are fixed rather than
    learned, which we found more stable here.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        freqs = 1.0 / (10.0 ** np.linspace(0, 9, dim))
        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        # dt: [...]; returns [..., dim]
        return torch.cos(dt.unsqueeze(-1).float() * self.freqs)
