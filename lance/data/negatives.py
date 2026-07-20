"""Negative sampling for link prediction.

Following the TGB protocol, two regimes are supported. Random sampling draws
destinations uniformly from the observed destinations. Historical sampling draws
a configurable fraction from a source's past partners other than the current
positive; these are harder negatives, and the ones a deletion can turn into a
confusable alternative to the true edge.
"""
from __future__ import annotations

import numpy as np


class NegativeSampler:
    def __init__(self, dst_pool: np.ndarray, seed: int = 0,
                 historical_frac: float = 0.0):
        self.dst_pool = np.asarray(dst_pool, dtype=np.int64)
        self.unique_dst = np.unique(self.dst_pool)
        self.historical_frac = float(historical_frac)
        self.rng = np.random.default_rng(seed)

    def sample(self, n: int, seen_dst: np.ndarray | None = None) -> np.ndarray:
        """Return ``n`` negative destination ids."""
        if self.historical_frac > 0.0 and seen_dst is not None and len(seen_dst) > 0:
            n_hist = int(n * self.historical_frac)
            hist = self.rng.choice(seen_dst, size=n_hist, replace=True)
            rand = self.rng.choice(self.unique_dst, size=n - n_hist, replace=True)
            return np.concatenate([hist, rand])
        return self.rng.choice(self.unique_dst, size=n, replace=True)

    def sample_matrix(self, n_pos: int, k: int,
                      positive_dst: np.ndarray | None = None) -> np.ndarray:
        """Return an ``[n_pos, k]`` matrix of negative destinations (for ranking)."""
        out = self.rng.choice(self.unique_dst, size=(n_pos, k), replace=True)
        if positive_dst is not None and len(self.unique_dst) > 1:
            positive_dst = np.asarray(positive_dst).reshape(-1)
            bad = out == positive_dst[:, None]
            while bad.any():
                out[bad] = self.rng.choice(self.unique_dst, size=int(bad.sum()), replace=True)
                bad = out == positive_dst[:, None]
        return out
