"""Shared fixtures: a tiny synthetic temporal graph (no disk I/O, CPU-only)."""
from __future__ import annotations

import numpy as np
import pytest

from lance.config import Config
from lance.data.dataset import TemporalGraphData


@pytest.fixture
def tiny_data() -> TemporalGraphData:
    rng = np.random.default_rng(0)
    n_nodes, n_events = 40, 600
    src = rng.integers(0, n_nodes // 2, size=n_events)
    dst = rng.integers(n_nodes // 2, n_nodes, size=n_events)
    t = np.sort(rng.uniform(0, 1000, size=n_events))
    feat = rng.normal(size=(n_events, 4)).astype(np.float32)
    return TemporalGraphData(src, dst, t, feat, n_nodes, 0.15, 0.15)


@pytest.fixture
def tiny_cfg() -> Config:
    return Config.from_dict({
        "model": {"memory_dim": 16, "time_dim": 8, "embedding_dim": 16, "predictor_hidden": 16},
        "train": {"epochs": 2, "batch_size": 64, "device": "cpu", "seed": 0},
        "eval": {"num_neg": 20, "hits_k": 10},
        "attack": {"ptb_rate": 0.1, "betweenness_k": 20},
        "defense": {"mode": "dtshield", "adv_every": 0, "smooth_lambda": 0.05},
    })
