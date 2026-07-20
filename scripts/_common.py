"""Shared helpers for the CLI scripts (path bootstrap + builders)."""
from __future__ import annotations

import os
import sys

# Allow running the scripts directly (``python scripts/train.py``) without install.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lance.config import Config  # noqa: E402
from lance.data import load_dataset  # noqa: E402
from lance.data.dataset import TemporalGraphData  # noqa: E402
from lance.models import TGNLite  # noqa: E402


def build_model(cfg: Config, data: TemporalGraphData) -> TGNLite:
    m = cfg.model
    return TGNLite(data.num_nodes, data.num_feats, m.memory_dim, m.time_dim,
                   m.embedding_dim, m.predictor_hidden, m.dropout)


def load_data(cfg: Config) -> TemporalGraphData:
    return load_dataset(cfg.data.root, cfg.data.name, cfg.data.fmt,
                        cfg.data.max_events, cfg.data.val_ratio, cfg.data.test_ratio)


def perturb_train(data: TemporalGraphData, attack_result) -> TemporalGraphData:
    """Replace the train split with a poisoned stream; keep val/test clean."""
    r = attack_result
    return TemporalGraphData.from_splits(
        data.num_nodes, data.num_feats,
        train=(r.src, r.dst, r.t, r.feat),
        val=data.split("val"), test=data.split("test"),
    )
