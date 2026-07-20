"""Tests for the trainer, including the adversarial-edge persistence monitor."""
import torch

from lance.data.dataset import EdgeBatch
from lance.models import TGNLite
from lance.training import Trainer


def _model(data, cfg):
    m = cfg.model
    return TGNLite(data.num_nodes, data.num_feats, m.memory_dim, m.time_dim,
                   m.embedding_dim, m.predictor_hidden, m.dropout)


def test_monitor_edges_records_persistence(tiny_data, tiny_cfg):
    s, d, t, f = tiny_data.split("train")
    k = 8
    edges = EdgeBatch(
        torch.as_tensor(s[:k], dtype=torch.long),
        torch.as_tensor(d[:k], dtype=torch.long),
        torch.as_tensor(t[:k], dtype=torch.float32),
        torch.as_tensor(f[:k], dtype=torch.float32),
    ).to("cpu")

    hist = Trainer(_model(tiny_data, tiny_cfg), tiny_cfg, device="cpu").fit(
        tiny_data, defense=None, verbose=False, monitor_edges=edges)
    assert len(hist) == tiny_cfg.train.epochs
    for row in hist:
        assert "adv_edge_score" in row
        assert 0.0 <= row["adv_edge_score"] <= 1.0


def test_monitor_absent_by_default(tiny_data, tiny_cfg):
    hist = Trainer(_model(tiny_data, tiny_cfg), tiny_cfg, device="cpu").fit(
        tiny_data, defense=None, verbose=False)
    assert all("adv_edge_score" not in row for row in hist)
