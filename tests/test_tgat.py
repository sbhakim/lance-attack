"""Tests for the memory-free attention victim TGATLite (transfer experiments)."""
import torch

from lance.data import NegativeSampler
from lance.models import TGATLite
from lance.training import Trainer


def _tgat(data):
    return TGATLite(data.num_nodes, data.num_feats, k_neighbors=6, time_dim=8,
                    node_dim=16, hidden=16, embedding_dim=16, predictor_hidden=16)


def test_tgat_scoring_shapes(tiny_data):
    m = _tgat(tiny_data)
    m.reset_state("cpu")
    for b in tiny_data.iter_batches("train", 64, "cpu"):
        if len(b):
            m.advance_memory(b)
    batch = next(tiny_data.iter_batches("val", 16, "cpu"))
    neg = torch.as_tensor(
        NegativeSampler(tiny_data.split("train")[1]).sample_matrix(len(batch), 10),
        dtype=torch.long)
    pos, negs = m.score_pos_neg(batch, neg)
    assert pos.shape == (len(batch),) and negs.shape == (len(batch), 10)
    assert torch.isfinite(pos).all() and torch.isfinite(negs).all()  # cold-node guard


def test_tgat_trains_via_trainer(tiny_data, tiny_cfg):
    hist = Trainer(_tgat(tiny_data), tiny_cfg, device="cpu").fit(
        tiny_data, defense=None, verbose=False)
    assert len(hist) == tiny_cfg.train.epochs
    for row in hist:
        assert 0.0 <= row["mrr"] <= 1.0
        assert row["loss"] == row["loss"]  # not NaN
