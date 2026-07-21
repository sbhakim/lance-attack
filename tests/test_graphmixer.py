"""Tests for the memory-free GraphMixerLite victim (transfer experiments)."""
import torch

from lance.data import NegativeSampler
from lance.models import GraphMixerLite
from lance.training import Trainer


def _mixer(data):
    return GraphMixerLite(data.num_nodes, data.num_feats, k_neighbors=6,
                          time_dim=8, hidden=16, id_dim=16, embedding_dim=16,
                          predictor_hidden=16)


def test_graphmixer_scoring_shapes(tiny_data):
    m = _mixer(tiny_data)
    m.reset_state("cpu")
    for b in tiny_data.iter_batches("train", 64, "cpu"):
        if len(b):
            m.advance_memory(b)
    batch = next(tiny_data.iter_batches("val", 16, "cpu"))
    neg = torch.as_tensor(
        NegativeSampler(tiny_data.split("train")[1]).sample_matrix(len(batch), 10),
        dtype=torch.long)
    pos, negs = m.score_pos_neg(batch, neg)
    assert pos.shape == (len(batch),)
    assert negs.shape == (len(batch), 10)
    assert torch.isfinite(pos).all() and torch.isfinite(negs).all()


def test_graphmixer_trains_via_trainer(tiny_data, tiny_cfg):
    # The Trainer is architecture-agnostic; the memory-free victim must train too.
    hist = Trainer(_mixer(tiny_data), tiny_cfg, device="cpu").fit(
        tiny_data, defense=None, verbose=False)
    assert len(hist) == tiny_cfg.train.epochs
    for row in hist:
        assert 0.0 <= row["mrr"] <= 1.0
        assert row["loss"] == row["loss"]  # not NaN
