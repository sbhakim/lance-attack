"""Model shape/contract tests and a 'training runs without diverging' check."""
import numpy as np
import torch

from lance.models import TGNLite
from lance.training import Trainer
from lance.utils import seed_everything


def test_forward_shapes(tiny_data, tiny_cfg):
    m = TGNLite(tiny_data.num_nodes, tiny_data.num_feats, 16, 8, 16, 16)
    m.reset_state("cpu")
    batch = next(tiny_data.iter_batches("train", 32))
    neg = torch.randint(0, tiny_data.num_nodes, (len(batch), 5))
    pos, negs = m.score_pos_neg(batch, neg)
    assert pos.shape == (len(batch),)
    assert negs.shape == (len(batch), 5)


def test_memory_advances_and_staleness(tiny_data):
    m = TGNLite(tiny_data.num_nodes, tiny_data.num_feats, 16, 8, 16, 16)
    m.reset_state("cpu")
    batch = next(tiny_data.iter_batches("train", 32))
    assert torch.allclose(m.memory, torch.zeros_like(m.memory))
    m.advance_memory(batch)
    # touched nodes now have non-zero memory and a recorded last_update.
    # (A node may recur within a batch; memory uses last-write-wins, so we
    # check last_update is set and lies within the batch's time span.)
    assert m.memory[batch.src].abs().sum() > 0
    assert torch.all(m.last_update[batch.src] > 0)
    assert torch.all(m.last_update[batch.src] <= float(batch.t.max()) + 1e-6)


def test_training_runs_without_diverging(tiny_data, tiny_cfg):
    # Deterministic + robust: on tiny random data the loss can wobble across a
    # couple of epochs, so we assert the pipeline trains (finite losses) and does
    # not diverge, rather than a strict monotone decrease.
    seed_everything(0)
    m = TGNLite(tiny_data.num_nodes, tiny_data.num_feats, 16, 8, 16, 16)
    hist = Trainer(m, tiny_cfg, device="cpu").fit(tiny_data, verbose=False)
    assert len(hist) == tiny_cfg.train.epochs
    assert all(np.isfinite(h["loss"]) for h in hist)
    assert hist[-1]["loss"] <= hist[0]["loss"] * 1.1
