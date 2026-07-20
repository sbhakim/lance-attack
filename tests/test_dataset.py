"""Dataset invariants: chronological order, disjoint splits, perturbed builder."""
import numpy as np

from lance.data.dataset import TemporalGraphData


def test_chronological_and_splits(tiny_data):
    assert np.all(np.diff(tiny_data.t) >= 0), "events must be time-sorted"
    n = len(tiny_data.src)
    assert len(tiny_data.train_idx) + len(tiny_data.val_idx) + len(tiny_data.test_idx) == n
    # splits are contiguous and disjoint
    assert tiny_data.train_idx[-1] < tiny_data.val_idx[0]
    assert tiny_data.val_idx[-1] < tiny_data.test_idx[0]


def test_batches_cover_split(tiny_data):
    total = sum(len(b) for b in tiny_data.iter_batches("train", 64))
    assert total == len(tiny_data.train_idx)


def test_from_splits_keeps_val_test(tiny_data):
    tr = tiny_data.split("train")
    va = tiny_data.split("val")
    te = tiny_data.split("test")
    # replace train with half of it; val/test must be preserved
    half = (tr[0][:50], tr[1][:50], tr[2][:50], tr[3][:50])
    rebuilt = TemporalGraphData.from_splits(tiny_data.num_nodes, tiny_data.num_feats,
                                            half, va, te)
    assert len(rebuilt.train_idx) == 50
    assert len(rebuilt.val_idx) == len(tiny_data.val_idx)
    assert len(rebuilt.test_idx) == len(tiny_data.test_idx)
