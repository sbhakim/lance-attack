"""Impact score: range, normalization, and that a hub scores highly."""
import numpy as np

from lance.attack import compute_impact


def test_impact_range_and_shape(tiny_data):
    s, d, _, _ = tiny_data.split("train")
    imp = compute_impact(s, d, tiny_data.num_nodes, betweenness_k=20)
    assert imp.shape == (tiny_data.num_nodes,)
    assert imp.min() >= 0.0 and imp.max() <= 1.0 + 1e-6


def test_hub_is_important():
    # node 0 is a hub connected to many others; it should outrank a leaf.
    n = 30
    src = np.zeros(40, dtype=np.int64)
    dst = np.arange(1, 41, dtype=np.int64) % n
    src = np.concatenate([src, np.array([15])])     # one extra edge for a leaf
    dst = np.concatenate([dst, np.array([16])])
    imp = compute_impact(src, dst, n, betweenness_k=10)
    assert imp[0] >= imp[16]
