"""HIA's node-importance (Impact) score, computed defensively or offensively.

    Impact(v) = w1 * temporal_degree_growth(v)
              + w2 * betweenness_centrality(v)
              + w3 * intra_community_degree(v)

Communities use Leiden if ``leidenalg`` is installed, otherwise NetworkX's
Louvain. Each component is min-max normalized and the final score lies in [0, 1].
The same routine serves the attacker, for target selection, and the C1 defense
component, which conditions screening on the same signal.
"""
from __future__ import annotations

import numpy as np
import networkx as nx


def _normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _communities(g: nx.Graph) -> dict[int, int]:
    """Return node -> community id, preferring Leiden, falling back to Louvain."""
    try:                                    # optional, exact-parity with HIA
        import igraph as ig
        import leidenalg as la
        mapping = {n: i for i, n in enumerate(g.nodes())}
        inv = {i: n for n, i in mapping.items()}
        ig_g = ig.Graph(edges=[(mapping[u], mapping[v]) for u, v in g.edges()],
                        n=len(mapping))
        part = la.find_partition(ig_g, la.ModularityVertexPartition)
        return {inv[i]: cid for cid, comm in enumerate(part) for i in comm}
    except Exception:
        comms = nx.community.louvain_communities(g, seed=0)
        return {n: cid for cid, comm in enumerate(comms) for n in comm}


def compute_impact(src: np.ndarray, dst: np.ndarray, num_nodes: int,
                   weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
                   betweenness_k: int = 200) -> np.ndarray:
    """Compute the per-node Impact score over an (undirected) edge snapshot."""
    w1, w2, w3 = weights
    n_edges = len(src)
    mid = n_edges // 2  # split the (time-ordered) stream into past vs recent

    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(zip(src.tolist(), dst.tolist()))

    # (1) temporal degree growth: recent-half degree minus past-half degree.
    deg_past = np.zeros(num_nodes)
    for u, v in zip(src[:mid], dst[:mid]):
        deg_past[u] += 1
        deg_past[v] += 1
    deg_all = np.zeros(num_nodes)
    for u, v in zip(src, dst):
        deg_all[u] += 1
        deg_all[v] += 1
    growth = np.clip(deg_all - 2 * deg_past, 0, None)

    # (2) betweenness centrality (k-sample approximation for scalability).
    k = min(betweenness_k, max(1, g.number_of_nodes()))
    bet = nx.betweenness_centrality(g, k=k, seed=0, normalized=True)
    bet_arr = np.array([bet.get(i, 0.0) for i in range(num_nodes)])

    # (3) intra-community degree.
    comm = _communities(g)
    intra = np.zeros(num_nodes)
    for u, v in zip(src.tolist(), dst.tolist()):
        if comm.get(u, -1) == comm.get(v, -2):
            intra[u] += 1
            intra[v] += 1

    impact = (w1 * _normalize(growth) + w2 * _normalize(bet_arr)
              + w3 * _normalize(intra))
    return _normalize(impact)
