"""Shared structural constraints for temporal edge-injection candidates."""
from __future__ import annotations

import numpy as np


def eligible_sources(src: np.ndarray, high_nodes=None, impact=None) -> np.ndarray:
    """Return valid source-side nodes, respecting bipartite event direction.

    JODIE datasets use disjoint source and destination ID spaces. Restricting
    injected sources to nodes already observed on the source side prevents
    invalid item-to-item interactions. For homogeneous graphs this remains a
    harmless restriction to nodes with observed outgoing interactions.
    """
    source_pool = np.unique(src).astype(np.int64)
    if high_nodes is None:
        return source_pool
    targeted = np.intersect1d(source_pool, np.asarray(list(high_nodes), dtype=np.int64))
    if len(targeted):
        return targeted
    if impact is not None and len(source_pool):
        # A tiny graph may have no source-side node above the global Impact
        # threshold; retain targeting by using its highest-Impact source.
        return source_pool[np.argsort(-impact[source_pool])[:1]]
    return source_pool


def filter_candidate_events(cu, cv, ct, existing_events=None):
    """Remove self-loops, duplicate candidates, and exact existing events."""
    existing = existing_events or set()
    keep = []
    seen = set()
    for i, (u, v, tt) in enumerate(zip(cu.tolist(), cv.tolist(), ct.tolist())):
        key = (int(u), int(v), float(tt))
        if u == v or key in seen or key in existing:
            continue
        seen.add(key)
        keep.append(i)
    return np.asarray(keep, dtype=np.int64)
