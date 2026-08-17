# lance/attack/oracle.py
"""A coverage-maximising deletion reference, and the coverage diagnostic behind it.

Every result in this project is relative: attack A degrades MRR more than attack B.
That is enough to rank attacks and not enough to interpret them. A $+0.02$ paired
drop could be most of the damage obtainable at that budget or a small fraction of
it, and the two readings support opposite conclusions. The gap matters most where
attacks converge: when targeted and random deletion land close together, it is
either because targeting failed or because little targetable structure exists, and
a purely relative measurement cannot tell those apart.

This module supplies the missing denominator.

``oracle_delete_attack`` is **not a proposed attack and not part of the threat
model**: it reads the test split, which a real adversary cannot. It exists to
bound how much damage the deletion channel can do at a given budget, so every
legitimate attack can be reported as a fraction of what was available.

The bound is built from how the victim is actually evaluated. The benchmark fixes
the evaluation history and the negative pool to the *clean* graph, so deleting a
training edge cannot change which queries are asked or which negatives they are
ranked against -- it can only degrade the state the victim brings to them. A test
query on $(u,v)$ is answered from whatever the victim learned about $u$ and $v$,
and what it learned about a node comes from that node's training events. So the
damage a deletion can do is the test-weighted share of node support it removes:

    value(a, b) = testfreq[a] / deg[a] + testfreq[b] / deg[b]

where ``testfreq[n]`` counts appearances of $n$ in test queries and ``deg[n]``
counts its training events. Dividing by degree is what makes this a *share* rather
than a count, and it is what stops the bound from collapsing onto a handful of hubs:
the tenth deletion from a node with ten events removes as much of that node's
remaining support as the first, but a node with a thousand events barely notices any
single one. Taking the top-$B$ edges by this value spreads the budget across nodes
in proportion to how much test-relevant support each one actually has.

**It is not an upper bound on damage, and measurement showed why.** On MOOC against
GraphMixerLite it attains the highest coverage of any attack (0.359 against LANCE's
0.227) while doing roughly 60% of LANCE's damage (+0.0296 against +0.0473). Coverage
and damage are therefore dissociated: removing the most test-relevant support by
volume is not the most damaging way to spend a deletion budget. Degree normalisation
pushes this strategy onto low-degree, high-test-frequency nodes and strips them
nearly bare, and a node whose history is removed outright appears to be less damaged
than one left partially corrupted -- the victim gets a clean "unknown" state rather
than a misleading one. GraphMixerLite masks empty neighbour buffers explicitly, so
that reading is architecture-specific and remains a hypothesis.

Use this as a *coverage-maximising reference*, never as a denominator for
"fraction of achievable damage": attacks routinely exceed it.
"""
from __future__ import annotations

import numpy as np

from lance.attack.baselines import _assemble


def query_support_value(src: np.ndarray, dst: np.ndarray, num_nodes: int,
                        test_src: np.ndarray, test_dst: np.ndarray) -> np.ndarray:
    """Per-training-edge share of test-relevant node support (see module docstring)."""
    deg = np.bincount(src, minlength=num_nodes) + np.bincount(dst, minlength=num_nodes)
    testfreq = (np.bincount(test_src, minlength=num_nodes)
                + np.bincount(test_dst, minlength=num_nodes)).astype(float)
    share = np.divide(testfreq, deg, out=np.zeros(num_nodes, dtype=float),
                      where=deg > 0)
    return share[src] + share[dst]


def deletion_coverage(src: np.ndarray, dst: np.ndarray, num_nodes: int,
                      test_src: np.ndarray, test_dst: np.ndarray,
                      deleted_idx: np.ndarray) -> dict:
    """How much test-relevant training support a deletion set removes.

    Returns the test-weighted fraction of node support deleted, which is the
    causal quantity ``\\Delta`` MRR only observes at the end of training. Reported
    alongside damage, it separates "the attack removed little that mattered" from
    "the attack removed a lot and the victim recovered anyway" -- two failure modes
    that look identical in a metric table.
    """
    deg = np.bincount(src, minlength=num_nodes) + np.bincount(dst, minlength=num_nodes)
    testfreq = (np.bincount(test_src, minlength=num_nodes)
                + np.bincount(test_dst, minlength=num_nodes)).astype(float)
    if len(deleted_idx) == 0 or testfreq.sum() == 0:
        return {"test_weighted_coverage": 0.0, "nodes_touched": 0,
                "test_active_nodes_touched": 0}
    d_src, d_dst = src[deleted_idx], dst[deleted_idx]
    removed = (np.bincount(d_src, minlength=num_nodes)
               + np.bincount(d_dst, minlength=num_nodes))
    frac = np.divide(removed, deg, out=np.zeros(num_nodes, dtype=float), where=deg > 0)
    touched = np.unique(np.concatenate([d_src, d_dst]))
    return {
        "test_weighted_coverage": float((testfreq * frac).sum() / testfreq.sum()),
        "nodes_touched": int(len(touched)),
        "test_active_nodes_touched": int((testfreq[touched] > 0).sum()),
        "test_active_nodes_total": int((testfreq > 0).sum()),
    }


def oracle_delete_attack(src, dst, t, feat, num_nodes, test_src, test_dst,
                         ptb_rate: float = 0.1, seed: int = 0, **_):
    """Delete the top-budget training edges by test-support value.

    Threat-model-violating by construction; see the module docstring. Use it to
    normalise other attacks, never as a baseline they are claimed to beat.
    """
    n = len(src)
    budget = min(int(ptb_rate * n), n)
    keep = np.ones(n, dtype=bool)
    if budget > 0:
        value = query_support_value(src, dst, num_nodes, test_src, test_dst)
        chosen = np.argsort(-value, kind="stable")[:budget]
        keep[chosen] = False
    empty = np.array([], dtype=np.int64)
    result = _assemble(src, dst, t, feat, keep, empty, empty,
                       np.array([], dtype=np.float64), int(budget))
    result.diagnostics = {
        "budget": int(budget),
        "selected_deletions": int(budget),
        "selected_injections": 0,
        "oracle": True,
        "coverage": deletion_coverage(src, dst, num_nodes, test_src, test_dst,
                                      np.where(~keep)[0]),
    }
    return result
