"""Contracts for the oracle deletion bound and the coverage diagnostic.

The oracle is a measurement instrument: other attacks are reported as a fraction
of it. A bound that is not actually an upper reference would silently rescale
every such number, so its defining properties are pinned here.
"""
import numpy as np

from lance.attack import (deletion_coverage, oracle_delete_attack, run_attack,
                          query_support_value)


def _split(tiny_data):
    s, d, t, f = tiny_data.split("train")
    ts, td = tiny_data.split("test")[:2]
    return s, d, t, f, ts, td


def test_oracle_spends_budget_and_only_deletes(tiny_data):
    s, d, t, f, ts, td = _split(tiny_data)
    res = oracle_delete_attack(s, d, t, f, tiny_data.num_nodes, ts, td, ptb_rate=0.2)
    budget = int(0.2 * len(s))
    assert res.n_deleted == budget and res.n_injected == 0
    assert len(res.src) == len(s) - budget


def test_oracle_beats_random_deletion_on_coverage(tiny_data):
    """The bound must dominate the control it normalises, on its own objective.

    If oracle coverage did not exceed random coverage the "fraction of achievable
    damage" framing would be meaningless.
    """
    s, d, t, f, ts, td = _split(tiny_data)
    n = tiny_data.num_nodes
    oracle = oracle_delete_attack(s, d, t, f, n, ts, td, ptb_rate=0.2)
    rand = run_attack("random_delete", s, d, t, f, n, ptb_rate=0.2, seed=0)

    def cov(res):
        kept = set(zip(res.src.tolist(), res.dst.tolist(), res.t.tolist()))
        idx = np.array([i for i, k in enumerate(zip(s.tolist(), d.tolist(), t.tolist()))
                        if k not in kept], dtype=np.int64)
        return deletion_coverage(s, d, n, ts, td, idx)["test_weighted_coverage"]

    assert cov(oracle) > cov(rand)


def test_coverage_is_a_bounded_fraction(tiny_data):
    s, d, t, f, ts, td = _split(tiny_data)
    n = tiny_data.num_nodes
    none = deletion_coverage(s, d, n, ts, td, np.array([], dtype=np.int64))
    everything = deletion_coverage(s, d, n, ts, td, np.arange(len(s)))
    assert none["test_weighted_coverage"] == 0.0
    assert everything["test_weighted_coverage"] == 1.0


def query_support_value_prefers_scarce_support_over_hubs(tiny_data):
    """Value is a *share* of a node's support, not a raw count.

    Without the degree normalisation the bound would pile the whole budget onto a
    few hubs, whose remaining support absorbs the loss, and would understate what
    the deletion channel can achieve.
    """
    s, d, t, f, ts, td = _split(tiny_data)
    n = tiny_data.num_nodes
    val = query_support_value(s, d, n, ts, td)
    deg = np.bincount(s, minlength=n) + np.bincount(d, minlength=n)
    # among edges whose endpoints are all test-active, low-degree endpoints score higher
    endpoint_deg = deg[s] + deg[d]
    hi, lo = val > np.median(val), endpoint_deg > np.median(endpoint_deg)
    assert endpoint_deg[hi].mean() < endpoint_deg[~hi].mean()
