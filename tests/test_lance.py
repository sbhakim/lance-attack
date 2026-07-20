"""Tests for the LANCE attack: adaptive perturbation core + the K1/K2/K3 orchestrator."""
import numpy as np

from lance.attack import compute_impact, adaptive_hybrid_attack, lance_attack, run_attack
from lance.data.dataset import TemporalGraphData


def _score_fn():
    rng = np.random.default_rng(0)
    return lambda a, b, c: rng.uniform(size=len(a))


def test_adaptive_hybrid_core(tiny_data):
    s, d, t, f = tiny_data.split("train")
    imp = compute_impact(s, d, tiny_data.num_nodes, betweenness_k=20)
    budget = int(0.2 * len(s))
    res = adaptive_hybrid_attack(s, d, t, f, tiny_data.num_nodes, imp, _score_fn(),
                                 budget, high_impact_frac=0.2, seed=0)
    # spends (about) the whole budget across deletions + injections, data-driven split
    assert res.n_deleted + res.n_injected <= budget
    assert res.n_deleted + res.n_injected >= 1
    assert len(res.src) == len(res.dst) == len(res.t) == len(res.feat)
    source_domain = set(s.tolist())
    existing = set(zip(s.tolist(), d.tolist(), t.tolist()))
    injected = list(zip(res.injected_src.tolist(), res.injected_dst.tolist(),
                        res.injected_t.tolist()))
    assert all(u in source_domain for u in res.injected_src)
    assert all(u != v for u, v in zip(res.injected_src, res.injected_dst))
    assert not any(edge in existing for edge in injected)
    assert len(injected) == len(set(injected))
    if res.n_injected:
        injected_feat = res.feat[res.adv_mask]
        assert not np.allclose(injected_feat, 0.0)
    assert res.diagnostics["budget"] == budget
    assert res.diagnostics["candidate_deletions"] >= res.n_deleted
    assert res.diagnostics["candidate_injections"] >= res.n_injected
    assert "selected_delete_score_summary" in res.diagnostics
    assert "selected_inject_score_summary" in res.diagnostics
    assert "delete_type_diagnostics" in res.diagnostics
    assert "future_reused_pair" in res.diagnostics["delete_type_diagnostics"]
    assert "injection_edge_diagnostics" in res.diagnostics
    assert res.diagnostics["injection_edge_diagnostics"]["n"] == res.n_injected
    if res.n_injected:
        assert len(res.diagnostics["injection_edge_diagnostics"]["per_edge"]) == res.n_injected


def test_adaptive_hybrid_prefers_repeated_high_score_deletions():
    src = np.array([0, 0, 0, 1, 1, 2, 2, 3], dtype=np.int64)
    dst = np.array([4, 4, 4, 5, 5, 6, 7, 7], dtype=np.int64)
    t = np.arange(len(src), dtype=np.float64)
    feat = np.ones((len(src), 2), dtype=np.float32)
    impact = np.ones(8, dtype=float)

    def score_fn(a, b, _):
        return np.where((a == 0) & (b == 4), 0.99, 0.15).astype(float)

    res = adaptive_hybrid_attack(src, dst, t, feat, 8, impact, score_fn,
                                 budget=3, high_impact_frac=1.0, seed=1)
    assert res.n_deleted >= 2
    assert res.diagnostics["selected_deletions"] == res.n_deleted
    assert res.diagnostics["selected_delete_score_summary"]["median"] > 0.9
    assert res.diagnostics["delete_type_diagnostics"]["future_reused_pair"] >= 1


def test_run_attack_lance_dispatch(tiny_data):
    s, d, t, f = tiny_data.split("train")
    imp = compute_impact(s, d, tiny_data.num_nodes, betweenness_k=20)
    res = run_attack("lance", s, d, t, f, tiny_data.num_nodes, impact=imp,
                     score_fn=_score_fn(), ptb_rate=0.2, seed=0)
    assert res is not None and res.adv_mask.shape[0] == len(res.src)


def test_lance_component_ablation_contracts(tiny_data):
    s, d, t, f = tiny_data.split("train")
    imp = compute_impact(s, d, tiny_data.num_nodes, betweenness_k=20)
    common = dict(impact=imp, score_fn=_score_fn(), ptb_rate=0.2, seed=0)
    inject = run_attack("lance_inject", s, d, t, f, tiny_data.num_nodes, **common)
    delete = run_attack("lance_delete", s, d, t, f, tiny_data.num_nodes, **common)
    fixed = run_attack("lance_fixed", s, d, t, f, tiny_data.num_nodes, **common)
    random_target = run_attack(
        "lance_random_target", s, d, t, f, tiny_data.num_nodes, **common)
    query = run_attack("lance_query", s, d, t, f, tiny_data.num_nodes, **common)
    query_inject = run_attack(
        "lance_query_inject", s, d, t, f, tiny_data.num_nodes, **common)
    assert inject.n_deleted == 0 and inject.n_injected > 0
    assert delete.n_injected == 0 and delete.n_deleted > 0
    assert fixed.n_deleted > 0 and fixed.n_injected > 0
    assert random_target.n_deleted + random_target.n_injected > 0
    assert query.diagnostics["query_aware_injection"] is True
    assert query_inject.n_deleted == 0 and query_inject.n_injected > 0
    assert query_inject.diagnostics["query_aware_injection"] is True


def test_lance_orchestrator_k1_k2(tiny_data, tiny_cfg):
    # full end-to-end: builds a (limited-knowledge) surrogate and perturbs the graph
    for knowledge in ("k1", "k2"):
        tiny_cfg.attack.knowledge = knowledge
        tiny_cfg.attack.ptb_rate = 0.15
        res = lance_attack(tiny_data, tiny_cfg, device="cpu")
        assert res is not None
        assert res.n_deleted + res.n_injected >= 0
        assert len(res.src) == len(res.dst) == len(res.t)


def test_lance_streaming_k3(tiny_data, tiny_cfg):
    tiny_cfg.attack.knowledge = "k3"
    tiny_cfg.attack.stream_windows = 3
    tiny_cfg.attack.ptb_rate = 0.2
    res = lance_attack(tiny_data, tiny_cfg, device="cpu")
    assert res is not None
    assert len(res.src) == len(res.dst) == len(res.t) == len(res.feat)


def test_k2_is_independent_of_hidden_suffix(tiny_data, tiny_cfg):
    tiny_cfg.attack.knowledge = "k2"
    tiny_cfg.attack.lk_cutoff_frac = 0.6
    tiny_cfg.attack.ptb_rate = 0.15
    train = tiny_data.split("train")
    cut = int(tiny_cfg.attack.lk_cutoff_frac * len(train[0]))

    changed = tuple(x.copy() for x in train)
    # Change every hidden endpoint and feature while keeping timestamps and the
    # global node universe fixed. A strict K2 attack must select the same edits.
    changed[0][cut:] = (changed[0][cut:] + 3) % tiny_data.num_nodes
    changed[1][cut:] = (changed[1][cut:] + 7) % tiny_data.num_nodes
    changed[3][cut:] *= -5.0
    alternate = TemporalGraphData.from_splits(
        tiny_data.num_nodes, tiny_data.num_feats, changed,
        tiny_data.split("val"), tiny_data.split("test"))

    first = lance_attack(tiny_data, tiny_cfg, device="cpu")
    second = lance_attack(alternate, tiny_cfg, device="cpu")
    assert first.n_deleted == second.n_deleted
    assert first.n_injected == second.n_injected
    np.testing.assert_array_equal(first.injected_src, second.injected_src)
    np.testing.assert_array_equal(first.injected_dst, second.injected_dst)
    np.testing.assert_allclose(first.injected_t, second.injected_t)

    cutoff_t = train[2][cut - 1]
    first_prefix = first.t <= cutoff_t
    second_prefix = second.t <= cutoff_t
    np.testing.assert_array_equal(first.src[first_prefix], second.src[second_prefix])
    np.testing.assert_array_equal(first.dst[first_prefix], second.dst[second_prefix])


def test_k2_never_modifies_hidden_suffix(tiny_data, tiny_cfg):
    tiny_cfg.attack.knowledge = "k2"
    tiny_cfg.attack.lk_cutoff_frac = 0.6
    train = tiny_data.split("train")
    cut = int(tiny_cfg.attack.lk_cutoff_frac * len(train[0]))
    result = lance_attack(tiny_data, tiny_cfg, device="cpu")
    suffix_mask = result.t > train[2][cut - 1]
    np.testing.assert_array_equal(result.src[suffix_mask], train[0][cut:])
    np.testing.assert_array_equal(result.dst[suffix_mask], train[1][cut:])
    assert not result.adv_mask[suffix_mask].any()
