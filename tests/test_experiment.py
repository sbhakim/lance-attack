"""Experiment-harness invariants needed for paired attack comparisons."""

import numpy as np

import lance.experiment as exp
from lance.data.dataset import TemporalGraphData
from lance.experiment import GridSpec, _train_test, paired_degradation, run_grid


def _k2_cfg(tiny_cfg):
    cfg = tiny_cfg
    cfg.attack.knowledge = "k2"
    cfg.attack.lk_cutoff_frac = 0.5
    cfg.attack.ptb_rate = 0.2
    cfg.defense.mode = "none"
    return cfg


def test_k2_benchmark_budget_is_prefix_derived(tiny_data, tiny_cfg, monkeypatch):
    """Under K2 the benchmark must spend a prefix budget, not a full-stream one.

    The harness previously always trained the surrogate on the whole training
    split, so every recorded run was K1 regardless of the configured regime.
    """
    monkeypatch.setattr(exp, "load_dataset", lambda *a, **k: tiny_data)
    cfg = _k2_cfg(tiny_cfg)
    spec = GridSpec(attacks=["none", "lance"], defenses=["none"], seeds=[0])
    result = run_grid(cfg, spec, device="cpu")

    n_train = len(tiny_data.split("train")[0])
    cut = max(10, int(cfg.attack.lk_cutoff_frac * n_train))
    prefix_budget = int(cfg.attack.ptb_rate * cut)
    edits = result["attack_edits"]["lance"][0]
    assert edits["deleted"] + edits["injected"] <= prefix_budget
    assert prefix_budget < int(cfg.attack.ptb_rate * n_train)


def test_k2_matched_budget_spends_the_k1_edit_count(tiny_data, tiny_cfg, monkeypatch):
    """K2 must be comparable to K1 without conflating knowledge with edit count.

    Under the default prefix budget a K2 attacker both knows less and edits less,
    so a K1-vs-K2 gap cannot be attributed to knowledge. The matched convention
    holds the edit count fixed.
    """
    monkeypatch.setattr(exp, "load_dataset", lambda *a, **k: tiny_data)
    spec = GridSpec(attacks=["none", "lance"], defenses=["none"], seeds=[0])

    cfg = _k2_cfg(tiny_cfg)
    cfg.attack.k2_budget = "matched"
    k2 = run_grid(cfg, spec, device="cpu")["attack_edits"]["lance"][0]

    cfg = _k2_cfg(tiny_cfg)
    cfg.attack.knowledge = "k1"
    k1 = run_grid(cfg, spec, device="cpu")["attack_edits"]["lance"][0]

    spent = lambda e: e["deleted"] + e["injected"]  # noqa: E731
    assert abs(spent(k2) - spent(k1)) <= 1          # integer-rounding slack only


def test_k2_benchmark_ignores_the_hidden_suffix(tiny_data, tiny_cfg, monkeypatch):
    """Changing events after t_a must not change the selected perturbation."""
    src, dst, t, feat = (tiny_data.src.copy(), tiny_data.dst.copy(),
                         tiny_data.t.copy(), tiny_data.feat.copy())
    n_train = len(tiny_data.train_idx)
    cut = max(10, int(0.5 * n_train))
    # rewrite every hidden-suffix endpoint/feature, keeping timestamps (and so
    # the prefix boundary) intact
    dst[cut:] = tiny_data.num_nodes - 1 - dst[cut:]
    feat[cut:] = -feat[cut:]
    altered = TemporalGraphData(src, dst, t, feat, tiny_data.num_nodes, 0.15, 0.15)

    edits = []
    for data in (tiny_data, altered):
        monkeypatch.setattr(exp, "load_dataset", lambda *a, _d=data, **k: _d)
        spec = GridSpec(attacks=["none", "lance"], defenses=["none"], seeds=[0])
        result = run_grid(_k2_cfg(tiny_cfg), spec, device="cpu")
        edits.append(result["attack_edits"]["lance"][0])

    assert edits[0]["deleted"] == edits[1]["deleted"]
    assert edits[0]["injected"] == edits[1]["injected"]
    assert (edits[0]["diagnostics"]["selected_delete_score_summary"]
            == edits[1]["diagnostics"]["selected_delete_score_summary"])


def test_repeated_condition_is_exactly_reproducible(tiny_data, tiny_cfg):
    history = None
    dst_pool = tiny_data.split("train")[1]
    first = _train_test(tiny_cfg, tiny_data, "none", "cpu", history, dst_pool)
    second = _train_test(tiny_cfg, tiny_data, "none", "cpu", history, dst_pool)
    assert first == second


def test_fixed_negative_pool_survives_deleted_destination(tiny_data, tiny_cfg):
    clean_pool = tiny_data.split("train")[1]
    reduced_pool = clean_pool[clean_pool != clean_pool[0]]
    assert len(np.unique(reduced_pool)) <= len(np.unique(clean_pool))
    # The public test path accepts the clean pool even if the condition's own
    # training pool is smaller; completing without error is the contract.
    metrics = _train_test(tiny_cfg, tiny_data, "none", "cpu", None, clean_pool)
    assert "mrr" in metrics


def test_paired_degradation_reports_effect_and_interval():
    result = paired_degradation([0.50, 0.60, 0.55, 0.65, 0.58],
                                [0.42, 0.49, 0.48, 0.52, 0.50])
    assert result["mean"] > 0.0
    assert result["ci95_low"] > 0.0
    assert 0.0 <= result["paired_t_p"] <= 1.0
    assert len(result["per_seed"]) == 5
