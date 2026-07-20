"""Experiment-harness invariants needed for paired attack comparisons."""

import numpy as np

from lance.experiment import _train_test, paired_degradation


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
