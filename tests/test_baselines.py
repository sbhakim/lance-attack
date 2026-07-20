"""Contracts for baseline attacks/defenses and the dispatcher."""
import numpy as np

from lance.attack import run_attack, compute_impact, ATTACKS
from lance.defense import build_defense
from lance.eval.metrics import detection_pr
from lance.models import TGNLite
from lance.training import Trainer


def _score_fn():
    rng = np.random.default_rng(0)
    return lambda a, b, c: rng.uniform(size=len(a))


def test_attack_dispatcher_runs_all(tiny_data):
    s, d, t, f = tiny_data.split("train")
    imp = compute_impact(s, d, tiny_data.num_nodes, betweenness_k=20)
    for name in ATTACKS:
        if name.startswith("lance_meta"):
            continue  # needs a differentiable surrogate; covered in test_meta.py
        res = run_attack(name, s, d, t, f, tiny_data.num_nodes, impact=imp,
                         score_fn=_score_fn(), ptb_rate=0.15, seed=0)
        if name == "none":
            assert res is None
        else:
            assert len(res.src) == len(res.dst) == len(res.t)
            assert res.adv_mask.shape[0] == len(res.src)


def test_random_delete_and_inject_baselines_are_component_specific(tiny_data):
    s, d, t, f = tiny_data.split("train")
    budget = int(0.2 * len(s))
    delete = run_attack("random_delete", s, d, t, f, tiny_data.num_nodes,
                        ptb_rate=0.2, seed=0)
    inject = run_attack("random_inject", s, d, t, f, tiny_data.num_nodes,
                        ptb_rate=0.2, seed=0)
    assert delete.n_deleted == budget and delete.n_injected == 0
    assert inject.n_deleted == 0
    assert 0 < inject.n_injected <= budget
    assert len(delete.src) == len(s) - budget
    assert len(inject.src) == len(s) + inject.n_injected
    diag = inject.diagnostics["injection_edge_diagnostics"]
    assert diag["n"] == inject.n_injected
    assert len(diag["per_edge"]) == inject.n_injected
    assert diag["structurally_valid_count"] == inject.n_injected


def test_adaptive_uses_recent_timestamps(tiny_data):
    s, d, t, f = tiny_data.split("train")
    imp = compute_impact(s, d, tiny_data.num_nodes, betweenness_k=20)
    res = run_attack("hia_adaptive", s, d, t, f, tiny_data.num_nodes, impact=imp,
                     score_fn=_score_fn(), ptb_rate=0.2, seed=0)
    if res.n_injected > 0:
        assert res.injected_t.min() >= t[int(0.8 * len(t))] - 1e-6


def test_baseline_defenses_train(tiny_data, tiny_cfg):
    for mode in ("tshield", "cosine"):
        tiny_cfg.defense.mode = mode
        m = TGNLite(tiny_data.num_nodes, tiny_data.num_feats, 16, 8, 16, 16)
        defense = build_defense(tiny_cfg, device="cpu")
        if hasattr(defense, "on_epoch_start"):
            defense.on_epoch_start(1, tiny_cfg.train.epochs)
        hist = Trainer(m, tiny_cfg, device="cpu").fit(tiny_data, defense=defense, verbose=False)
        assert len(hist) == tiny_cfg.train.epochs


def test_detection_pr_perfect():
    # suspicion ranks the two injected edges first -> precision/recall = 1 at q
    suspicion = np.array([0.9, 0.8, 0.1, 0.2, 0.05])
    adv = np.array([True, True, False, False, False])
    pr = detection_pr(suspicion, adv, q=0.4)   # flags top-2
    assert pr["det_recall"] == 1.0 and pr["det_precision"] == 1.0
