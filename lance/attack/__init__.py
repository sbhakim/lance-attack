"""Node-importance scoring, the HIA/LANCE attacks, baselines, and a dispatcher.

LANCE is this project's contribution: an enhanced, limited-knowledge, adaptive
poisoning attack built on HIA. ``lance_attack`` is the self-contained orchestrator
(it trains its own limited-knowledge surrogate); ``adaptive_hybrid_attack`` is the
perturbation core callable with a passed surrogate ``score_fn``.
"""
from __future__ import annotations

import numpy as np

from lance.attack.importance import compute_impact
from lance.attack.hia import hia_attack, AttackResult
from lance.attack.baselines import (
    random_attack,
    random_delete_attack,
    random_inject_attack,
    degree_attack,
    tspear_attack,
)
from lance.attack.lance import lance_attack, adaptive_hybrid_attack

__all__ = ["compute_impact", "hia_attack", "AttackResult", "lance_attack",
           "adaptive_hybrid_attack", "run_attack", "ATTACKS"]

# Attacks usable by the benchmark. "none" = leave the graph clean.
# "lance" = LANCE's adaptive-budget hybrid perturbation (uses the passed surrogate).
ATTACKS = ["none", "random", "random_delete", "random_inject",
           "degree", "tspear", "hia", "hia_adaptive",
           "lance", "lance_fixed", "lance_inject", "lance_delete",
           "lance_random_target", "lance_query", "lance_query_inject",
           "lance_meta", "lance_meta_inject", "lance_meta_delete",
           "lance_meta_hard"]


def run_attack(name: str, src, dst, t, feat, num_nodes, *, impact=None,
               score_fn=None, ptb_rate=0.1, seed=0,
               high_impact_frac=0.1, del_percentile=85.0,
               inj_percentile=10.0, grad_scorer=None) -> AttackResult | None:
    """Dispatch by name. Returns ``None`` for ``none`` (no perturbation).

    For the *full* LANCE attack with limited-knowledge surrogate and the K1/K2/K3
    settings, use ``lance_attack(data, cfg, device)`` directly; ``run_attack("lance",
    ...)`` here applies LANCE's adaptive-budget perturbation with a supplied
    surrogate, for apples-to-apples comparison inside the benchmark harness.
    """
    if name == "none":
        return None
    if name == "random":
        return random_attack(src, dst, t, feat, num_nodes, ptb_rate, seed)
    if name == "random_delete":
        return random_delete_attack(src, dst, t, feat, num_nodes, ptb_rate, seed)
    if name == "random_inject":
        return random_inject_attack(src, dst, t, feat, num_nodes, ptb_rate, seed)
    if name == "degree":
        return degree_attack(src, dst, t, feat, num_nodes, ptb_rate, seed)
    if name == "tspear":
        assert score_fn is not None, "tspear needs a surrogate score_fn"
        return tspear_attack(src, dst, t, feat, num_nodes, score_fn, ptb_rate, seed=seed)
    if name in ("hia", "hia_adaptive"):
        assert impact is not None and score_fn is not None, "hia needs impact + score_fn"
        return hia_attack(
            src, dst, t, feat, num_nodes, impact, score_fn, ptb_rate=ptb_rate,
            del_percentile=del_percentile, inj_percentile=inj_percentile,
            high_impact_frac=high_impact_frac, seed=seed,
            adaptive=(name == "hia_adaptive"))
    if name == "lance_fixed":
        assert impact is not None and score_fn is not None, "lance_fixed needs impact + score_fn"
        return hia_attack(
            src, dst, t, feat, num_nodes, impact, score_fn, ptb_rate=ptb_rate,
            del_percentile=del_percentile, inj_percentile=inj_percentile,
            high_impact_frac=high_impact_frac, seed=seed)
    if name in {"lance", "lance_inject", "lance_delete", "lance_random_target",
                "lance_query", "lance_query_inject"}:
        assert impact is not None and score_fn is not None, "lance needs impact + score_fn"
        target_impact = np.ones_like(impact) if name == "lance_random_target" else impact
        return adaptive_hybrid_attack(
            src, dst, t, feat, num_nodes, target_impact, score_fn,
            int(ptb_rate * len(src)), high_impact_frac, seed=seed,
            allow_delete=name not in {"lance_inject", "lance_query_inject"},
            allow_inject=name != "lance_delete",
            query_aware_injection=name in {"lance_query", "lance_query_inject"})
    if name in {"lance_meta", "lance_meta_inject", "lance_meta_delete",
                "lance_meta_hard"}:
        assert score_fn is not None, "lance_meta needs a surrogate score_fn"
        assert grad_scorer is not None, "lance_meta needs a MetaGradientScorer"
        imp = np.ones(num_nodes) if impact is None else impact
        return adaptive_hybrid_attack(
            src, dst, t, feat, num_nodes, imp, score_fn,
            int(ptb_rate * len(src)), high_impact_frac, seed=seed,
            allow_delete=name != "lance_meta_inject",
            allow_inject=name != "lance_meta_delete",
            grad_scorer=grad_scorer,
            hard_unlearn=name == "lance_meta_hard")
    raise ValueError(f"unknown attack: {name}")
