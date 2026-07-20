"""Tests for the meta-gradient (damage-aware) edit scorer and ``lance_meta``."""
import json

import numpy as np
import torch

from lance.attack import compute_impact, run_attack
from lance.attack.meta import MetaGradientScorer
from lance.models import TGNLite
from lance.training import Trainer


def _surrogate(data, cfg):
    """Train a tiny surrogate and warm its memory through the train stream."""
    m = cfg.model
    model = TGNLite(data.num_nodes, data.num_feats, m.memory_dim, m.time_dim,
                    m.embedding_dim, m.predictor_hidden, m.dropout)
    Trainer(model, cfg, device="cpu").fit(data, defense=None, verbose=False)
    model.reset_state("cpu")
    for b in data.iter_batches("train", cfg.train.batch_size, "cpu"):
        if len(b):
            model.advance_memory(b)
    return model


def _score_fn(model):
    return lambda a, b, c: model.surrogate_scores(
        torch.as_tensor(a), torch.as_tensor(b),
        torch.as_tensor(c, dtype=torch.float32)).cpu().numpy()


def test_meta_scorer_shapes_and_sign(tiny_data, tiny_cfg):
    model = _surrogate(tiny_data, tiny_cfg)
    s, d, t, f = tiny_data.split("train")
    scorer = MetaGradientScorer(model, s, d, t, f, tiny_data.num_nodes, "cpu",
                                hist_frac=0.7, n_queries=64, n_neg=10, seed=0)
    assert scorer.G.shape == (tiny_data.num_nodes, tiny_cfg.model.memory_dim)
    assert torch.isfinite(scorer.G).all()

    inj = scorer.injection_damage(s[:20], d[:20], t[:20], f[:20])
    dele = scorer.deletion_damage(s[:20], d[:20], t[:20], f[:20])
    assert inj.shape == (20,) and np.all(np.isfinite(inj))
    # deleting an event is the sign-flip of injecting the same event
    np.testing.assert_allclose(dele, -inj, rtol=1e-5, atol=1e-6)
    # empty input is handled
    assert scorer.injection_damage(np.array([]), np.array([]),
                                   np.array([]), np.zeros((0, tiny_data.num_feats))).shape == (0,)


def test_lance_meta_dispatch_validity_and_gating(tiny_data, tiny_cfg):
    model = _surrogate(tiny_data, tiny_cfg)
    s, d, t, f = tiny_data.split("train")
    imp = compute_impact(s, d, tiny_data.num_nodes, betweenness_k=20)
    scorer = MetaGradientScorer(model, s, d, t, f, tiny_data.num_nodes, "cpu",
                                n_queries=64, n_neg=10, seed=0)
    res = run_attack("lance_meta", s, d, t, f, tiny_data.num_nodes, impact=imp,
                     score_fn=_score_fn(model), ptb_rate=0.2, seed=0,
                     grad_scorer=scorer)
    budget = int(0.2 * len(s))
    assert 1 <= res.n_deleted + res.n_injected <= budget
    dg = res.diagnostics
    assert dg["grad_scored"] is True
    # gate observability: eligible (positive-damage) counts bound the selection,
    # and budget_gate_limited is a bool that only trips when eligibles run out
    assert dg["eligible_deletions"] >= res.n_deleted
    assert dg["eligible_injections"] >= res.n_injected
    assert isinstance(dg["budget_gate_limited"], bool)
    if res.n_deleted + res.n_injected == budget:
        assert dg["budget_gate_limited"] is False

    # structural validity of injected edges (same guarantees as the heuristic core)
    existing = set(zip(s.tolist(), d.tolist(), t.tolist()))
    injected = list(zip(res.injected_src.tolist(), res.injected_dst.tolist(),
                        res.injected_t.tolist()))
    assert all(u != v for u, v in zip(res.injected_src, res.injected_dst))
    assert not any(edge in existing for edge in injected)
    assert len(injected) == len(set(injected))
    json.dumps(res.diagnostics)  # diagnostics remain JSON-serializable


def test_meta_scorer_is_robust_and_side_effect_free(tiny_data, tiny_cfg):
    model = _surrogate(tiny_data, tiny_cfg)
    # capture parameter grads before scoring; building G must not touch them
    grads_before = [None if p.grad is None else p.grad.clone()
                    for p in model.parameters()]
    s, d, t, f = tiny_data.split("train")
    scorer = MetaGradientScorer(model, s, d, t, f, tiny_data.num_nodes, "cpu",
                                n_queries=32, n_neg=8, seed=1)
    assert torch.isfinite(scorer.G).all()          # nan_to_num guard holds
    for p, g0 in zip(model.parameters(), grads_before):
        if g0 is None:
            assert p.grad is None
        else:
            assert torch.allclose(p.grad, g0)      # param grads untouched

    # degenerate empty stream yields a zero (finite) damage field, no crash
    empty = np.array([], dtype=np.int64)
    empty_scorer = MetaGradientScorer(
        model, empty, empty, np.array([], dtype=np.float64),
        np.zeros((0, tiny_data.num_feats), dtype=np.float32),
        tiny_data.num_nodes, "cpu", seed=0)
    assert torch.isfinite(empty_scorer.G).all()
    assert float(empty_scorer.G.abs().sum()) == 0.0


def test_lance_meta_hard_restricts_to_surprising_injections(tiny_data, tiny_cfg):
    model = _surrogate(tiny_data, tiny_cfg)
    s, d, t, f = tiny_data.split("train")
    imp = compute_impact(s, d, tiny_data.num_nodes, betweenness_k=20)
    scorer = MetaGradientScorer(model, s, d, t, f, tiny_data.num_nodes, "cpu",
                                n_queries=64, n_neg=10, seed=0)
    score = _score_fn(model)
    common = dict(impact=imp, score_fn=score, ptb_rate=0.2, seed=0, grad_scorer=scorer)
    hard = run_attack("lance_meta_hard", s, d, t, f, tiny_data.num_nodes, **common)
    base = run_attack("lance_meta", s, d, t, f, tiny_data.num_nodes, **common)

    assert hard.diagnostics["grad_scored"] is True
    assert hard.n_deleted + hard.n_injected >= 1
    # the hard variant's injected edges sit in the low-likelihood tail: their mean
    # surrogate likelihood should not exceed the unrestricted meta variant's
    if hard.n_injected and base.n_injected:
        hard_yhat = score(hard.injected_src, hard.injected_dst, hard.injected_t).mean()
        base_yhat = score(base.injected_src, base.injected_dst, base.injected_t).mean()
        assert hard_yhat <= base_yhat + 1e-6


def test_lance_meta_components(tiny_data, tiny_cfg):
    model = _surrogate(tiny_data, tiny_cfg)
    s, d, t, f = tiny_data.split("train")
    imp = compute_impact(s, d, tiny_data.num_nodes, betweenness_k=20)
    scorer = MetaGradientScorer(model, s, d, t, f, tiny_data.num_nodes, "cpu",
                                n_queries=64, n_neg=10, seed=0)
    common = dict(impact=imp, score_fn=_score_fn(model), ptb_rate=0.2, seed=0,
                  grad_scorer=scorer)
    inject = run_attack("lance_meta_inject", s, d, t, f, tiny_data.num_nodes, **common)
    delete = run_attack("lance_meta_delete", s, d, t, f, tiny_data.num_nodes, **common)
    assert inject.n_deleted == 0
    assert delete.n_injected == 0
