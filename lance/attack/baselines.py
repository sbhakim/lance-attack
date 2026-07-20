"""Reference attacks for the benchmark, each returning an :class:`AttackResult`.

  ``random``         random deletions and random injections
  ``random_delete``  random observed-edge deletions only
  ``random_inject``  random valid injections only
  ``degree``         injections onto high-degree hubs (no surrogate)
  ``tspear``         injection-only, lowest-likelihood edges among recent nodes;
                     a stand-in for T-SPEAR, without importance or deletion

The importance-targeted hybrid attack, HIA, lives in ``hia.py``, together with an
adaptive variant (``adaptive=True``) intended to evade the C1/C2 screening.
"""
from __future__ import annotations

import numpy as np

from lance.attack.hia import AttackResult
from lance.attack.candidates import eligible_sources, filter_candidate_events
from lance.attack.diagnostics import injection_diagnostics


def _assemble(src, dst, t, feat, keep, inj_s, inj_d, inj_t, n_del, inj_feat=None,
              diagnostics=None):
    if inj_feat is None:
        inj_feat = np.zeros((len(inj_s), feat.shape[1]), dtype=np.float32)
    new_src = np.concatenate([src[keep], inj_s]).astype(np.int64)
    new_dst = np.concatenate([dst[keep], inj_d]).astype(np.int64)
    new_t = np.concatenate([t[keep], inj_t]).astype(np.float64)
    new_feat = np.concatenate([feat[keep], inj_feat]).astype(np.float32)
    adv = np.concatenate([np.zeros(int(keep.sum()), bool), np.ones(len(inj_s), bool)])
    order = np.argsort(new_t, kind="stable")
    result = AttackResult(
        new_src[order], new_dst[order], new_t[order], new_feat[order],
        adv[order], int(n_del), int(len(inj_s)),
        inj_s.astype(np.int64), inj_d.astype(np.int64), inj_t.astype(np.float64))
    if diagnostics is not None:
        result.diagnostics = diagnostics
    return result


def random_attack(src, dst, t, feat, num_nodes, ptb_rate=0.1, seed=0, **_):
    rng = np.random.default_rng(seed)
    n = len(src)
    budget = int(ptb_rate * n)
    n_del = budget // 2
    n_inj = budget - n_del
    keep = np.ones(n, bool)
    keep[rng.choice(n, size=min(n_del, n), replace=False)] = False
    pool = max(3 * n_inj, 1)
    inj_s = rng.choice(eligible_sources(src), size=pool)
    inj_d = rng.choice(np.unique(dst), size=pool)
    ti = rng.integers(0, n, size=pool)
    inj_t = t[ti].astype(np.float64)
    valid = filter_candidate_events(inj_s, inj_d, inj_t,
                                    set(zip(src.tolist(), dst.tolist(), t.tolist())))[:n_inj]
    sel_s, sel_d, sel_t, sel_feat = inj_s[valid], inj_d[valid], inj_t[valid], feat[ti[valid]]
    diag = {"injection_edge_diagnostics": injection_diagnostics(
        src, dst, t, sel_s, sel_d, sel_t, num_nodes=num_nodes, inj_feat=sel_feat)}
    return _assemble(src, dst, t, feat, keep, sel_s, sel_d, sel_t,
                     n_del, sel_feat, diag)


def random_delete_attack(src, dst, t, feat, num_nodes, ptb_rate=0.1, seed=0, **_):
    rng = np.random.default_rng(seed)
    n = len(src)
    budget = min(int(ptb_rate * n), n)
    keep = np.ones(n, bool)
    if budget > 0:
        keep[rng.choice(n, size=budget, replace=False)] = False
    empty = np.array([], dtype=np.int64)
    return _assemble(src, dst, t, feat, keep, empty, empty,
                     np.array([], dtype=np.float64), budget)


def random_inject_attack(src, dst, t, feat, num_nodes, ptb_rate=0.1, seed=0, **_):
    rng = np.random.default_rng(seed)
    n = len(src)
    budget = int(ptb_rate * n)
    pool = max(5 * budget, 1)
    inj_s = rng.choice(eligible_sources(src), size=pool)
    inj_d = rng.choice(np.unique(dst), size=pool)
    ti = rng.integers(0, n, size=pool)
    inj_t = t[ti].astype(np.float64)
    valid = filter_candidate_events(inj_s, inj_d, inj_t,
                                    set(zip(src.tolist(), dst.tolist(), t.tolist())))[:budget]
    keep = np.ones(n, bool)
    sel_s, sel_d, sel_t, sel_feat = inj_s[valid], inj_d[valid], inj_t[valid], feat[ti[valid]]
    diag = {"injection_edge_diagnostics": injection_diagnostics(
        src, dst, t, sel_s, sel_d, sel_t, num_nodes=num_nodes, inj_feat=sel_feat)}
    return _assemble(src, dst, t, feat, keep, sel_s, sel_d, sel_t,
                     0, sel_feat, diag)


def degree_attack(src, dst, t, feat, num_nodes, ptb_rate=0.1, seed=0, **_):
    rng = np.random.default_rng(seed)
    n = len(src)
    n_inj = int(ptb_rate * n)
    deg = np.bincount(np.concatenate([src, dst]), minlength=num_nodes).astype(float)
    source_pool = eligible_sources(src)
    p = deg[source_pool]
    p = p / p.sum() if p.sum() > 0 else None
    pool = max(3 * n_inj, 1)
    inj_s = rng.choice(source_pool, size=pool, p=p)            # target valid source hubs
    inj_d = rng.choice(np.unique(dst), size=pool)
    ti = rng.integers(0, n, size=pool)
    inj_t = t[ti].astype(np.float64)
    valid = filter_candidate_events(inj_s, inj_d, inj_t,
                                    set(zip(src.tolist(), dst.tolist(), t.tolist())))[:n_inj]
    keep = np.ones(n, bool)
    sel_s, sel_d, sel_t, sel_feat = inj_s[valid], inj_d[valid], inj_t[valid], feat[ti[valid]]
    diag = {"injection_edge_diagnostics": injection_diagnostics(
        src, dst, t, sel_s, sel_d, sel_t, num_nodes=num_nodes, inj_feat=sel_feat)}
    return _assemble(src, dst, t, feat, keep, sel_s, sel_d, sel_t,
                     0, sel_feat, diag)


def tspear_attack(src, dst, t, feat, num_nodes, score_fn, ptb_rate=0.1,
                  inj_percentile=10.0, seed=0, **_):
    """Injection-only, lowest-likelihood edges among recently active nodes."""
    rng = np.random.default_rng(seed)
    n = len(src)
    n_inj = int(ptb_rate * n)
    recent = np.unique(src[-n // 2:])  # valid, recently active source endpoints
    pool = max(20 * n_inj, 1)
    cu = rng.choice(recent, size=pool)
    cv = rng.choice(np.unique(dst), size=pool)
    ti = rng.integers(0, n, size=pool)
    ct = t[ti]
    valid = filter_candidate_events(cu, cv, ct,
                                    set(zip(src.tolist(), dst.tolist(), t.tolist())))
    cu, cv, ct, ti = cu[valid], cv[valid], ct[valid], ti[valid]
    ys = score_fn(cu, cv, ct)
    cut = np.percentile(ys, inj_percentile)
    weak = np.where(ys <= cut)[0]
    sel = weak[np.argsort(ys[weak])][:n_inj]
    keep = np.ones(n, bool)
    sel_s, sel_d = cu[sel], cv[sel]
    sel_t, sel_feat, sel_scores = ct[sel].astype(np.float64), feat[ti[sel]], ys[sel]
    diag = {"injection_edge_diagnostics": injection_diagnostics(
        src, dst, t, sel_s, sel_d, sel_t, num_nodes=num_nodes,
        scores=sel_scores, inj_feat=sel_feat)}
    return _assemble(src, dst, t, feat, keep, sel_s, sel_d,
                     sel_t, 0, sel_feat, diag)
