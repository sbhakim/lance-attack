"""HIA (High Impact Attack): a hybrid delete-and-inject poisoning attack.

Given the training stream, a surrogate scoring function ``score_fn`` (link
likelihoods), and a per-node impact map, HIA spends a budget
``Delta = ptb_rate * |E_train|`` on two operations. It deletes high-likelihood
edges incident to high-impact nodes (the upper likelihood tail), and injects
low-likelihood edges incident to high-impact nodes (the lower tail), drawing
injected timestamps from the observed time distribution. The re-implementation
here also serves as the inner attacker for the C3 defense component.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from lance.attack.candidates import eligible_sources, filter_candidate_events

ScoreFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class AttackResult:
    src: np.ndarray
    dst: np.ndarray
    t: np.ndarray
    feat: np.ndarray
    adv_mask: np.ndarray       # bool: True where the edge is adversarial (injected)
    n_deleted: int
    n_injected: int
    injected_src: np.ndarray   # the injected edges alone (used as C3 hard negatives)
    injected_dst: np.ndarray
    injected_t: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def hia_attack(src: np.ndarray, dst: np.ndarray, t: np.ndarray, feat: np.ndarray,
               num_nodes: int, impact: np.ndarray, score_fn: ScoreFn,
               ptb_rate: float = 0.1, del_percentile: float = 85.0,
               inj_percentile: float = 10.0, high_impact_frac: float = 0.1,
               seed: int = 0, adaptive: bool = False) -> AttackResult:
    """HIA poisoning. If ``adaptive`` is True the attack is *defense-aware*: it
    injects *mid-likelihood* edges with *recent* timestamps so they evade both
    C1's low-likelihood screen and C2's staleness band -- the tier-2 stress test.
    """
    rng = np.random.default_rng(seed)
    n = len(src)
    budget = int(ptb_rate * n)
    n_del = budget // 2
    n_inj = budget - n_del

    thr = np.quantile(impact, 1.0 - high_impact_frac)
    high = set(np.where(impact >= thr)[0].tolist())
    dst_pool = np.unique(dst)

    # ----- deletions: high-likelihood edges touching high-impact nodes --------
    incident = np.array([(u in high) or (v in high) for u, v in zip(src, dst)])
    cand_idx = np.where(incident)[0]
    to_delete = np.array([], dtype=np.int64)
    if len(cand_idx) > 0 and n_del > 0:
        yhat = score_fn(src[cand_idx], dst[cand_idx], t[cand_idx])
        cut = np.percentile(yhat, del_percentile)
        strong = cand_idx[yhat >= cut]
        order = np.argsort(-score_fn(src[strong], dst[strong], t[strong])) if len(strong) else []
        to_delete = strong[order][:n_del] if len(strong) else np.array([], dtype=np.int64)

    keep = np.ones(n, dtype=bool)
    keep[to_delete] = False

    # ----- injections: low-likelihood edges touching high-impact nodes --------
    inj_s = inj_d = inj_t = np.array([], dtype=np.int64)
    if n_inj > 0 and len(high) > 0:
        pool = max(20 * n_inj, 1)
        source_targets = eligible_sources(src, high, impact)
        cand_u = rng.choice(source_targets, size=pool)
        cand_v = rng.choice(dst_pool, size=pool)
        if adaptive:
            # recent timestamps -> small staleness -> inside C2's band
            t_recent = t[int(0.8 * len(t)):]
            time_pool = t_recent if len(t_recent) else t
            base = int(0.8 * len(t)) if len(t_recent) else 0
            cand_ti = rng.integers(0, len(time_pool), size=pool) + base
        else:
            cand_ti = rng.integers(0, len(t), size=pool)
        cand_t = t[cand_ti]
        valid = filter_candidate_events(
            cand_u, cand_v, cand_t, set(zip(src.tolist(), dst.tolist(), t.tolist())))
        cand_u, cand_v, cand_t, cand_ti = (
            cand_u[valid], cand_v[valid], cand_t[valid], cand_ti[valid])
        ys = score_fn(cand_u, cand_v, cand_t)
        if adaptive:
            # mid-likelihood edges -> low (1-yhat)*Imp -> evade C1's injection screen
            sel = np.argsort(np.abs(ys - np.median(ys)))[:n_inj]
        else:
            cut = np.percentile(ys, inj_percentile)
            weak = np.where(ys <= cut)[0]
            sel = weak[np.argsort(ys[weak])][:n_inj]
        inj_s, inj_d, inj_t = cand_u[sel], cand_v[sel], cand_t[sel].astype(np.float64)
        inj_feat = feat[cand_ti[sel]].astype(np.float32)
    else:
        inj_feat = np.zeros((0, feat.shape[1]), dtype=np.float32)


    # ----- assemble the perturbed, time-ordered stream ------------------------
    new_src = np.concatenate([src[keep], inj_s]).astype(np.int64)
    new_dst = np.concatenate([dst[keep], inj_d]).astype(np.int64)
    new_t = np.concatenate([t[keep], inj_t]).astype(np.float64)
    new_feat = np.concatenate([feat[keep], inj_feat]).astype(np.float32)
    adv = np.concatenate([np.zeros(keep.sum(), bool), np.ones(len(inj_s), bool)])

    order = np.argsort(new_t, kind="stable")
    return AttackResult(
        src=new_src[order], dst=new_dst[order], t=new_t[order], feat=new_feat[order],
        adv_mask=adv[order], n_deleted=int(len(to_delete)), n_injected=int(len(inj_s)),
        injected_src=inj_s.astype(np.int64), injected_dst=inj_d.astype(np.int64),
        injected_t=inj_t.astype(np.float64),
    )
