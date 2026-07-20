"""The LANCE attack orchestrator.

LANCE keeps the three HIA components (a surrogate, impact-based node selection,
and hybrid injection/deletion) and adds three elements. Under limited knowledge
(K2) the surrogate is trained only on the event prefix up to the attack time
``t_a`` and cannot observe the post-attack stream; K1 uses the full history as an
upper bound, and K3 is an online, windowed variant. Rather than a fixed 50/50
split, the budget is allocated adaptively: all candidate edits are ranked by a
common priority and spent greedily across deletions and injections. In the K3
setting edits are emitted window by window, re-targeting from the running
surrogate.

``adaptive_hybrid_attack`` is the perturbation core and takes a surrogate
``score_fn`` supplied by the caller (for example the benchmark harness).
``lance_attack`` is the self-contained orchestrator that builds the
limited-knowledge surrogate itself.
"""
from __future__ import annotations

import bisect
import copy

import numpy as np
import torch

from lance.attack.importance import compute_impact
from lance.attack.hia import AttackResult, hia_attack
from lance.attack.baselines import _assemble
from lance.attack.candidates import eligible_sources, filter_candidate_events
from lance.attack.diagnostics import injection_diagnostics
from lance.data.dataset import TemporalGraphData
from lance.models import TGNLite
from lance.training import Trainer
from lance.utils import seed_everything


def _norm(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.ones_like(x) * 0.5


def _summary(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"n": 0}
    finite = x.astype(float)[np.isfinite(x)]
    if len(finite) == 0:
        return {"n": int(len(x)), "finite_n": 0}
    q = np.quantile(finite, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "n": int(len(x)),
        "finite_n": int(len(finite)),
        "min": float(q[0]),
        "q25": float(q[1]),
        "median": float(q[2]),
        "q75": float(q[3]),
        "max": float(q[4]),
        "mean": float(np.mean(finite)),
    }


def _pair_recurrence(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    pairs = np.stack([src, dst], axis=1)
    _, inv, counts = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    return counts[inv].astype(float)


def _future_stats(src: np.ndarray, dst: np.ndarray, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    future_pair = np.zeros(len(src), dtype=float)
    future_src = np.zeros(len(src), dtype=float)
    pair_counts: dict[tuple[int, int], int] = {}
    src_counts = np.zeros(num_nodes, dtype=np.int64)
    for i in range(len(src) - 1, -1, -1):
        u, v = int(src[i]), int(dst[i])
        future_pair[i] = pair_counts.get((u, v), 0)
        future_src[i] = src_counts[u]
        pair_counts[(u, v)] = pair_counts.get((u, v), 0) + 1
        src_counts[u] += 1
    return future_pair, future_src


def _delete_type_diagnostics(src, dst, idx, recurrence, future_pair, future_src) -> dict:
    if len(idx) == 0:
        return {
            "repeated_pair": 0,
            "one_off_pair": 0,
            "future_reused_pair": 0,
            "future_active_source": 0,
            "unique_deleted_sources": 0,
            "unique_deleted_destinations": 0,
        }
    rec = recurrence[idx]
    fp = future_pair[idx]
    fs = future_src[idx]
    return {
        "repeated_pair": int(np.sum(rec > 1)),
        "one_off_pair": int(np.sum(rec <= 1)),
        "future_reused_pair": int(np.sum(fp > 0)),
        "future_active_source": int(np.sum(fs > 0)),
        "unique_deleted_sources": int(len(np.unique(src[idx]))),
        "unique_deleted_destinations": int(len(np.unique(dst[idx]))),
        "future_pair_summary": _summary(fp),
        "future_source_summary": _summary(fs),
    }


def _injection_context_features(src, dst, t, cu, cv, ct, num_nodes):
    source_times: dict[int, list[float]] = {}
    dest_times: dict[int, list[float]] = {}
    pair_times: dict[tuple[int, int], list[float]] = {}
    for u0, v0, ts0 in zip(src.tolist(), dst.tolist(), t.tolist()):
        u, v, ts = int(u0), int(v0), float(ts0)
        source_times.setdefault(u, []).append(ts)
        dest_times.setdefault(v, []).append(ts)
        pair_times.setdefault((u, v), []).append(ts)
    src_degree = np.bincount(src, minlength=num_nodes).astype(float)
    dst_degree = np.bincount(dst, minlength=num_nodes).astype(float)
    source_future = np.zeros(len(cu), dtype=float)
    source_history = np.zeros(len(cu), dtype=float)
    dest_history = np.zeros(len(cu), dtype=float)
    pair_history = np.zeros(len(cu), dtype=float)
    pair_future = np.zeros(len(cu), dtype=float)
    for i, (u0, v0, ts0) in enumerate(zip(cu.tolist(), cv.tolist(), ct.tolist())):
        u, v, ts = int(u0), int(v0), float(ts0)
        st = source_times.get(u, [])
        dt = dest_times.get(v, [])
        pt = pair_times.get((u, v), [])
        s_split = bisect.bisect_left(st, ts)
        d_split = bisect.bisect_left(dt, ts)
        p_split = bisect.bisect_left(pt, ts)
        source_history[i] = s_split
        source_future[i] = len(st) - s_split
        dest_history[i] = d_split
        pair_history[i] = p_split
        pair_future[i] = len(pt) - p_split
    return {
        "source_future": source_future,
        "source_history": source_history,
        "dest_history": dest_history,
        "pair_history": pair_history,
        "pair_future": pair_future,
        "source_degree": src_degree[cu],
        "dest_degree": dst_degree[cv],
    }


def adaptive_hybrid_attack(src, dst, t, feat, num_nodes, impact, score_fn,
                           budget, high_impact_frac=0.1, seed=0,
                           allow_delete=True, allow_inject=True,
                           query_aware_injection=False,
                           grad_scorer=None) -> AttackResult:
    """Spend ``budget`` edits greedily over deletion and injection candidates.

    By default edits are ranked by a heuristic priority (likelihood extremity
    times endpoint impact), so the injection/deletion split is data-driven rather
    than fixed. When ``grad_scorer`` (a
    :class:`~lance.attack.meta.MetaGradientScorer`) is supplied, priorities come
    instead from a first-order estimate of each edit's effect on the victim
    ranking loss, and edits with non-positive estimated effect are discarded.
    Deletion and injection scores share one scale, so the greedy queue draws from
    whichever pool contributes more."""
    rng = np.random.default_rng(seed)
    n = len(src)
    if budget <= 0 or n == 0:
        return _assemble(src, dst, t, feat, np.ones(n, bool),
                         np.array([], np.int64), np.array([], np.int64),
                         np.array([], np.float64), 0)
    thr = np.quantile(impact, 1.0 - high_impact_frac)
    high = np.where(impact >= thr)[0]
    high_set = set(high.tolist())
    dst_pool = np.unique(dst)
    recurrence_all = _pair_recurrence(src, dst)
    future_pair_all, future_src_all = _future_stats(src, dst, num_nodes)

    # deletion candidates. With a grad_scorer every observed edge competes and
    # its priority is the estimated marginal damage of removing it; otherwise
    # candidates are edges touching high-impact nodes, ranked by a heuristic in
    # which pair recurrence and recency prefer repeated/recent interactions.
    if grad_scorer is not None:
        di = np.arange(n, dtype=np.int64) if allow_delete else np.array([], dtype=np.int64)
    else:
        incident = np.array([(u in high_set) or (v in high_set) for u, v in zip(src, dst)])
        di = np.where(incident)[0] if allow_delete else np.array([], dtype=np.int64)
    if len(di):
        yd = score_fn(src[di], dst[di], t[di])
        if grad_scorer is not None:
            raw_d = grad_scorer.deletion_damage(src[di], dst[di], t[di], feat[di])
            prio_d = np.where(raw_d > 0.0, raw_d, -np.inf)
        else:
            impe_d = np.maximum(impact[src[di]], impact[dst[di]])
            recur = recurrence_all[di]
            future_pair = future_pair_all[di]
            future_src = future_src_all[di]
            recency = np.linspace(0.0, 1.0, n, dtype=float)[di] if n > 1 else np.ones(len(di))
            prio_d = (
                0.34 * _norm(yd)
                + 0.18 * _norm(impe_d)
                + 0.16 * _norm(recur)
                + 0.17 * _norm(future_pair)
                + 0.10 * _norm(future_src)
                + 0.05 * recency
            )
    else:
        prio_d = np.array([])
        yd = np.array([])

    # injection candidates: sampled valid non-edges. With a grad_scorer the
    # source pool is unrestricted and edges are ranked by estimated damage;
    # otherwise sources are restricted to high-impact nodes and edges are the
    # low-likelihood tail.
    inj_s = inj_d = inj_t = np.array([], np.int64)
    prio_i = np.array([])
    inj_feat_candidates = np.zeros((0, feat.shape[1]), dtype=np.float32)
    if allow_inject and (len(high) or grad_scorer is not None):
        pool = max(10 * budget, 1)
        source_targets = (eligible_sources(src) if grad_scorer is not None
                          else eligible_sources(src, high, impact))
        cu = rng.choice(source_targets, size=pool)
        cv = rng.choice(dst_pool, size=pool)
        ti = rng.integers(0, len(t), size=pool)
        ct = t[ti]
        valid = filter_candidate_events(
            cu, cv, ct, set(zip(src.tolist(), dst.tolist(), t.tolist())))
        cu, cv, ct, ti = cu[valid], cv[valid], ct[valid], ti[valid]
        inj_feat_candidates = feat[ti]
        yi = score_fn(cu, cv, ct)
        impe_i = np.maximum(impact[cu], impact[cv])
        if grad_scorer is not None:
            raw_i = grad_scorer.injection_damage(cu, cv, ct, inj_feat_candidates)
            prio_i = np.where(raw_i > 0.0, raw_i, -np.inf)
        elif query_aware_injection:
            ctx = _injection_context_features(src, dst, t, cu, cv, ct, num_nodes)
            prio_i = 0.60 * (
                0.34 * _norm(ctx["source_future"])
                + 0.18 * _norm(ctx["pair_history"])
                + 0.16 * _norm(ctx["dest_history"])
                + 0.12 * _norm(ctx["source_degree"])
                + 0.10 * _norm(impe_i)
                + 0.10 * _norm(1.0 - yi)
            )
            inj_gate = yi <= np.quantile(yi, 0.25)
        else:
            # In the corrected ablation, injection-only improved MRR. Keep injections
            # possible, but make them compete as lower-confidence edits until a better
            # injection-specific objective is validated.
            prio_i = 0.35 * (0.65 * _norm(1.0 - yi) + 0.35 * _norm(impe_i))
            inj_gate = yi <= np.quantile(yi, 0.05)
        if grad_scorer is None:                     # grad path is already gated
            prio_i = np.where(inj_gate, prio_i, -np.inf)
    else:
        yi = np.array([])

    # merge both pools and take the top-`budget` edits by priority
    tagged = [("d", p, j) for j, p in enumerate(prio_d) if np.isfinite(p)] + \
             [("i", p, j) for j, p in enumerate(prio_i) if np.isfinite(p)]
    tagged.sort(key=lambda z: z[1], reverse=True)
    chosen = []
    chosen_keys = set()
    target_counts: dict[int, int] = {}
    # Diversity cap: spread edits across target nodes. The grad path targets the
    # whole node set rather than the high-impact subset, so key the cap on the
    # number of distinct candidate nodes there.
    cap_nodes = (len(np.unique(np.concatenate([src, dst])))
                 if grad_scorer is not None else (len(high) or 1))
    target_cap = max(2, int(np.ceil(max(1, budget) / max(1, cap_nodes))))
    for kind, priority, idx in tagged:
        if len(chosen) >= budget:
            break
        if kind == "d":
            endpoints = (int(src[di[idx]]), int(dst[di[idx]]))
        else:
            endpoints = (int(cu[idx]), int(cv[idx]))
        if any(target_counts.get(v, 0) >= target_cap for v in endpoints):
            continue
        chosen.append((kind, priority, idx))
        chosen_keys.add((kind, idx))
        for v in endpoints:
            target_counts[v] = target_counts.get(v, 0) + 1
    # The diversity cap is a preference, not a reason to leave budget unused.
    # Fill the remainder by raw priority after the diversified first pass.
    for kind, priority, idx in tagged:
        if len(chosen) >= budget:
            break
        if (kind, idx) in chosen_keys:
            continue
        if kind == "d":
            endpoints = (int(src[di[idx]]), int(dst[di[idx]]))
        else:
            endpoints = (int(cu[idx]), int(cv[idx]))
        chosen.append((kind, priority, idx))
        chosen_keys.add((kind, idx))
        for v in endpoints:
            target_counts[v] = target_counts.get(v, 0) + 1
    del_sel = np.array([j for k, _, j in chosen if k == "d"], dtype=np.int64)
    inj_sel = np.array([j for k, _, j in chosen if k == "i"], dtype=np.int64)

    keep = np.ones(n, bool)
    if len(del_sel):
        keep[di[del_sel]] = False
    if len(inj_sel):
        inj_s, inj_d, inj_t = cu[inj_sel], cv[inj_sel], ct[inj_sel].astype(np.float64)
    inj_feat = inj_feat_candidates[inj_sel] if len(inj_sel) else None
    result = _assemble(src, dst, t, feat, keep, inj_s, inj_d, inj_t,
                       int(len(del_sel)), inj_feat)
    selected_delete_scores = yd[del_sel] if len(del_sel) else np.array([])
    selected_inject_scores = yi[inj_sel] if len(inj_sel) else np.array([])
    selected_inject_priorities = prio_i[inj_sel] if len(inj_sel) else np.array([])
    # Number of eligible (finite-priority) candidates. Under the grad path these
    # are the edits with positive estimated effect; when fewer than the budget
    # exist, the gate caps the spend, which is recorded rather than forced.
    eligible_deletions = int(np.isfinite(prio_d).sum())
    eligible_injections = int(np.isfinite(prio_i).sum())
    n_selected = int(len(del_sel) + len(inj_sel))
    result.diagnostics = {
        "budget": int(budget),
        "grad_scored": bool(grad_scorer is not None),
        "candidate_deletions": int(len(prio_d)),
        "candidate_injections": int(len(prio_i)),
        "eligible_deletions": eligible_deletions,
        "eligible_injections": eligible_injections,
        "selected_deletions": int(len(del_sel)),
        "selected_injections": int(len(inj_sel)),
        "budget_gate_limited": bool(n_selected < budget
                                    and eligible_deletions + eligible_injections <= n_selected),
        "unique_selected_targets": int(len(target_counts)),
        "target_cap": int(target_cap),
        "delete_score_summary": _summary(yd),
        "inject_score_summary": _summary(yi),
        "delete_priority_summary": _summary(prio_d),
        "inject_priority_summary": _summary(prio_i),
        "selected_delete_score_summary": _summary(selected_delete_scores),
        "selected_inject_score_summary": _summary(selected_inject_scores),
        "query_aware_injection": bool(query_aware_injection),
        "delete_type_diagnostics": _delete_type_diagnostics(
            src, dst, di[del_sel] if len(del_sel) else np.array([], dtype=np.int64),
            recurrence_all, future_pair_all, future_src_all),
        "injection_edge_diagnostics": injection_diagnostics(
            src, dst, t, inj_s, inj_d, inj_t, num_nodes=num_nodes, impact=impact,
            scores=selected_inject_scores, priorities=selected_inject_priorities,
            inj_feat=inj_feat),
    }
    return result


def _build_surrogate_score_fn(obs, num_nodes, num_feats, cfg, device):
    """Train a surrogate TGN on the observed prefix ``obs`` and return a score_fn
    plus the warmed model (the limited-knowledge surrogate)."""
    s, d, t, f = obs
    n = len(s)
    v = max(2, int(0.1 * n))
    sub = TemporalGraphData.from_splits(
        num_nodes, num_feats,
        (s[:n - v], d[:n - v], t[:n - v], f[:n - v]),
        (s[n - v:], d[n - v:], t[n - v:], f[n - v:]),
        (s[n - v:], d[n - v:], t[n - v:], f[n - v:]),
    )
    m = cfg.model
    surr = TGNLite(num_nodes, num_feats, m.memory_dim, m.time_dim,
                   m.embedding_dim, m.predictor_hidden, m.dropout)
    scfg = copy.deepcopy(cfg)
    scfg.defense.mode = "none"
    Trainer(surr, scfg, device=device).fit(sub, defense=None, verbose=False)
    surr.reset_state(device)
    for b in sub.iter_batches("train", cfg.train.batch_size, device):
        if len(b):
            surr.advance_memory(b)

    def score_fn(a, b, c):
        return surr.surrogate_scores(
            torch.as_tensor(a, device=device), torch.as_tensor(b, device=device),
            torch.as_tensor(c, dtype=torch.float32, device=device)).cpu().numpy()
    return score_fn, surr


def lance_attack(data, cfg, device: str = "cpu") -> AttackResult:
    """Self-contained LANCE poisoning of ``data``'s training stream, honoring
    ``cfg.attack.knowledge`` (k1/k2/k3) and ``cfg.attack.adaptive``."""
    a = cfg.attack
    if a.knowledge not in {"k1", "k2", "k3"}:
        raise ValueError(f"unknown LANCE knowledge setting: {a.knowledge}")
    seed_everything(cfg.train.seed, deterministic=True)
    src, dst, t, feat = data.split("train")
    n = len(src)
    cut = n if a.knowledge == "k1" else max(10, int(a.lk_cutoff_frac * n))
    obs = (src[:cut], dst[:cut], t[:cut], feat[:cut])

    score_fn, _ = _build_surrogate_score_fn(obs, data.num_nodes, data.num_feats, cfg, device)
    impact = compute_impact(obs[0], obs[1], data.num_nodes, a.impact_weights, a.betweenness_k)
    budget = int(a.ptb_rate * (cut if a.knowledge == "k2" else n))

    if a.knowledge == "k3":                                  # streaming: per-window edits
        W = max(1, a.stream_windows)
        bounds = np.linspace(0, n, W + 1).astype(int)
        parts = []
        per = max(1, budget // W)
        for w in range(W):
            lo, hi = bounds[w], bounds[w + 1]
            if hi <= lo:
                continue
            parts.append(adaptive_hybrid_attack(
                src[lo:hi], dst[lo:hi], t[lo:hi], feat[lo:hi], data.num_nodes,
                impact, score_fn, per, a.high_impact_frac, seed=cfg.train.seed + w,
                query_aware_injection=a.query_aware_injection))
        return _merge_results(parts, feat.shape[1])

    # K2 is a strict observable-prefix attack: hidden post-attack events are
    # neither scored nor eligible for deletion. They are appended untouched
    # after perturbing the observed prefix.
    attack_src, attack_dst, attack_t, attack_feat = (
        obs if a.knowledge == "k2" else (src, dst, t, feat))
    if a.adaptive:
        result = adaptive_hybrid_attack(
            attack_src, attack_dst, attack_t, attack_feat, data.num_nodes, impact,
            score_fn, budget, a.high_impact_frac, seed=cfg.train.seed,
            query_aware_injection=a.query_aware_injection)
    else:
        result = hia_attack(
            attack_src, attack_dst, attack_t, attack_feat, data.num_nodes, impact, score_fn,
            ptb_rate=a.ptb_rate, del_percentile=a.del_percentile,
            inj_percentile=a.inj_percentile, high_impact_frac=a.high_impact_frac,
            seed=cfg.train.seed)
    if a.knowledge == "k2":
        result = _append_clean_suffix(
            result, src[cut:], dst[cut:], t[cut:], feat[cut:])
    return result


def _append_clean_suffix(result, src, dst, t, feat) -> AttackResult:
    """Append an unseen, unmodified suffix to a poisoned observable prefix."""
    new_src = np.concatenate([result.src, src]).astype(np.int64)
    new_dst = np.concatenate([result.dst, dst]).astype(np.int64)
    new_t = np.concatenate([result.t, t]).astype(np.float64)
    new_feat = np.concatenate([result.feat, feat]).astype(np.float32)
    adv = np.concatenate([result.adv_mask, np.zeros(len(src), dtype=bool)])
    order = np.argsort(new_t, kind="stable")
    return AttackResult(
        new_src[order], new_dst[order], new_t[order], new_feat[order], adv[order],
        result.n_deleted, result.n_injected, result.injected_src,
        result.injected_dst, result.injected_t, result.diagnostics)


def _merge_results(parts, n_feats) -> AttackResult:
    if not parts:
        empty = np.array([], np.int64)
        return AttackResult(empty, empty, np.array([], np.float64),
                            np.zeros((0, n_feats), np.float32), np.array([], bool),
                            0, 0, empty, empty, np.array([], np.float64), {})
    cat = lambda key: np.concatenate([getattr(p, key) for p in parts])  # noqa: E731
    order = np.argsort(cat("t"), kind="stable")
    return AttackResult(
        cat("src")[order], cat("dst")[order], cat("t")[order], cat("feat")[order],
        cat("adv_mask")[order], int(sum(p.n_deleted for p in parts)),
        int(sum(p.n_injected for p in parts)),
        cat("injected_src"), cat("injected_dst"), cat("injected_t"),
        {"windows": [p.diagnostics for p in parts]},
    )
