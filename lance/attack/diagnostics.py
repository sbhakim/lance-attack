"""Read-only summaries of the edits an attack has selected.

These helpers run after selection and do not affect which edits are chosen. The
caller passes the clean training stream and the selected injected edges, and the
returned dictionary is JSON-serializable.
"""
from __future__ import annotations

import bisect

import numpy as np


def summary(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"n": 0}
    finite = np.asarray(x, dtype=float)
    finite = finite[np.isfinite(finite)]
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


def injection_diagnostics(
    src: np.ndarray,
    dst: np.ndarray,
    t: np.ndarray,
    inj_s: np.ndarray,
    inj_d: np.ndarray,
    inj_t: np.ndarray,
    *,
    num_nodes: int,
    impact: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    priorities: np.ndarray | None = None,
    inj_feat: np.ndarray | None = None,
) -> dict:
    """Describe selected injected edges relative to the clean training stream.

    The diagnostics are designed for post-hoc comparison between LANCE and
    random-inject. They estimate query relevance using source future activity in
    the observed train stream and estimate plausibility using pair/source/dest
    history before the injected timestamp.
    """
    n_inj = len(inj_s)
    if n_inj == 0:
        return {
            "n": 0,
            "per_edge": [],
            "source_future_activity_summary": {"n": 0},
            "pair_future_reuse_summary": {"n": 0},
            "pair_history_summary": {"n": 0},
            "timestamp_position_summary": {"n": 0},
        }

    clean_times = np.asarray(t, dtype=float)
    t_min = float(clean_times.min()) if len(clean_times) else 0.0
    t_span = float(clean_times.max() - t_min) if len(clean_times) else 0.0
    src_degree = np.bincount(src, minlength=num_nodes)
    dst_degree = np.bincount(dst, minlength=num_nodes)

    pair_times: dict[tuple[int, int], list[float]] = {}
    source_times: dict[int, list[float]] = {}
    dest_times: dict[int, list[float]] = {}
    for u, v, ts in zip(src.tolist(), dst.tolist(), clean_times.tolist()):
        pair_times.setdefault((int(u), int(v)), []).append(float(ts))
        source_times.setdefault(int(u), []).append(float(ts))
        dest_times.setdefault(int(v), []).append(float(ts))

    def before_after(times: list[float], ts: float) -> tuple[int, int]:
        split = bisect.bisect_left(times, float(ts))
        return split, len(times) - split

    if scores is None:
        scores = np.full(n_inj, np.nan, dtype=float)
    if priorities is None:
        priorities = np.full(n_inj, np.nan, dtype=float)
    if impact is None:
        impact = np.full(num_nodes, np.nan, dtype=float)

    feature_norms = (
        np.linalg.norm(inj_feat.astype(float), axis=1)
        if inj_feat is not None and len(inj_feat) == n_inj else np.full(n_inj, np.nan)
    )

    per_edge = []
    pair_hist, pair_future, source_future, source_hist, dest_hist = [], [], [], [], []
    timestamp_pos, source_deg, dest_deg, max_impact = [], [], [], []
    structurally_valid = []
    for i, (u0, v0, ts0) in enumerate(zip(inj_s.tolist(), inj_d.tolist(), inj_t.tolist())):
        u, v, ts = int(u0), int(v0), float(ts0)
        ph, pf = before_after(pair_times.get((u, v), []), ts)
        sh, sf = before_after(source_times.get(u, []), ts)
        dh, _ = before_after(dest_times.get(v, []), ts)
        sdeg = int(src_degree[u]) if 0 <= u < len(src_degree) else 0
        ddeg = int(dst_degree[v]) if 0 <= v < len(dst_degree) else 0
        imp_u = float(impact[u]) if 0 <= u < len(impact) else float("nan")
        imp_v = float(impact[v]) if 0 <= v < len(impact) else float("nan")
        pos = (ts - t_min) / t_span if t_span > 0 else 0.0
        valid = bool(u != v and sdeg > 0)

        pair_hist.append(ph)
        pair_future.append(pf)
        source_hist.append(sh)
        source_future.append(sf)
        dest_hist.append(dh)
        timestamp_pos.append(pos)
        source_deg.append(sdeg)
        dest_deg.append(ddeg)
        max_impact.append(max(imp_u, imp_v))
        structurally_valid.append(valid)
        per_edge.append({
            "src": u,
            "dst": v,
            "t": ts,
            "score": float(scores[i]),
            "priority": float(priorities[i]),
            "source_degree": sdeg,
            "destination_degree": ddeg,
            "source_impact": imp_u,
            "destination_impact": imp_v,
            "max_endpoint_impact": float(max(imp_u, imp_v)),
            "pair_history_count": int(ph),
            "pair_future_reuse": int(pf),
            "source_history_count": int(sh),
            "source_future_activity": int(sf),
            "destination_history_count": int(dh),
            "timestamp_position": float(pos),
            "feature_norm": float(feature_norms[i]),
            "structurally_valid": valid,
        })

    return {
        "n": int(n_inj),
        "per_edge": per_edge,
        "structurally_valid_count": int(np.sum(structurally_valid)),
        "historically_plausible_pair_count": int(np.sum(np.asarray(pair_hist) > 0)),
        "future_reused_pair_count": int(np.sum(np.asarray(pair_future) > 0)),
        "future_active_source_count": int(np.sum(np.asarray(source_future) > 0)),
        "nonzero_feature_count": int(np.sum(np.asarray(feature_norms) > 0)),
        "source_degree_summary": summary(np.asarray(source_deg)),
        "destination_degree_summary": summary(np.asarray(dest_deg)),
        "endpoint_impact_summary": summary(np.asarray(max_impact)),
        "score_summary": summary(np.asarray(scores)),
        "priority_summary": summary(np.asarray(priorities)),
        "pair_history_summary": summary(np.asarray(pair_hist)),
        "pair_future_reuse_summary": summary(np.asarray(pair_future)),
        "source_history_summary": summary(np.asarray(source_hist)),
        "source_future_activity_summary": summary(np.asarray(source_future)),
        "destination_history_summary": summary(np.asarray(dest_hist)),
        "timestamp_position_summary": summary(np.asarray(timestamp_pos)),
        "feature_norm_summary": summary(np.asarray(feature_norms)),
    }
