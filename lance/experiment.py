"""Benchmark harness: a paired defense-by-attack comparison over seeds.

For each seed the harness trains an undefended victim on the clean graph (the
reference), trains a clean surrogate the attacker uses to score edges, poisons the
training stream with each attack, and trains every defense on every poisoned
graph. For each (defense, attack) pair it reports MRR, Hit@k, and AP on the clean
test set, together with robustness recovery,
(MRR_def - MRR_undef) / (MRR_clean - MRR_undef), and clean retention,
MRR_def(clean) / MRR_clean, aggregated as mean and standard deviation over seeds.
The same grid serves attack effectiveness (degradation against baselines, with
defense=none) and, as an extension, DT-SHIELD's robustness recovery.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats

from lance.attack import compute_impact, run_attack
from lance.attack.meta import MetaGradientScorer
from lance.data import load_dataset
from lance.data.dataset import TemporalGraphData
from lance.eval.metrics import build_history
from lance.defense import build_defense
from lance.models import TGNLite
from lance.training import Trainer, resolve_device
from lance.utils import seed_everything, get_logger

_LOG = get_logger()

# Attacks whose edit scoring needs the differentiable surrogate (meta-gradient).
_META_ATTACKS = {"lance_meta", "lance_meta_inject", "lance_meta_delete",
                 "lance_meta_hard"}


@dataclass
class GridSpec:
    attacks: list[str]
    defenses: list[str]
    seeds: list[int]


def _model(cfg, data):
    """The attacker's surrogate is always the memory-based TGNLite."""
    m = cfg.model
    return TGNLite(data.num_nodes, data.num_feats, m.memory_dim, m.time_dim,
                   m.embedding_dim, m.predictor_hidden, m.dropout)


def _victim(cfg, data, victim_cls):
    """Build the victim. It may differ from the surrogate (transfer study);
    only TGNLite consumes the model config, other families use their defaults."""
    if victim_cls is TGNLite:
        return _model(cfg, data)
    return victim_cls(data.num_nodes, data.num_feats)


def _make_score_fn(model, device):
    def fn(s, d, t):
        return model.surrogate_scores(
            torch.as_tensor(s, device=device), torch.as_tensor(d, device=device),
            torch.as_tensor(t, dtype=torch.float32, device=device)).cpu().numpy()
    return fn


def _train_test(cfg, data, defense_mode, device, history=None,
                negative_dst_pool=None, victim_cls=TGNLite):
    cfg = copy.deepcopy(cfg)
    cfg.defense.mode = defense_mode
    # Pair conditions within a seed: model initialization, dropout, defense
    # sampling, and training-time randomness all restart from the same state.
    seed_everything(cfg.train.seed, deterministic=True)
    model = _victim(cfg, data, victim_cls)
    defense = build_defense(cfg, device=device)
    tr = Trainer(model, cfg, device=device)
    tr.fit(data, defense=defense, verbose=False)
    return tr.test(data, history=history, negative_dst_pool=negative_dst_pool)


def run_grid(cfg, spec: GridSpec, device: str | None = None,
             victim_cls=TGNLite) -> dict:
    device = device or resolve_device(cfg.train.device)
    # cell[(defense, attack)] -> list of metric dicts across seeds
    cells: dict[tuple[str, str], list[dict]] = {}
    clean_ref: list[float] = []          # undefended-on-clean MRR per seed
    retention: dict[str, list[float]] = {d: [] for d in spec.defenses}
    attack_edits: dict[str, list[dict]] = {a: [] for a in spec.attacks if a != "none"}

    for seed in spec.seeds:
        seed_everything(seed)
        cfg.train.seed = seed
        data = load_dataset(cfg.data.root, cfg.data.name, cfg.data.fmt,
                            cfg.data.max_events, cfg.data.val_ratio, cfg.data.test_ratio)

        # fixed clean history -> identical historical-negative eval set for all
        # conditions (clean / attacked / defended), so the comparison is fair.
        clean_history = build_history(data)
        clean_dst_pool = data.split("train")[1]

        # reference: undefended victim on clean data
        clean = _train_test(cfg, data, "none", device, clean_history,
                            clean_dst_pool, victim_cls=victim_cls)
        clean_ref.append(clean["mrr"])

        # attacker's surrogate (clean undefended) + importance map
        seed_everything(seed, deterministic=True)
        surrogate = _model(cfg, data)
        Trainer(surrogate, cfg, device=device).fit(data, defense=None, verbose=False)
        surrogate.reset_state(device)
        for b in data.iter_batches("train", cfg.train.batch_size, device):
            if len(b):
                surrogate.advance_memory(b)
        s, d, t, f = data.split("train")
        impact = compute_impact(s, d, data.num_nodes, cfg.attack.impact_weights,
                                cfg.attack.betweenness_k)
        score_fn = _make_score_fn(surrogate, device)

        # Meta-gradient scorer (built once per seed) reuses the warmed surrogate
        # to estimate each edit's marginal effect on the victim ranking loss.
        grad_scorer = None
        if any(a in _META_ATTACKS for a in spec.attacks):
            grad_scorer = MetaGradientScorer(
                surrogate, s, d, t, f, data.num_nodes, device,
                hist_frac=cfg.eval.historical_neg_frac or 0.7, seed=seed)

        # poison the train stream with each attack (val/test stay clean)
        poisoned: dict[str, TemporalGraphData] = {}
        for atk in spec.attacks:
            res = run_attack(atk, s, d, t, f, data.num_nodes, impact=impact,
                             score_fn=score_fn, ptb_rate=cfg.attack.ptb_rate, seed=seed,
                             high_impact_frac=cfg.attack.high_impact_frac,
                             del_percentile=cfg.attack.del_percentile,
                             inj_percentile=cfg.attack.inj_percentile,
                             grad_scorer=grad_scorer)
            if res is not None:
                attack_edits[atk].append({"seed": seed, "deleted": res.n_deleted,
                                          "injected": res.n_injected,
                                          "diagnostics": res.diagnostics})
            poisoned[atk] = data if res is None else TemporalGraphData.from_splits(
                data.num_nodes, data.num_feats, (res.src, res.dst, res.t, res.feat),
                data.split("val"), data.split("test"))

        # every defense on clean (retention) and on each poisoned graph
        for dfn in spec.defenses:
            retention[dfn].append(_train_test(
                cfg, data, dfn, device, clean_history, clean_dst_pool,
                victim_cls=victim_cls)["mrr"])
            for atk in spec.attacks:
                if atk == "none":
                    continue
                m = _train_test(
                    cfg, poisoned[atk], dfn, device, clean_history, clean_dst_pool,
                    victim_cls=victim_cls)
                cells.setdefault((dfn, atk), []).append(m)
        _LOG.info(f"seed {seed} done (clean MRR={clean['mrr']:.4f})")

    return _aggregate(cfg, spec, cells, clean_ref, retention, attack_edits)


def _agg(vals):
    a = np.array(vals, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a))


def paired_degradation(clean, attacked) -> dict:
    """Paired clean-minus-condition MRR summary with 95% CI and tests."""
    clean_a = np.asarray(clean, dtype=float)
    attacked_a = np.asarray(attacked, dtype=float)
    if len(clean_a) != len(attacked_a):
        raise ValueError("paired comparison requires one condition result per clean seed")
    diff = clean_a - attacked_a
    n = len(diff)
    mean = float(diff.mean()) if n else float("nan")
    sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
    if n > 1:
        half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
        t_p = float(stats.ttest_rel(clean_a, attacked_a).pvalue)
        try:
            w_p = float(stats.wilcoxon(diff).pvalue)
        except ValueError:
            w_p = float("nan")
    else:
        half = t_p = w_p = float("nan")
    return {"mean": mean, "std": sd, "ci95_low": mean - half,
            "ci95_high": mean + half, "paired_t_p": t_p,
            "wilcoxon_p": w_p, "per_seed": diff.tolist()}


def _aggregate(cfg, spec, cells, clean_ref, retention, attack_edits) -> dict:
    clean_mu, clean_sd = _agg(clean_ref)
    rows = []
    # undefended-on-attack MRR per attack (denominator anchor for recovery)
    undef = {atk: _agg([m["mrr"] for m in cells[("none", atk)]])[0]
             for atk in spec.attacks if atk != "none" and ("none", atk) in cells}
    k = cfg.eval.hits_k
    for dfn in spec.defenses:
        ret_mu, _ = _agg(retention[dfn])
        for atk in spec.attacks:
            if atk == "none" or (dfn, atk) not in cells:
                continue
            ms = cells[(dfn, atk)]
            mrr_mu, mrr_sd = _agg([m["mrr"] for m in ms])
            hit_mu, _ = _agg([m[f"hits@{k}"] for m in ms])
            gap = clean_mu - undef.get(atk, clean_mu)
            recovery = (mrr_mu - undef.get(atk, mrr_mu)) / gap if abs(gap) > 1e-6 else float("nan")
            rows.append({"defense": dfn, "attack": atk, "mrr": mrr_mu, "mrr_std": mrr_sd,
                         f"hit@{k}": hit_mu, "recovery": recovery, "retention": ret_mu})
            rows[-1]["per_seed"] = ms
            rows[-1]["edits_per_seed"] = attack_edits.get(atk, [])
            rows[-1]["degradation"] = paired_degradation(
                clean_ref, [m["mrr"] for m in ms])
    return {"dataset": cfg.data.name, "clean_mrr": clean_mu, "clean_mrr_std": clean_sd,
            "clean_mrr_per_seed": clean_ref, "retention_per_seed": retention,
            "attack_edits": attack_edits,
            "hits_k": k, "rows": rows, "seeds": spec.seeds,
            "config": cfg.to_dict()}


def to_markdown(result: dict) -> str:
    k = result["hits_k"]
    out = [f"## LANCE benchmark — `{result['dataset']}`",
           f"Clean (undefended) MRR = **{result['clean_mrr']:.4f}** "
           f"± {result['clean_mrr_std']:.4f}  | seeds={result['seeds']}",
           "",
           f"| Defense | Attack | MRR | Hit@{k} | Paired ΔMRR | paired-t p | Recovery | Retention |",
           "|---|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        deg = r["degradation"]
        out.append(f"| {r['defense']} | {r['attack']} | {r['mrr']:.4f}±{r['mrr_std']:.3f} "
                   f"| {r[f'hit@{k}']:.4f} | {deg['mean']:+.4f} "
                   f"| {deg['paired_t_p']:.3g} | {r['recovery']:+.1%} "
                   f"| {r['retention']:.4f} |")
    return "\n".join(out)
