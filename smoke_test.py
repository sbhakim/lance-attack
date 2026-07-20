#!/usr/bin/env python
"""End-to-end smoke test: clean -> HIA-attacked -> DT-SHIELD-defended.

Runs on a small slice of a real dataset so it finishes in seconds, and asserts
the pipeline behaves sanely:
  * the clean victim learns (MRR well above random),
  * HIA actually perturbs the graph (deletes and injects edges),
  * DT-SHIELD trains to completion and recovers some of the attacked gap.

Usage:  python smoke_test.py --data-root ../Dataset [--dataset mooc]
"""
from __future__ import annotations

import argparse

import torch

from lance.config import Config
from lance.data import load_dataset
from lance.data.dataset import TemporalGraphData
from lance.models import TGNLite
from lance.training import Trainer, resolve_device
from lance.attack import compute_impact, hia_attack
from lance.defense import build_defense
from lance.utils import seed_everything, get_logger

_LOG = get_logger()


def _cfg(data_root: str, dataset: str) -> Config:
    fmt = "bitcoinotc" if dataset == "bitcoinotc" else "jodie"
    return Config.from_dict({
        "data": {"root": data_root, "name": dataset, "fmt": fmt, "max_events": 15000},
        "model": {"memory_dim": 48, "time_dim": 32, "embedding_dim": 48, "predictor_hidden": 48},
        "train": {"epochs": 3, "batch_size": 200, "device": "auto", "seed": 0},
        "eval": {"num_neg": 100, "hits_k": 10},
        "attack": {"ptb_rate": 0.1},
        "defense": {"mode": "dtshield", "adv_every": 3, "smooth_lambda": 0.05},
    })


def _model(cfg, data):
    return TGNLite(data.num_nodes, data.num_feats, cfg.model.memory_dim, cfg.model.time_dim,
                   cfg.model.embedding_dim, cfg.model.predictor_hidden, cfg.model.dropout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../Dataset")
    ap.add_argument("--dataset", default="mooc")
    args = ap.parse_args()

    seed_everything(0)
    cfg = _cfg(args.data_root, args.dataset)
    device = resolve_device("auto")
    data = load_dataset(cfg.data.root, cfg.data.name, cfg.data.fmt, cfg.data.max_events)
    _LOG.info(f"loaded {cfg.data.name}: {data.num_nodes} nodes, {len(data.src)} events, device={device}")
    random_mrr = 1.0 / (cfg.eval.num_neg + 1)

    # 1) clean victim
    clean_model = _model(cfg, data)
    clean = Trainer(clean_model, cfg, device=device).fit(data, verbose=False)[-1]
    _LOG.info(f"[clean]    val MRR={clean['mrr']:.4f}  (random ~ {random_mrr:.4f})")
    assert clean["mrr"] > 2 * random_mrr, "clean model failed to learn"

    # 2) HIA poisoning (surrogate = the clean victim)
    clean_model.reset_state(device)
    for b in data.iter_batches("train", 200, device):
        if len(b):
            clean_model.advance_memory(b)

    def score_fn(s, d, t):
        return clean_model.surrogate_scores(
            torch.as_tensor(s, device=device), torch.as_tensor(d, device=device),
            torch.as_tensor(t, dtype=torch.float32, device=device)).cpu().numpy()

    s, d, t, f = data.split("train")
    impact = compute_impact(s, d, data.num_nodes, cfg.attack.impact_weights,
                            cfg.attack.betweenness_k)
    res = hia_attack(s, d, t, f, data.num_nodes, impact, score_fn,
                     ptb_rate=cfg.attack.ptb_rate, seed=0)
    _LOG.info(f"[attack]   HIA deleted={res.n_deleted} injected={res.n_injected}")
    assert res.n_deleted > 0 and res.n_injected > 0, "attack did not perturb the graph"

    poisoned = TemporalGraphData.from_splits(
        data.num_nodes, data.num_feats,
        train=(res.src, res.dst, res.t, res.feat),
        val=data.split("val"), test=data.split("test"))

    # 3) attacked vs defended
    cfg.defense.mode = "none"
    atk = Trainer(_model(cfg, data), cfg, device=device).fit(poisoned, verbose=False)[-1]
    cfg.defense.mode = "dtshield"
    dmodel = _model(cfg, data)
    defense = build_defense(cfg, device=device)
    dfd = Trainer(dmodel, cfg, device=device).fit(poisoned, defense=defense, verbose=False)[-1]

    _LOG.info(f"[attacked] val MRR={atk['mrr']:.4f}")
    _LOG.info(f"[defended] val MRR={dfd['mrr']:.4f}")
    _LOG.info("SMOKE TEST PASSED: pipeline runs clean -> attack -> defense end-to-end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
