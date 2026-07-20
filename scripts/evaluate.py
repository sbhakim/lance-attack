"""CLI: the full clean -> attacked -> defended comparison for one dataset.

Trains a clean victim, poisons the train graph with HIA, then trains an
undefended victim and a DT-SHIELD-defended victim on the poisoned graph, and
reports robustness recovery. This is the core experimental table, in miniature.

Example:
    python scripts/evaluate.py --config configs/mooc.yaml
"""
from __future__ import annotations

import argparse
import json
import os

try:
    from scripts._common import Config, build_model, load_data, perturb_train
    from scripts.run_attack import make_score_fn
except ModuleNotFoundError:
    from _common import Config, build_model, load_data, perturb_train
    from run_attack import make_score_fn

from lance.attack import compute_impact, hia_attack
from lance.training import Trainer, resolve_device
from lance.defense import build_defense
from lance.utils import seed_everything, get_logger

_LOG = get_logger()


def _train_test(cfg, data, device, defense_mode):
    cfg.defense.mode = defense_mode
    model = build_model(cfg, data)
    defense = build_defense(cfg, device=device)
    tr = Trainer(model, cfg, device=device)
    tr.fit(data, defense=defense, verbose=False)
    return tr.test(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ptb-rate", type=float, default=None)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.ptb_rate is not None:
        cfg.attack.ptb_rate = args.ptb_rate
    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    data = load_data(cfg)

    clean = _train_test(cfg, data, device, "none")
    _LOG.info(f"[clean]    MRR={clean['mrr']:.4f}")

    # surrogate + HIA poisoning of the train stream
    surrogate = build_model(cfg, data)
    Trainer(surrogate, cfg, device=device).fit(data, defense=None, verbose=False)
    surrogate.reset_state(device)
    for b in data.iter_batches("train", cfg.train.batch_size, device):
        if len(b):
            surrogate.advance_memory(b)
    s, d, t, f = data.split("train")
    impact = compute_impact(s, d, data.num_nodes, cfg.attack.impact_weights,
                            cfg.attack.betweenness_k)
    res = hia_attack(s, d, t, f, data.num_nodes, impact, make_score_fn(surrogate, device),
                     ptb_rate=cfg.attack.ptb_rate, seed=cfg.train.seed)
    poisoned = perturb_train(data, res)

    attacked = _train_test(cfg, poisoned, device, "none")
    defended = _train_test(cfg, poisoned, device, "dtshield")
    gap = clean["mrr"] - attacked["mrr"]
    recovery = (defended["mrr"] - attacked["mrr"]) / gap if gap > 1e-6 else float("nan")

    _LOG.info(f"[attacked] MRR={attacked['mrr']:.4f}  (HIA del={res.n_deleted} inj={res.n_injected})")
    _LOG.info(f"[defended] MRR={defended['mrr']:.4f}  -> robustness recovery={recovery:.2%}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"compare_{cfg.data.name}.json"), "w") as fh:
        json.dump({"clean": clean, "attacked": attacked, "defended": defended,
                   "recovery": recovery, "n_deleted": res.n_deleted,
                   "n_injected": res.n_injected}, fh, indent=2)


if __name__ == "__main__":
    main()
