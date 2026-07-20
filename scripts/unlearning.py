"""CLI: measure how a poisoned victim treats injected edges over training.

For each attack, this poisons the training stream, trains a victim, and records
each epoch the victim's mean predicted probability on the injected edges. A rising
curve means the victim increasingly fits the injected edges as genuine, i.e. it
unlearns their adversarial effect. Comparing a targeted attack (``lance_meta``)
against ``random_inject`` shows whether targeted injections are absorbed faster.

Example:
    python scripts/unlearning.py --config configs/mooc.yaml \
        --max-events 40000 --epochs 12 --ptb-rate 0.3 --hist-neg 0.7
"""
from __future__ import annotations

import argparse
import json
import os

import torch

try:
    from scripts._common import Config, build_model, load_data, perturb_train
except ModuleNotFoundError:
    from _common import Config, build_model, load_data, perturb_train

from lance.attack import compute_impact, run_attack
from lance.attack.meta import MetaGradientScorer
from lance.data.dataset import EdgeBatch
from lance.training import Trainer, resolve_device
from lance.utils import seed_everything, get_logger

_LOG = get_logger()


def _score_fn(model, device):
    return lambda a, b, c: model.surrogate_scores(
        torch.as_tensor(a, device=device), torch.as_tensor(b, device=device),
        torch.as_tensor(c, dtype=torch.float32, device=device)).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--attacks", nargs="+", default=["random_inject", "lance_meta"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--ptb-rate", type=float, default=None)
    ap.add_argument("--hist-neg", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/unlearning")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.max_events is not None:
        cfg.data.max_events = args.max_events
    if args.ptb_rate is not None:
        cfg.attack.ptb_rate = args.ptb_rate
    if args.hist_neg is not None:
        cfg.eval.historical_neg_frac = args.hist_neg
    cfg.train.seed = args.seed
    device = resolve_device(cfg.train.device)

    seed_everything(args.seed, deterministic=True)
    data = load_data(cfg)

    # attacker's surrogate (clean, warmed) + impact map + meta-gradient scorer
    surrogate = build_model(cfg, data)
    Trainer(surrogate, cfg, device=device).fit(data, defense=None, verbose=False)
    surrogate.reset_state(device)
    for b in data.iter_batches("train", cfg.train.batch_size, device):
        if len(b):
            surrogate.advance_memory(b)
    s, d, t, f = data.split("train")
    impact = compute_impact(s, d, data.num_nodes, cfg.attack.impact_weights,
                            cfg.attack.betweenness_k)
    score_fn = _score_fn(surrogate, device)
    grad_scorer = MetaGradientScorer(
        surrogate, s, d, t, f, data.num_nodes, device,
        hist_frac=cfg.eval.historical_neg_frac or 0.7, seed=args.seed)

    curves = {}
    for atk in args.attacks:
        seed_everything(args.seed, deterministic=True)
        res = run_attack(atk, s, d, t, f, data.num_nodes, impact=impact,
                         score_fn=score_fn, ptb_rate=cfg.attack.ptb_rate,
                         seed=args.seed, grad_scorer=grad_scorer)
        monitor = EdgeBatch(
            torch.as_tensor(res.injected_src, dtype=torch.long),
            torch.as_tensor(res.injected_dst, dtype=torch.long),
            torch.as_tensor(res.injected_t, dtype=torch.float32),
            torch.as_tensor(res.feat[res.adv_mask], dtype=torch.float32),
        ).to(device)

        poisoned = perturb_train(data, res)
        seed_everything(args.seed, deterministic=True)
        victim = build_model(cfg, data)
        hist = Trainer(victim, cfg, device=device).fit(
            poisoned, defense=None, verbose=False, monitor_edges=monitor)

        adv = [row.get("adv_edge_score") for row in hist]
        curves[atk] = {"n_injected": int(res.n_injected),
                       "adv_edge_score": adv,
                       "val_mrr": [row["mrr"] for row in hist]}
        _LOG.info(f"{atk}: injected={res.n_injected} "
                  f"adv_edge_score={[round(x, 3) for x in adv]}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "unlearning.json"), "w") as fh:
        json.dump({"config": cfg.to_dict(), "curves": curves}, fh, indent=2)
    _LOG.info(f"wrote {args.out}/unlearning.json")


if __name__ == "__main__":
    main()
