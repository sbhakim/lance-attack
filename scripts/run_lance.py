"""CLI: run the full LANCE poisoning attack and report its effectiveness.

Trains a clean victim (reference), poisons the training graph with LANCE under the
chosen knowledge setting (k1/k2/k3), retrains the victim on the poisoned graph, and
reports the MRR degradation.

Example:
    python scripts/run_lance.py --config configs/mooc.yaml --knowledge k2 --ptb-rate 0.3
"""
from __future__ import annotations

import argparse

try:
    from scripts._common import Config, build_model, load_data, perturb_train
except ModuleNotFoundError:
    from _common import Config, build_model, load_data, perturb_train

from lance.attack import lance_attack
from lance.eval.metrics import build_history
from lance.training import Trainer, resolve_device
from lance.utils import seed_everything, get_logger

_LOG = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--knowledge", choices=["k1", "k2", "k3"], default=None)
    ap.add_argument("--ptb-rate", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--hist-neg", type=float, default=None)
    ap.add_argument("--no-adaptive", action="store_true", help="use fixed 50/50 split")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.knowledge is not None:
        cfg.attack.knowledge = args.knowledge
    if args.ptb_rate is not None:
        cfg.attack.ptb_rate = args.ptb_rate
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.max_events is not None:
        cfg.data.max_events = args.max_events
    if args.hist_neg is not None:
        cfg.eval.historical_neg_frac = args.hist_neg
    if args.no_adaptive:
        cfg.attack.adaptive = False
    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    data = load_data(cfg)
    clean_history = build_history(data)
    clean_dst_pool = data.split("train")[1]

    # reference: clean undefended victim
    clean_model = build_model(cfg, data)
    Trainer(clean_model, cfg, device=device).fit(data, defense=None, verbose=False)
    clean = Trainer(clean_model, cfg, device=device).test(
        data, history=clean_history, negative_dst_pool=clean_dst_pool)

    # LANCE poisoning (self-contained: builds its own limited-knowledge surrogate)
    res = lance_attack(data, cfg, device=device)
    _LOG.info(f"LANCE [{cfg.attack.knowledge}, adaptive={cfg.attack.adaptive}]: "
              f"deleted={res.n_deleted} injected={res.n_injected}")
    poisoned = perturb_train(data, res)

    # Pair victim initialization and training randomness with the clean model.
    seed_everything(cfg.train.seed, deterministic=True)
    victim = build_model(cfg, data)
    Trainer(victim, cfg, device=device).fit(poisoned, defense=None, verbose=False)
    attacked = Trainer(victim, cfg, device=device).test(
        poisoned, history=clean_history, negative_dst_pool=clean_dst_pool)

    drop = clean["mrr"] - attacked["mrr"]
    _LOG.info(f"clean MRR={clean['mrr']:.4f} | attacked MRR={attacked['mrr']:.4f} "
              f"| degradation={drop:+.4f} ({100*drop/max(clean['mrr'],1e-9):+.1f}%)")


if __name__ == "__main__":
    main()
