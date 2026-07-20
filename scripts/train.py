"""CLI: train a victim TGNN, optionally with the DT-SHIELD defense.

Example:
    python scripts/train.py --config configs/mooc.yaml --defense dtshield --epochs 10
"""
from __future__ import annotations

import argparse
import json
import os

try:
    from scripts._common import Config, build_model, load_data
except ModuleNotFoundError:
    from _common import Config, build_model, load_data

from lance.training import Trainer, resolve_device
from lance.defense import build_defense
from lance.utils import seed_everything, get_logger

_LOG = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--defense", choices=["none", "dtshield"], default=None,
                    help="override config.defense.mode")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.defense is not None:
        cfg.defense.mode = args.defense
    if args.epochs is not None:
        cfg.train.epochs = args.epochs

    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    data = load_data(cfg)
    model = build_model(cfg, data)
    defense = build_defense(cfg, device=device)
    _LOG.info(f"dataset={cfg.data.name} nodes={data.num_nodes} "
              f"defense={cfg.defense.mode} device={device}")

    trainer = Trainer(model, cfg, device=device)
    trainer.fit(data, defense=defense)
    test = trainer.test(data)
    _LOG.info(f"TEST {cfg.data.name} [{cfg.defense.mode}]: "
              f"MRR={test['mrr']:.4f} Hit@{cfg.eval.hits_k}={test[f'hits@{cfg.eval.hits_k}']:.4f}")

    os.makedirs(args.out, exist_ok=True)
    tag = f"{cfg.data.name}_{cfg.defense.mode}"
    with open(os.path.join(args.out, f"metrics_{tag}.json"), "w") as fh:
        json.dump({"config": cfg.to_dict(), "test": test}, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
