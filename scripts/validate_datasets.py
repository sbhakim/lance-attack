"""Run the same paired attack benchmark across several dataset configs."""
from __future__ import annotations

import argparse
import json
import os

try:
    from scripts._common import Config
except ModuleNotFoundError:
    from _common import Config

from lance.experiment import GridSpec, run_grid, to_markdown


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--attacks", nargs="+", default=["none", "random", "hia", "lance"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--ptb-rate", type=float, default=None)
    ap.add_argument("--hist-neg", type=float, default=0.7)
    ap.add_argument("--out", default="artifacts/validation")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results = []
    for config_path in args.configs:
        cfg = Config.from_yaml(config_path)
        if args.epochs is not None:
            cfg.train.epochs = args.epochs
        if args.max_events is not None:
            cfg.data.max_events = args.max_events
        if args.ptb_rate is not None:
            cfg.attack.ptb_rate = args.ptb_rate
        cfg.eval.historical_neg_frac = args.hist_neg
        result = run_grid(
            cfg, GridSpec(attacks=args.attacks, defenses=["none"], seeds=args.seeds))
        results.append(result)
        stem = f"validation_{cfg.data.name}"
        with open(os.path.join(args.out, f"{stem}.json"), "w") as fh:
            json.dump(result, fh, indent=2)
        with open(os.path.join(args.out, f"{stem}.md"), "w") as fh:
            fh.write(to_markdown(result) + "\n")

    with open(os.path.join(args.out, "validation_all.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("\n\n".join(to_markdown(result) for result in results))


if __name__ == "__main__":
    main()
