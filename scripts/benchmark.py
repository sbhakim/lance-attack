"""CLI: run the defense x attack benchmark grid and write a comparison table.

Example:
    python scripts/benchmark.py --config configs/mooc.yaml \
        --defenses none tshield cosine dtshield \
        --attacks none hia hia_adaptive tspear --seeds 0 1 --epochs 10
"""
from __future__ import annotations

import argparse
import json
import os

try:  # module/console-script execution
    from scripts._common import Config
except ModuleNotFoundError:  # direct ``python scripts/benchmark.py`` execution
    from _common import Config

from lance.experiment import GridSpec, run_grid, to_markdown
from lance.utils import get_logger

_LOG = get_logger()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--defenses", nargs="+", default=["none", "tshield", "cosine", "dtshield"])
    ap.add_argument("--attacks", nargs="+", default=["none", "hia", "hia_adaptive", "tspear"])
    ap.add_argument("--ablation-suite", action="store_true",
                    help="run LANCE adaptive/fixed/component/targeting ablations")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--ptb-rate", type=float, default=None)
    ap.add_argument("--hist-neg", type=float, default=None,
                    help="fraction of historical negatives at eval (TGB-style, 0..1)")
    ap.add_argument("--out", default="artifacts")
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
    if args.ablation_suite:
        args.attacks = ["none", "lance", "lance_fixed", "lance_inject",
                        "lance_delete", "lance_random_target"]

    spec = GridSpec(attacks=args.attacks, defenses=args.defenses, seeds=args.seeds)
    _LOG.info(f"benchmark {cfg.data.name}: defenses={args.defenses} "
              f"attacks={args.attacks} seeds={args.seeds} epochs={cfg.train.epochs}")
    result = run_grid(cfg, spec)
    md = to_markdown(result)
    print("\n" + md + "\n")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"benchmark_{cfg.data.name}.md"), "w") as fh:
        fh.write(md + "\n")
    with open(os.path.join(args.out, f"benchmark_{cfg.data.name}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    _LOG.info(f"wrote artifacts/benchmark_{cfg.data.name}.{{md,json}}")


if __name__ == "__main__":
    main()
