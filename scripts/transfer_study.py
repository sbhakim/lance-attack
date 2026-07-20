"""CLI: two-regime comparison for a fixed attack set.

Runs the same attacks (built against the TGNLite surrogate) against two victims
under an identical paired protocol: a same-architecture, white-box victim
(TGNLite) and a different-architecture transfer victim (GraphMixerLite). The
combined table makes the transfer question explicit -- whether the white-box
ordering of attacks survives a change of victim architecture.

Example:
    python scripts/transfer_study.py --config configs/bitcoinotc.yaml \
        --seeds 0 1 2 3 4 --epochs 20 --ptb-rate 0.3 --hist-neg 0.7
"""
from __future__ import annotations

import argparse
import copy
import json
import os

try:
    from scripts._common import Config
except ModuleNotFoundError:
    from _common import Config

from lance.experiment import GridSpec, run_grid
from lance.models import TGNLite, GraphMixerLite
from lance.utils import get_logger

_LOG = get_logger()


def _by_attack(result: dict) -> dict:
    return {r["attack"]: r["degradation"] for r in result["rows"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--attacks", nargs="+",
                    default=["none", "random", "random_delete", "random_inject",
                             "hia", "lance", "lance_meta"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--ptb-rate", type=float, default=None)
    ap.add_argument("--hist-neg", type=float, default=None)
    ap.add_argument("--out", default="artifacts/transfer_study")
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

    spec = GridSpec(attacks=args.attacks, defenses=["none"], seeds=args.seeds)
    _LOG.info(f"white-box regime (victim=TGNLite) on {cfg.data.name}")
    wb = run_grid(copy.deepcopy(cfg), spec, victim_cls=TGNLite)
    _LOG.info(f"transfer regime (victim=GraphMixerLite) on {cfg.data.name}")
    tr = run_grid(copy.deepcopy(cfg), spec, victim_cls=GraphMixerLite)

    wbd, trd = _by_attack(wb), _by_attack(tr)
    rows = []
    for atk in args.attacks:
        if atk == "none":
            continue
        w, t = wbd.get(atk, {}), trd.get(atk, {})
        rows.append({"attack": atk,
                     "wb_dmrr": w.get("mean"), "wb_p": w.get("paired_t_p"),
                     "tr_dmrr": t.get("mean"), "tr_p": t.get("paired_t_p")})

    out = {"dataset": wb["dataset"], "seeds": args.seeds,
           "clean_mrr_whitebox": wb["clean_mrr"],
           "clean_mrr_transfer": tr["clean_mrr"], "rows": rows}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"transfer_{cfg.data.name}.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    md = [f"## Two-regime transfer -- `{wb['dataset']}`",
          f"White-box clean MRR {wb['clean_mrr']:.4f} | transfer clean MRR "
          f"{tr['clean_mrr']:.4f} | seeds={args.seeds}", "",
          "| Attack | White-box dMRR | p | Transfer dMRR | p |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['attack']} | {r['wb_dmrr']:+.4f} | {r['wb_p']:.3g} "
                  f"| {r['tr_dmrr']:+.4f} | {r['tr_p']:.3g} |")
    text = "\n".join(md)
    with open(os.path.join(args.out, f"transfer_{cfg.data.name}.md"), "w") as fh:
        fh.write(text + "\n")
    print("\n" + text + "\n")


if __name__ == "__main__":
    main()
