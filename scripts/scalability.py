"""CLI: measure attack and training cost as the event stream grows.

For each subsampled stream size this records surrogate/victim training time per
epoch, node-impact computation time, the cost of building the meta-gradient
scorer, the LANCE and meta-gradient perturbation times, and peak GPU memory.
Together these cover the effectiveness-adjacent analyses (time complexity,
scalability, training/inference time) the project is expected to report.

Example:
    python scripts/scalability.py --config configs/mooc.yaml \
        --sizes 5000 10000 20000 40000 --epochs 3 --ptb-rate 0.3 --hist-neg 0.7
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import torch

try:
    from scripts._common import Config, build_model, load_data, perturb_train
except ModuleNotFoundError:
    from _common import Config, build_model, load_data, perturb_train

from lance.attack import compute_impact, run_attack
from lance.attack.meta import MetaGradientScorer
from lance.training import Trainer, resolve_device
from lance.utils import seed_everything, get_logger

_LOG = get_logger()


def _clock(device: str) -> float:
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def _score_fn(model, device):
    return lambda a, b, c: model.surrogate_scores(
        torch.as_tensor(a, device=device), torch.as_tensor(b, device=device),
        torch.as_tensor(c, dtype=torch.float32, device=device)).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[5000, 10000, 20000, 40000])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--ptb-rate", type=float, default=None)
    ap.add_argument("--hist-neg", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/scalability")
    args = ap.parse_args()

    base = Config.from_yaml(args.config)
    if args.ptb_rate is not None:
        base.attack.ptb_rate = args.ptb_rate
    if args.hist_neg is not None:
        base.eval.historical_neg_frac = args.hist_neg
    base.train.epochs = args.epochs
    device = resolve_device(base.train.device)

    # Warm up the device (context init and allocator) on the smallest size so the
    # first measured row is not charged one-time startup cost.
    seed_everything(args.seed, deterministic=True)
    warm_cfg = copy.deepcopy(base)
    warm_cfg.data.max_events = min(args.sizes)
    warm_data = load_data(warm_cfg)
    Trainer(build_model(warm_cfg, warm_data), warm_cfg, device=device).fit(
        warm_data, defense=None, verbose=False)

    rows = []
    for size in args.sizes:
        cfg = copy.deepcopy(base)
        cfg.data.max_events = size
        cfg.train.seed = args.seed
        seed_everything(args.seed, deterministic=True)
        data = load_data(cfg)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # surrogate training (reported per epoch)
        surrogate = build_model(cfg, data)
        t0 = _clock(device)
        Trainer(surrogate, cfg, device=device).fit(data, defense=None, verbose=False)
        surr_s = (_clock(device) - t0) / args.epochs
        surrogate.reset_state(device)
        for b in data.iter_batches("train", cfg.train.batch_size, device):
            if len(b):
                surrogate.advance_memory(b)

        s, d, t, f = data.split("train")
        score_fn = _score_fn(surrogate, device)

        t0 = _clock(device)
        impact = compute_impact(s, d, data.num_nodes, cfg.attack.impact_weights,
                                cfg.attack.betweenness_k)
        impact_s = _clock(device) - t0

        t0 = _clock(device)
        grad = MetaGradientScorer(surrogate, s, d, t, f, data.num_nodes, device,
                                  hist_frac=cfg.eval.historical_neg_frac or 0.7,
                                  seed=args.seed)
        meta_build_s = _clock(device) - t0

        t0 = _clock(device)
        res = run_attack("lance", s, d, t, f, data.num_nodes, impact=impact,
                         score_fn=score_fn, ptb_rate=cfg.attack.ptb_rate, seed=args.seed)
        lance_s = _clock(device) - t0

        t0 = _clock(device)
        run_attack("lance_meta", s, d, t, f, data.num_nodes, impact=impact,
                   score_fn=score_fn, ptb_rate=cfg.attack.ptb_rate, seed=args.seed,
                   grad_scorer=grad)
        meta_attack_s = _clock(device) - t0

        # victim training on the poisoned stream (reported per epoch)
        poisoned = perturb_train(data, res)
        victim = build_model(cfg, data)
        t0 = _clock(device)
        Trainer(victim, cfg, device=device).fit(poisoned, defense=None, verbose=False)
        victim_s = (_clock(device) - t0) / args.epochs

        peak_mb = (torch.cuda.max_memory_allocated() / 1e6
                   if device == "cuda" else float("nan"))

        rows.append({
            "events": data.num_train_events(), "nodes": data.num_nodes,
            "train_s_per_epoch": round(victim_s, 3),
            "surrogate_s_per_epoch": round(surr_s, 3),
            "impact_s": round(impact_s, 3),
            "meta_build_s": round(meta_build_s, 3),
            "lance_attack_s": round(lance_s, 3),
            "meta_attack_s": round(meta_attack_s, 3),
            "peak_mem_mb": round(peak_mb, 1),
        })
        _LOG.info(f"E={rows[-1]['events']:6d} nodes={data.num_nodes:5d} "
                  f"train/ep={victim_s:.2f}s impact={impact_s:.2f}s "
                  f"lance={lance_s:.2f}s meta={meta_attack_s:.2f}s peak={peak_mb:.0f}MB")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "scalability.json"), "w") as fh:
        json.dump({"config": base.to_dict(), "rows": rows}, fh, indent=2)
    header = ("| Events | Nodes | Train s/ep | Surrogate s/ep | Impact s | "
              "Meta build s | LANCE attack s | Meta attack s | Peak mem MB |")
    lines = [header, "|" + "---|" * 9]
    for r in rows:
        lines.append(f"| {r['events']} | {r['nodes']} | {r['train_s_per_epoch']} "
                     f"| {r['surrogate_s_per_epoch']} | {r['impact_s']} "
                     f"| {r['meta_build_s']} | {r['lance_attack_s']} "
                     f"| {r['meta_attack_s']} | {r['peak_mem_mb']} |")
    with open(os.path.join(args.out, "scalability.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
