"""CLI: train a surrogate, run HIA, and report the attacked-vs-clean gap.

Example:
    python scripts/run_attack.py --config configs/mooc.yaml --ptb-rate 0.1
"""
from __future__ import annotations

import argparse

import torch

try:
    from scripts._common import Config, build_model, load_data, perturb_train
except ModuleNotFoundError:
    from _common import Config, build_model, load_data, perturb_train

from lance.attack import compute_impact, hia_attack
from lance.training import Trainer, resolve_device
from lance.utils import seed_everything, get_logger

_LOG = get_logger()


def make_score_fn(model, device):
    def score_fn(s, d, t):
        return model.surrogate_scores(
            torch.as_tensor(s, device=device), torch.as_tensor(d, device=device),
            torch.as_tensor(t, dtype=torch.float32, device=device)).cpu().numpy()
    return score_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ptb-rate", type=float, default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.ptb_rate is not None:
        cfg.attack.ptb_rate = args.ptb_rate
    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    data = load_data(cfg)

    # Surrogate: a victim trained on clean data (the attacker's black-box proxy).
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
                     ptb_rate=cfg.attack.ptb_rate, del_percentile=cfg.attack.del_percentile,
                     inj_percentile=cfg.attack.inj_percentile, seed=cfg.train.seed)
    _LOG.info(f"HIA: deleted={res.n_deleted} injected={res.n_injected} "
              f"(budget {int(cfg.attack.ptb_rate*len(s))})")

    poisoned = perturb_train(data, res)
    victim = build_model(cfg, data)
    Trainer(victim, cfg, device=device).fit(poisoned, defense=None, verbose=False)
    attacked = Trainer(victim, cfg, device=device).test(poisoned)
    _LOG.info(f"ATTACKED TEST: MRR={attacked['mrr']:.4f}")


if __name__ == "__main__":
    main()
