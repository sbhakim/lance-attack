"""C3 -- importance-guided adversarial training.

Every ``adv_every`` epochs we run the inner HIA attacker against the *current*
victim (used as its own surrogate) to synthesize importance-targeted injected
edges, which are fed to the trainer as hard negatives. This hardens the victim
against exactly the multi-step, targeted edits HIA produces, rather than random
perturbations.
"""
from __future__ import annotations

import numpy as np
import torch

from lance.attack.hia import hia_attack
from lance.data.dataset import EdgeBatch


class AdvTrainer:
    def __init__(self, impact: np.ndarray, adv_every: int = 0,
                 adv_ptb_rate: float = 0.05, seed: int = 0):
        self.impact = impact
        self.adv_every = adv_every
        self.adv_ptb_rate = adv_ptb_rate
        self.seed = seed

    def enabled(self) -> bool:
        return self.adv_every > 0

    def generate(self, epoch: int, data, model, device: str = "cpu") -> EdgeBatch | None:
        if not self.enabled() or (epoch % self.adv_every != 0):
            return None
        src, dst, t, feat = data.split("train")

        def score_fn(s, d, tt):
            return model.surrogate_scores(
                torch.as_tensor(s, device=device),
                torch.as_tensor(d, device=device),
                torch.as_tensor(tt, dtype=torch.float32, device=device)).cpu().numpy()

        # warm the surrogate memory so its scores are meaningful
        model.reset_state(device)
        for batch in data.iter_batches("train", 200, device):
            if len(batch):
                model.advance_memory(batch)

        res = hia_attack(src, dst, t, feat, data.num_nodes, self.impact, score_fn,
                         ptb_rate=self.adv_ptb_rate, seed=self.seed + epoch)
        if res.n_injected == 0:
            return None
        return EdgeBatch(
            torch.as_tensor(res.injected_src, dtype=torch.long),
            torch.as_tensor(res.injected_dst, dtype=torch.long),
            torch.as_tensor(res.injected_t, dtype=torch.float32),
            torch.zeros(res.n_injected, data.num_feats, dtype=torch.float32),
        )
