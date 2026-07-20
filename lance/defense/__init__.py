"""DT-SHIELD defense: orchestrates C1 (screening), C2 (consistency), C3 (AT).

``DTShieldDefense`` exposes the duck-typed interface the :class:`Trainer`
expects (``precompute`` / ``weight_batch`` / ``adv_negatives`` /
``extra_positives`` / ``smooth_lambda``). The per-batch loss weight is the
*product* of the C1 (injection) and C2 (consistency) weights, so an edge must
look benign under *both* views to keep full weight.
"""
from __future__ import annotations

import numpy as np
import torch

from lance.attack.importance import compute_impact
from lance.data.dataset import EdgeBatch
from lance.defense.screening import Screening
from lance.defense.consistency import Consistency
from lance.defense.adv_training import AdvTrainer
from lance.defense.baselines import TShieldDefense, CosineDefense

__all__ = ["DTShieldDefense", "TShieldDefense", "CosineDefense", "build_defense"]


class DTShieldDefense:
    def __init__(self, cfg, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.smooth_lambda = cfg.defense.smooth_lambda
        self._impact: np.ndarray | None = None
        self.c1: Screening | None = None
        self.c2: Consistency | None = None
        self.c3: AdvTrainer | None = None
        self._imputed: EdgeBatch | None = None

    # -- one-time setup --------------------------------------------------------
    def precompute(self, data, model) -> None:
        d = self.cfg.defense
        src, dst, _, _ = data.split("train")
        self._impact = compute_impact(src, dst, data.num_nodes,
                                       weights=self.cfg.attack.impact_weights,
                                       betweenness_k=self.cfg.attack.betweenness_k)
        self.c1 = Screening(self._impact, d.screen_q, d.impute_q)
        self.c2 = Consistency(d.band_low, d.band_high, d.clean_prefix_frac).fit(data)
        self.c3 = AdvTrainer(self._impact, d.adv_every, d.adv_ptb_rate, self.cfg.train.seed)

        # C1 imputation needs a warmed surrogate; reuse the (untrained-at-first)
        # model -- imputations are refreshed cheaply and only add positives.
        model.reset_state(self.device)
        for b in data.iter_batches("train", self.cfg.train.batch_size, self.device):
            if len(b):
                model.advance_memory(b)
        su, dv, tt = self.c1.imputed_edges(model, data, device=self.device)
        if len(su):
            self._imputed = EdgeBatch(
                torch.as_tensor(su, dtype=torch.long),
                torch.as_tensor(dv, dtype=torch.long),
                torch.as_tensor(tt, dtype=torch.float32),
                torch.zeros(len(su), data.num_feats, dtype=torch.float32),
            )

    # -- per-batch hooks -------------------------------------------------------
    @torch.no_grad()
    def weight_batch(self, model, batch) -> torch.Tensor:
        # Average (not product) of the C1 and C2 weights: an edge is only
        # strongly down-weighted when *both* views agree, which keeps the
        # defense near-harmless on clean data (high clean retention).
        w1 = self.c1.injection_weights(model, batch)     # C1
        w2 = self.c2.weights(model, batch)               # C2
        return 0.5 * (w1 + w2)

    def adv_negatives(self, epoch, data, model):
        return self.c3.generate(epoch, data, model, self.device) if self.c3 else None

    def extra_positives(self, epoch, data, model):
        """C1-imputed deletion-suspect edges, re-introduced as positives."""
        return self._imputed.to(self.device) if self._imputed is not None else None


def build_defense(cfg, device: str = "cpu"):
    """Construct a defense from ``cfg.defense.mode``.

    Returns ``None`` for the undefended baseline; otherwise one of the SOTA
    baselines or DT-SHIELD. All share the trainer's duck-typed interface.
    """
    mode = cfg.defense.mode
    if mode in ("none", None):
        return None
    if mode == "tshield":
        return TShieldDefense(cfg)
    if mode == "cosine":
        return CosineDefense(cfg)
    if mode == "dtshield":
        return DTShieldDefense(cfg, device=device)
    raise ValueError(f"unknown defense mode: {mode}")
