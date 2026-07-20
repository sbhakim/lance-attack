"""SOTA / reference defenses, implemented in the same training pipeline so the
comparison against DT-SHIELD is apples-to-apples.

  * :class:`TShieldDefense`  -- the AAAI'24 T-SHIELD: a *single-tail*,
    cosine-annealed low-likelihood edge filter plus an embedding-smoothness term.
    Deliberately importance-agnostic and deletion-blind (that is the point of the
    comparison).
  * :class:`CosineDefense`   -- a GNNGuard-style baseline: down-weight edges whose
    endpoint embeddings have low cosine affinity (homophily prior).

Both expose the same duck-typed interface as :class:`DTShieldDefense`
(``precompute`` / ``on_epoch_start`` / ``weight_batch`` / ``adv_negatives`` /
``extra_positives`` / ``smooth_lambda``), so the :class:`Trainer` is unchanged.
"""
from __future__ import annotations

import math

import torch


class _BaseDefense:
    """Common no-op hooks; subclasses override ``weight_batch``."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.smooth_lambda = cfg.defense.smooth_lambda

    def precompute(self, data, model) -> None:        # no offline state needed
        pass

    def on_epoch_start(self, epoch: int, total: int) -> None:
        pass

    def adv_negatives(self, epoch, data, model):
        return None

    def extra_positives(self, epoch, data, model):
        return None


class TShieldDefense(_BaseDefense):
    """T-SHIELD: single-tail edge filter with a cosine-annealed threshold."""

    def __init__(self, cfg, min_weight: float = 0.05):
        super().__init__(cfg)
        self.pct_s = cfg.defense.tshield_pct_s
        self.pct_e = cfg.defense.tshield_pct_e
        self.min_weight = min_weight
        self.cur_pct = self.pct_s

    def on_epoch_start(self, epoch: int, total: int) -> None:
        # cosine annealing of the drop-percentile, as in the paper
        frac = 0.5 * (1.0 - math.cos(math.pi * (epoch - 1) / max(total - 1, 1)))
        self.cur_pct = self.pct_s + frac * (self.pct_e - self.pct_s)

    @torch.no_grad()
    def weight_batch(self, model, batch) -> torch.Tensor:
        yhat = model.surrogate_scores(batch.src, batch.dst, batch.t)
        if yhat.numel() == 0:
            return yhat
        thr = torch.quantile(yhat, self.cur_pct / 100.0)
        w = torch.ones_like(yhat)
        w[yhat < thr] = self.min_weight            # drop only the LOW tail
        return w


class CosineDefense(_BaseDefense):
    """GNNGuard-style: down-weight low embedding-affinity (likely-fake) edges."""

    def __init__(self, cfg, min_weight: float = 0.05):
        super().__init__(cfg)
        self.q = cfg.defense.cosine_q
        self.min_weight = min_weight

    @torch.no_grad()
    def weight_batch(self, model, batch) -> torch.Tensor:
        z_s = model._embed(batch.src, batch.t)
        z_d = model._embed(batch.dst, batch.t)
        cos = torch.nn.functional.cosine_similarity(z_s, z_d, dim=-1)
        if cos.numel() == 0:
            return cos
        thr = torch.quantile(cos, self.q)          # bottom-q affinity = suspicious
        w = torch.ones_like(cos)
        w[cos <= thr] = self.min_weight
        return w
