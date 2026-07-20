"""C1: dual-tail, importance-conditioned edge screening.

At runtime, per training batch, an observed edge ``e=(u,v,t)`` receives an
injection-suspicion score ``sigma_inj = (1 - yhat) * Imp(e)``, which is large for
low-likelihood edges near influential nodes; suspect edges are down-weighted in
the loss. Offline, ``imputed_edges`` recovers high-likelihood pairs that are
absent near high-impact nodes, the complementary deletion signature, so the
trainer can re-introduce them as positives.
"""
from __future__ import annotations

import numpy as np
import torch


class Screening:
    def __init__(self, impact: np.ndarray, screen_q: float = 0.10,
                 impute_q: float = 0.02, min_weight: float = 0.5):
        self.impact_np = impact
        self.screen_q = screen_q
        self.impute_q = impute_q
        self.min_weight = min_weight
        self.impact: torch.Tensor | None = None     # device tensor, set on first use

    def _imp(self, device) -> torch.Tensor:
        if self.impact is None or self.impact.device != torch.device(device):
            self.impact = torch.as_tensor(self.impact_np, dtype=torch.float32, device=device)
        return self.impact

    @torch.no_grad()
    def injection_weights(self, model, batch) -> torch.Tensor:
        """Per-edge weight in [min_weight, 1]: low for suspected injections."""
        imp = self._imp(batch.src.device)
        yhat = model.surrogate_scores(batch.src, batch.dst, batch.t)        # [B]
        imp_e = torch.maximum(imp[batch.src], imp[batch.dst])               # [B]
        sigma = (1.0 - yhat) * imp_e                                        # injection suspicion
        if sigma.numel() == 0:
            return sigma
        # down-weight the top screen_q fraction of suspects
        thr = torch.quantile(sigma, 1.0 - self.screen_q)
        w = torch.ones_like(sigma)
        flagged = sigma >= thr
        w[flagged] = self.min_weight
        return w

    @torch.no_grad()
    def imputed_edges(self, model, data, max_candidates: int = 4000, device="cpu"):
        """Recover deletion-suspect edges: absent high-likelihood pairs near
        high-Impact nodes. Returns (src, dst, t) numpy arrays to re-insert."""
        src, dst, t, _ = data.split("train")
        observed = set(zip(src.tolist(), dst.tolist()))
        imp = self.impact_np
        thr = np.quantile(imp, 0.9)
        high = np.where(imp >= thr)[0]
        if len(high) == 0:
            return (np.array([], np.int64),) * 3
        rng = np.random.default_rng(0)
        cu = rng.choice(high, size=max_candidates)
        cv = rng.choice(np.unique(dst), size=max_candidates)
        ct = rng.choice(t, size=max_candidates)
        mask = np.array([(u, v) not in observed for u, v in zip(cu.tolist(), cv.tolist())])
        cu, cv, ct = cu[mask], cv[mask], ct[mask]
        if len(cu) == 0:
            return (np.array([], np.int64),) * 3
        ys = model.surrogate_scores(
            torch.as_tensor(cu, device=device), torch.as_tensor(cv, device=device),
            torch.as_tensor(ct, dtype=torch.float32, device=device)).cpu().numpy()
        n_imp = int(self.impute_q * len(src))
        sel = np.argsort(-ys)[:n_imp]                       # highest-likelihood absent pairs
        return cu[sel].astype(np.int64), cv[sel].astype(np.int64), ct[sel].astype(np.float64)
