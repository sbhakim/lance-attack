# lance/models/edgebank.py
"""EdgeBankLite: a non-learned memorization baseline for temporal link prediction.

EdgeBank predicts a link purely from whether the pair has been seen before
(Poursafaei et al., "Towards Better Evaluation for Dynamic Link Prediction"). It
has no parameters and no training: it scores a candidate high if that exact
``(src, dst)`` pair occurred in the observed stream, low otherwise.

It is here as a *reference victim*, not as a competitor. The attack results are
measured under historical negatives, which rank a true future edge against the
source's own past partners — a protocol that deliberately rewards memorization.
That invites an obvious objection: if the learned victims are mostly memorizing,
then "deleting real edges hurts them" is a statement about a lookup table rather
than about temporal graph learning, and the deletion finding is far less
interesting than it looks. Reporting EdgeBank under the identical protocol
settles that: it puts a number on how much of each victim's MRR memorization
alone can account for, and it shows how a pure memorizer responds to the same
poisoning.

Two standard variants are supported: unlimited history (``window=None``) and a
recency window holding only pairs seen within the last ``window`` fraction of the
observed time span.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from lance.data.dataset import EdgeBatch


class EdgeBankLite(nn.Module):
    """Memorization-only link predictor with the victim interface.

    The model is non-parametric, but the harness trains every victim with an
    optimizer and a backward pass. It therefore carries a single unused scalar
    that scores are offset by with a zero coefficient: gradients exist and flow
    nowhere, training is a no-op, and predictions are exactly EdgeBank's. This
    keeps the comparison honest -- identical trainer, identical evaluation, identical
    negative pools -- rather than special-casing the harness.
    """

    def __init__(self, num_nodes: int, num_feats: int, window: float | None = None,
                 pos_logit: float = 4.0, neg_logit: float = -4.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_feats = num_feats
        self.window = window
        self.pos_logit = pos_logit
        self.neg_logit = neg_logit
        self._unused = nn.Parameter(torch.zeros(1))   # see class docstring
        self._device = "cpu"
        self.last_seen: dict[tuple[int, int], float] = {}
        self._t_min: float | None = None
        self._t_max: float | None = None

    # -- state -----------------------------------------------------------------
    def reset_state(self, device: str | torch.device | None = None) -> None:
        if device is not None:
            self._device = device
        self.last_seen = {}
        self._t_min = self._t_max = None

    def detach_state(self) -> None:
        pass

    def staleness(self, nodes: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)

    def update_memory(self, batch: EdgeBatch) -> None:
        s = batch.src.detach().cpu().numpy()
        d = batch.dst.detach().cpu().numpy()
        ts = batch.t.detach().cpu().numpy()
        for u, v, tt in zip(s.tolist(), d.tolist(), ts.tolist()):
            self.last_seen[(int(u), int(v))] = float(tt)
        lo, hi = float(ts.min()), float(ts.max())
        self._t_min = lo if self._t_min is None else min(self._t_min, lo)
        self._t_max = hi if self._t_max is None else max(self._t_max, hi)

    @torch.no_grad()
    def advance_memory(self, batch: EdgeBatch) -> None:
        self.update_memory(batch)

    # -- scoring ---------------------------------------------------------------
    def _seen(self, src: np.ndarray, dst: np.ndarray, t: np.ndarray) -> np.ndarray:
        """True where the pair was observed, honouring the recency window."""
        cutoff = None
        if self.window is not None and self._t_min is not None:
            span = max(self._t_max - self._t_min, 1e-9)
            cutoff = self.window * span
        out = np.empty(len(src), dtype=bool)
        for i, (u, v, tt) in enumerate(zip(src.tolist(), dst.tolist(), t.tolist())):
            seen_at = self.last_seen.get((int(u), int(v)))
            out[i] = (seen_at is not None
                      and (cutoff is None or (float(tt) - seen_at) <= cutoff))
        return out

    def _logits(self, src: torch.Tensor, dst: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        seen = self._seen(src.detach().cpu().numpy(), dst.detach().cpu().numpy(),
                          t.detach().cpu().numpy())
        base = torch.where(
            torch.as_tensor(seen, device=self._device),
            torch.full((len(seen),), self.pos_logit, device=self._device),
            torch.full((len(seen),), self.neg_logit, device=self._device))
        # zero-coefficient offset: keeps the graph connected for the trainer's
        # backward pass without altering a single prediction
        return base + 0.0 * self._unused.to(base.device)

    def _embed(self, nodes: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # no embedding space exists; the C2 smoothness term must stay inert
        return torch.zeros(len(nodes), 1, device=self._device)

    def score_pairs(self, src, dst, t) -> torch.Tensor:
        return self._logits(src, dst, t)

    def score_pos_neg(self, batch: EdgeBatch, neg: torch.Tensor):
        pos = self._logits(batch.src, batch.dst, batch.t)
        b, m = neg.shape
        t_rep = batch.t.unsqueeze(1).expand(b, m).reshape(-1)
        src_rep = batch.src.unsqueeze(1).expand(b, m).reshape(-1)
        return pos, self._logits(src_rep, neg.reshape(-1), t_rep).view(b, m)

    @torch.no_grad()
    def surrogate_scores(self, src, dst, t) -> torch.Tensor:
        return torch.sigmoid(self.score_pairs(src, dst, t))
