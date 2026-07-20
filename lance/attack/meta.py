"""First-order gradient scoring of candidate poisoning edits.

The adaptive core in :mod:`lance.attack.lance` ranks edits by a fixed combination
of surrogate likelihood and node importance. That priority is not tied to the
quantity the victim is evaluated on, its ranking loss under historical negatives,
which is part of why it does not improve on random perturbation.

MetaGradientScorer scores an edit by a first-order estimate of its effect on that
loss, adapting the surrogate-gradient approach used for static-graph poisoning
(Nettack) to a memory-based model. Let M denote the surrogate memory after the
observed stream. We evaluate the victim ranking loss on a prefix-only query set
and take its gradient G = dL/dM. An edit is then scored by how far the memory
update it would induce moves along G: injecting an edge is damaging when its
update aligns with G, and deleting an observed event is damaging when that event's
update opposes G. The two scores share a scale, so deletions and injections are
ranked in a single queue and edits with non-positive estimated effect are dropped.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _build_history(src: np.ndarray, dst: np.ndarray, num_nodes: int) -> dict[int, np.ndarray]:
    """Map each source to the destinations it has linked to (observed stream)."""
    from collections import defaultdict
    hist: dict[int, list] = defaultdict(list)
    for u, v in zip(src.tolist(), dst.tolist()):
        hist[u].append(v)
    return {k: np.asarray(v, dtype=np.int64) for k, v in hist.items()}


class MetaGradientScorer:
    """Scores edits by a first-order estimate of their effect on the ranking loss.

    Args:
        model: a :class:`~lance.models.TGNLite` already trained and advanced
            (``advance_memory``) through the observed stream, so ``model.memory``
            holds the final node states.
        src, dst, t, feat: the observed training stream (numpy arrays).
        num_nodes: node-id space size.
        device: torch device the model lives on.
        hist_frac: fraction of query negatives drawn from a source's history
            (matches the evaluation's historical-negative fraction).
        n_queries: number of observed edges sampled to form the damage loss.
        n_neg: negatives per query.
        seed: RNG seed for query/negative sampling (deterministic given inputs).
    """

    def __init__(self, model, src, dst, t, feat, num_nodes, device,
                 hist_frac: float = 0.7, n_queries: int = 1024, n_neg: int = 20,
                 seed: int = 0):
        self.model = model
        self.device = device
        self.num_feats = int(feat.shape[1])
        self._rng = np.random.default_rng(seed)
        self.G = self._damage_gradient(src, dst, t, num_nodes, hist_frac,
                                       n_queries, n_neg)
        # Detached snapshots of the state used to evaluate candidate edits.
        self.mem = model.memory.detach()
        self.last_update = model.last_update.detach()

    # ------------------------------------------------------------------
    def _damage_gradient(self, src, dst, t, num_nodes, hist_frac, n_queries,
                         n_neg) -> torch.Tensor:
        model = self.model
        was_training = model.training
        model.eval()

        n = len(src)
        if n == 0:                       # degenerate stream: no damage signal
            return torch.zeros_like(model.memory)
        q = min(int(n_queries), n)
        qi = self._rng.choice(n, size=q, replace=False)
        qs = src[qi].astype(np.int64)
        qd = dst[qi].astype(np.int64)
        qt = t[qi].astype(np.float32)

        history = _build_history(src, dst, num_nodes)
        dst_pool = np.unique(dst)
        n_hist = int(n_neg * hist_frac)
        neg = np.empty((q, n_neg), dtype=np.int64)
        for i in range(q):
            s0, pos = int(qs[i]), int(qd[i])
            random_pool = dst_pool[dst_pool != pos]
            if len(random_pool) == 0:
                random_pool = dst_pool
            h = history.get(s0)
            hist_pool = h[h != pos] if h is not None else None
            if hist_pool is not None and len(hist_pool) and n_hist:
                hist_neg = self._rng.choice(hist_pool, size=n_hist)
            else:
                hist_neg = self._rng.choice(random_pool, size=n_hist)
            rand_neg = self._rng.choice(random_pool, size=n_neg - n_hist)
            neg[i] = np.concatenate([hist_neg, rand_neg])

        # Swap in a differentiable copy of the warmed memory as a leaf.
        orig_mem = model.memory
        M = orig_mem.detach().clone().requires_grad_(True)
        model.memory = M
        try:
            qs_t = torch.as_tensor(qs, device=self.device)
            qd_t = torch.as_tensor(qd, device=self.device)
            qt_t = torch.as_tensor(qt, dtype=torch.float32, device=self.device)
            neg_t = torch.as_tensor(neg.reshape(-1), device=self.device)
            qt_rep = qt_t.unsqueeze(1).expand(q, n_neg).reshape(-1)

            pos_score = model.score_pairs(qs_t, qd_t, qt_t)              # [q]
            src_rep = qs_t.unsqueeze(1).expand(q, n_neg).reshape(-1)
            neg_score = model.score_pairs(src_rep, neg_t, qt_rep).view(q, n_neg)

            # Softplus ranking margin. The victim minimizes this, so its gradient
            # w.r.t. memory points toward higher victim error.
            loss = F.softplus(neg_score - pos_score.unsqueeze(1)).mean()
            # Differentiate w.r.t. memory only, leaving the model's parameter
            # gradients untouched (the surrogate is not optimized after this).
            (G,) = torch.autograd.grad(loss, M)
            G = torch.nan_to_num(G.detach()).clone()
        finally:
            model.memory = orig_mem
            if was_training:
                model.train()
        return G

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _alignment(self, u, v, t, feat) -> np.ndarray:
        """Inner product of the damage gradient with the memory update the event
        would induce, summed over both endpoints."""
        if len(u) == 0:
            return np.array([], dtype=float)
        u_t = torch.as_tensor(np.asarray(u), dtype=torch.long, device=self.device)
        v_t = torch.as_tensor(np.asarray(v), dtype=torch.long, device=self.device)
        t_t = torch.as_tensor(np.asarray(t), dtype=torch.float32, device=self.device)
        f_t = torch.as_tensor(np.asarray(feat), dtype=torch.float32, device=self.device)
        if f_t.dim() == 1:
            f_t = f_t.unsqueeze(-1)

        mem_u, mem_v = self.mem[u_t], self.mem[v_t]
        te_u = self.model.time_encoder((t_t - self.last_update[u_t]).clamp(min=0.0))
        te_v = self.model.time_encoder((t_t - self.last_update[v_t]).clamp(min=0.0))
        msg_u = torch.cat([mem_u, mem_v, te_u, f_t], dim=-1)
        msg_v = torch.cat([mem_v, mem_u, te_v, f_t], dim=-1)
        new_u = self.model.gru(msg_u, mem_u)
        new_v = self.model.gru(msg_v, mem_v)
        align = ((self.G[u_t] * (new_u - mem_u)).sum(-1)
                 + (self.G[v_t] * (new_v - mem_v)).sum(-1))
        return align.detach().cpu().numpy().astype(float)

    def injection_damage(self, cu, cv, ct, cf) -> np.ndarray:
        """Estimated victim-loss increase from injecting each candidate edge."""
        return self._alignment(cu, cv, ct, cf)

    def deletion_damage(self, su, sv, st, sf) -> np.ndarray:
        """Estimated victim-loss increase from deleting each observed edge."""
        return -self._alignment(su, sv, st, sf)
