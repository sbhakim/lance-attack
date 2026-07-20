"""TGNLite: a compact memory-based temporal GNN for link prediction.

The design follows the TGN family with minimal dependencies. Each node holds a
recurrent memory state updated by a ``GRUCell``; memory is advanced through the
event stream and detached every ``bptt`` steps (truncated backpropagation through
time). Predictions are always made before an event updates memory, so no edge
informs its own prediction. Node embeddings combine memory with a staleness time
encoding ``t - last_update[i]``, and a :class:`LinkPredictor` MLP scores pairs.
The ``staleness`` and ``surrogate_scores`` methods expose signals used by the
attack and defense.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lance.models.memory import TimeEncoder
from lance.models.link_predictor import LinkPredictor
from lance.data.dataset import EdgeBatch


class TGNLite(nn.Module):
    def __init__(self, num_nodes: int, num_feats: int, memory_dim: int = 100,
                 time_dim: int = 100, embedding_dim: int = 100,
                 predictor_hidden: int = 80, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_feats = num_feats
        self.memory_dim = memory_dim

        self.time_encoder = TimeEncoder(time_dim)
        msg_dim = 2 * memory_dim + time_dim + num_feats
        self.gru = nn.GRUCell(msg_dim, memory_dim)
        self.embed = nn.Sequential(
            nn.Linear(memory_dim + time_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.predictor = LinkPredictor(embedding_dim, predictor_hidden, dropout)

        # Recurrent state (not parameters). Allocated by ``reset_state``.
        self.memory: torch.Tensor | None = None
        self.last_update: torch.Tensor | None = None
        self._device = "cpu"

    # -- state management ------------------------------------------------------
    def reset_state(self, device: str | torch.device | None = None) -> None:
        if device is not None:
            self._device = device
        self.memory = torch.zeros(self.num_nodes, self.memory_dim, device=self._device)
        self.last_update = torch.zeros(self.num_nodes, device=self._device)

    def detach_state(self) -> None:
        if self.memory is not None:
            self.memory = self.memory.detach()

    def staleness(self, nodes: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return ``t - last_update[nodes]`` (>=0); large + unchanged => frozen."""
        return (t - self.last_update[nodes]).clamp(min=0.0)

    # -- embedding & scoring ---------------------------------------------------
    def _embed(self, nodes: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mem = self.memory[nodes]
        dt = (t - self.last_update[nodes]).clamp(min=0.0)
        te = self.time_encoder(dt)
        return self.embed(torch.cat([mem, te], dim=-1))

    def score_pairs(self, src: torch.Tensor, dst: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
        return self.predictor(self._embed(src, t), self._embed(dst, t))

    def score_pos_neg(self, batch: EdgeBatch, neg: torch.Tensor):
        """Return (pos_scores [B], neg_scores [B, M]) for ranking/eval."""
        z_src = self._embed(batch.src, batch.t)         # [B, E]
        z_dst = self._embed(batch.dst, batch.t)         # [B, E]
        pos = self.predictor(z_src, z_dst)              # [B]
        b, m = neg.shape
        t_rep = batch.t.unsqueeze(1).expand(b, m).reshape(-1)
        z_neg = self._embed(neg.reshape(-1), t_rep)     # [B*M, E]
        z_src_rep = z_src.unsqueeze(1).expand(b, m, z_src.shape[-1]).reshape(-1, z_src.shape[-1])
        neg_scores = self.predictor(z_src_rep, z_neg).view(b, m)
        return pos, neg_scores

    # -- memory update (predict-then-update is enforced by the caller) ---------
    def update_memory(self, batch: EdgeBatch) -> None:
        src, dst, t, feat = batch.src, batch.dst, batch.t, batch.feat
        mem_s, mem_d = self.memory[src], self.memory[dst]
        te_s = self.time_encoder((t - self.last_update[src]).clamp(min=0.0))
        te_d = self.time_encoder((t - self.last_update[dst]).clamp(min=0.0))
        msg_s = torch.cat([mem_s, mem_d, te_s, feat], dim=-1)
        msg_d = torch.cat([mem_d, mem_s, te_d, feat], dim=-1)
        new_s = self.gru(msg_s, mem_s)
        new_d = self.gru(msg_d, mem_d)
        mem = self.memory.index_copy(0, src, new_s)
        mem = mem.index_copy(0, dst, new_d)
        self.memory = mem
        with torch.no_grad():
            self.last_update[src] = t
            self.last_update[dst] = t

    @torch.no_grad()
    def advance_memory(self, batch: EdgeBatch) -> None:
        """Update memory without tracking gradients (used during evaluation)."""
        self.update_memory(batch)
        self.detach_state()

    @torch.no_grad()
    def surrogate_scores(self, src: torch.Tensor, dst: torch.Tensor,
                         t: torch.Tensor) -> torch.Tensor:
        """Sigmoid link likelihoods from the current state (used by C1 / HIA)."""
        return torch.sigmoid(self.score_pairs(src, dst, t))
