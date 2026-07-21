"""TGATLite: a memory-free attention-based temporal link predictor.

A third victim family for the transfer study, distinct from both the memory-based
``TGNLite`` surrogate and the MLP-mixer ``GraphMixerLite``. A node is represented by
attending over its most recent neighbors: each recent neighbor contributes a
learnable node embedding plus a time encoding of the gap to the query time, and a
single-head attention (query = the target node's own embedding) pools them. This
uses neighbor *identity* and attention, where GraphMixer uses link *features* and an
MLP-mixer -- so agreement across the two is evidence an attack transfers to the
attention family, not to one specific model.

State is a fixed-size ring buffer of recent (neighbor, time) per node; it carries
inputs, not gradient. Predict-then-update is preserved by scoring a batch before
appending it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lance.models.memory import TimeEncoder
from lance.models.link_predictor import LinkPredictor
from lance.data.dataset import EdgeBatch


class TGATLite(nn.Module):
    def __init__(self, num_nodes: int, num_feats: int, k_neighbors: int = 10,
                 time_dim: int = 64, node_dim: int = 64, hidden: int = 64,
                 embedding_dim: int = 64, predictor_hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_feats = num_feats
        self.K = k_neighbors
        self.hidden = hidden

        self.node_embed = nn.Embedding(num_nodes, node_dim)
        self.time_encoder = TimeEncoder(time_dim)
        kv_dim = node_dim + time_dim
        self.q_proj = nn.Linear(node_dim, hidden)
        self.k_proj = nn.Linear(kv_dim, hidden)
        self.v_proj = nn.Linear(kv_dim, hidden)
        self.out = nn.Sequential(nn.Linear(hidden + node_dim, embedding_dim),
                                 nn.ReLU(), nn.Dropout(dropout))
        self.predictor = LinkPredictor(embedding_dim, predictor_hidden, dropout)

        self._device = "cpu"
        self.buf_nbr = self.buf_t = self.buf_valid = self.buf_pos = None

    # -- state (per-node ring buffer of recent neighbors; numpy) ---------------
    def reset_state(self, device: str | torch.device | None = None) -> None:
        if device is not None:
            self._device = device
        n, k = self.num_nodes, self.K
        self.buf_nbr = np.zeros((n, k), dtype=np.int64)
        self.buf_t = np.zeros((n, k), dtype=np.float64)
        self.buf_valid = np.zeros((n, k), dtype=bool)
        self.buf_pos = np.zeros(n, dtype=np.int64)

    def detach_state(self) -> None:
        pass

    def _append_batch(self, src, dst, t) -> None:
        for u, v, ts in zip(src, dst, t):
            iu, iv = int(u), int(v)
            for node, nbr in ((iu, iv), (iv, iu)):
                p = self.buf_pos[node]
                self.buf_nbr[node, p] = nbr
                self.buf_t[node, p] = ts
                self.buf_valid[node, p] = True
                self.buf_pos[node] = (p + 1) % self.K

    def update_memory(self, batch: EdgeBatch) -> None:
        self._append_batch(batch.src.cpu().numpy(), batch.dst.cpu().numpy(),
                           batch.t.cpu().numpy())

    @torch.no_grad()
    def advance_memory(self, batch: EdgeBatch) -> None:
        self.update_memory(batch)

    def staleness(self, nodes: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)

    # -- representation & scoring ----------------------------------------------
    def _node_repr(self, nodes: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        dev = self._device
        idx = nodes.detach().cpu().numpy()
        nbr = torch.as_tensor(self.buf_nbr[idx], device=dev)                 # [B,K]
        times = torch.as_tensor(self.buf_t[idx], device=dev, dtype=torch.float32)
        valid = torch.as_tensor(self.buf_valid[idx], device=dev)            # [B,K] bool

        nbr_emb = self.node_embed(nbr)                                      # [B,K,node]
        dt = (t.unsqueeze(1) - times).clamp(min=0.0)                        # [B,K]
        kv = torch.cat([nbr_emb, self.time_encoder(dt)], dim=-1)           # [B,K,kv]

        q = self.q_proj(self.node_embed(nodes))                            # [B,H]
        k = self.k_proj(kv)                                                # [B,K,H]
        v = self.v_proj(kv)                                                # [B,K,H]
        scores = (q.unsqueeze(1) * k).sum(-1) / (self.hidden ** 0.5)        # [B,K]
        scores = scores.masked_fill(~valid, -1e9)
        attn = F.softmax(scores, dim=1)                                    # [B,K]
        context = (attn.unsqueeze(-1) * v).sum(dim=1)                       # [B,H]
        context = context * valid.any(dim=1, keepdim=True)                 # zero cold nodes
        return self.out(torch.cat([context, self.node_embed(nodes)], dim=-1))

    def _embed(self, nodes: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._node_repr(nodes, t)

    def score_pairs(self, src, dst, t) -> torch.Tensor:
        return self.predictor(self._node_repr(src, t), self._node_repr(dst, t))

    def score_pos_neg(self, batch: EdgeBatch, neg: torch.Tensor):
        z_src = self._node_repr(batch.src, batch.t)
        z_dst = self._node_repr(batch.dst, batch.t)
        pos = self.predictor(z_src, z_dst)
        b, m = neg.shape
        t_rep = batch.t.unsqueeze(1).expand(b, m).reshape(-1)
        z_neg = self._node_repr(neg.reshape(-1), t_rep)
        z_src_rep = z_src.unsqueeze(1).expand(b, m, z_src.shape[-1]).reshape(-1, z_src.shape[-1])
        return pos, self.predictor(z_src_rep, z_neg).view(b, m)

    @torch.no_grad()
    def surrogate_scores(self, src, dst, t) -> torch.Tensor:
        return torch.sigmoid(self.score_pairs(src, dst, t))
