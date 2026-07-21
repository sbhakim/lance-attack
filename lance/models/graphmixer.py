"""GraphMixerLite: a memory-free temporal link predictor.

This is a deliberately different victim family from :class:`TGNLite`. Instead of a
recurrent per-node memory, each node is represented from its most recent
interactions: the last ``k_neighbors`` events (their link features and a time
encoding of the gap to the query time) are combined by an MLP-mixer, pooled, and
concatenated with a learnable node embedding before an MLP decoder scores a pair.

The point is transfer: the attacker builds its perturbation against a memory-based
surrogate, and we test whether it degrades a victim with a different inductive
bias. State here is a fixed-size ring buffer of recent events per node; it carries
inputs, not gradient, so training flows only through the mixer, embeddings, and
decoder. Predict-then-update is preserved by scoring a batch before appending it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from lance.models.memory import TimeEncoder
from lance.models.link_predictor import LinkPredictor
from lance.data.dataset import EdgeBatch


class GraphMixerLite(nn.Module):
    def __init__(self, num_nodes: int, num_feats: int, k_neighbors: int = 10,
                 time_dim: int = 64, hidden: int = 64, id_dim: int = 64,
                 embedding_dim: int = 64, predictor_hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_feats = num_feats
        self.K = k_neighbors

        self.time_encoder = TimeEncoder(time_dim)
        token_dim = num_feats + time_dim
        self.token_proj = nn.Linear(token_dim, hidden)
        self.token_mix = nn.Sequential(nn.Linear(k_neighbors, k_neighbors),
                                       nn.GELU(), nn.Dropout(dropout))
        self.channel_norm = nn.LayerNorm(hidden)
        self.channel_mix = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                         nn.Dropout(dropout))
        self.id_embed = nn.Embedding(num_nodes, id_dim)
        self.out = nn.Sequential(nn.Linear(hidden + id_dim, embedding_dim), nn.ReLU(),
                                 nn.Dropout(dropout))
        self.predictor = LinkPredictor(embedding_dim, predictor_hidden, dropout)

        self._device = "cpu"
        self.buf_t = self.buf_feat = self.buf_valid = self.buf_pos = None

    # -- state (per-node ring buffer of recent events; numpy, non-differentiable) --
    def reset_state(self, device: str | torch.device | None = None) -> None:
        if device is not None:
            self._device = device
        n, k = self.num_nodes, self.K
        self.buf_t = np.zeros((n, k), dtype=np.float64)
        self.buf_feat = np.zeros((n, k, self.num_feats), dtype=np.float32)
        self.buf_valid = np.zeros((n, k), dtype=bool)
        self.buf_pos = np.zeros(n, dtype=np.int64)

    def detach_state(self) -> None:  # no autograd flows through the buffer
        pass

    def _append_batch(self, src, dst, t, feat) -> None:
        for u, v, ts, fe in zip(src, dst, t, feat):
            for node in (int(u), int(v)):
                p = self.buf_pos[node]
                self.buf_t[node, p] = ts
                self.buf_feat[node, p] = fe
                self.buf_valid[node, p] = True
                self.buf_pos[node] = (p + 1) % self.K

    def update_memory(self, batch: EdgeBatch) -> None:
        self._append_batch(batch.src.cpu().numpy(), batch.dst.cpu().numpy(),
                           batch.t.cpu().numpy(), batch.feat.cpu().numpy())

    @torch.no_grad()
    def advance_memory(self, batch: EdgeBatch) -> None:
        self.update_memory(batch)

    def staleness(self, nodes: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)

    # -- representation & scoring ----------------------------------------------
    def _node_repr(self, nodes: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        dev = self._device
        idx = nodes.detach().cpu().numpy()
        feats = torch.as_tensor(self.buf_feat[idx], device=dev)             # [B,K,F]
        times = torch.as_tensor(self.buf_t[idx], device=dev, dtype=torch.float32)
        mask = torch.as_tensor(self.buf_valid[idx], device=dev,
                               dtype=torch.float32).unsqueeze(-1)           # [B,K,1]
        dt = (t.unsqueeze(1) - times).clamp(min=0.0)                        # [B,K]
        tokens = torch.cat([feats, self.time_encoder(dt)], dim=-1) * mask   # [B,K,tok]

        h = self.token_proj(tokens)                                        # [B,K,H]
        h = h + self.token_mix(h.transpose(1, 2)).transpose(1, 2)          # mix neighbors
        h = h + self.channel_mix(self.channel_norm(h))                     # mix channels
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)     # [B,H]
        emb = self.out(torch.cat([pooled, self.id_embed(nodes)], dim=-1))
        return emb

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
