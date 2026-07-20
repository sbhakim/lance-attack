"""A self-contained, memory-based Temporal GNN and link predictor.

The model mirrors the TGN family (per-node memory updated by a GRU, time
encoding, MLP link decoder) but is implemented in pure PyTorch so it runs on a
modern torch+CUDA stack without DGL. It exposes the hooks the defense needs:
per-node ``staleness`` (for C2) and ``surrogate_scores`` (for C1/HIA).
"""
from lance.models.tgn import TGNLite
from lance.models.link_predictor import LinkPredictor
from lance.models.memory import TimeEncoder

__all__ = ["TGNLite", "LinkPredictor", "TimeEncoder"]
