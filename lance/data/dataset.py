"""Continuous-time temporal-graph dataset.

Supports two on-disk formats:
  * ``jodie``       : SNAP/JODIE CSVs (``user_id,item_id,timestamp,state_label,<feats>``)
                      used by WIKI / Reddit / MOOC / LastFM.
  * ``bitcoinotc``  : SNAP signed-network CSV (``source,target,rating,timestamp``).

Nodes are remapped to a single contiguous id space (users first, then items for
the bipartite JODIE graphs). Events are sorted chronologically and split 70/15/15
by time. Downstream code consumes only the integer arrays held here and is
therefore independent of the source format.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class EdgeBatch:
    """A chronological batch of interaction events (all 1-D, equal length)."""
    src: torch.Tensor      # int64 [B]
    dst: torch.Tensor      # int64 [B]
    t: torch.Tensor        # float64/float32 [B]
    feat: torch.Tensor     # float32 [B, F]  (zeros if dataset is featureless)

    def to(self, device: str | torch.device) -> "EdgeBatch":
        return EdgeBatch(self.src.to(device), self.dst.to(device),
                         self.t.to(device), self.feat.to(device))

    def __len__(self) -> int:
        return int(self.src.numel())


class TemporalGraphData:
    """Holds the full event stream plus chronological train/val/test masks."""

    def __init__(self, src: np.ndarray, dst: np.ndarray, t: np.ndarray,
                 feat: np.ndarray, num_nodes: int, val_ratio: float, test_ratio: float):
        order = np.argsort(t, kind="stable")
        self.src = src[order].astype(np.int64)
        self.dst = dst[order].astype(np.int64)
        self.t = t[order].astype(np.float64)
        self.feat = feat[order].astype(np.float32)
        self.num_nodes = int(num_nodes)
        self.num_feats = int(feat.shape[1])

        n = len(self.src)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        n_train = n - n_val - n_test
        self.train_idx = np.arange(0, n_train)
        self.val_idx = np.arange(n_train, n_train + n_val)
        self.test_idx = np.arange(n_train + n_val, n)

    @classmethod
    def from_splits(cls, num_nodes: int, num_feats: int,
                    train: tuple, val: tuple, test: tuple) -> "TemporalGraphData":
        """Build a dataset with *explicit* train/val/test splits (each a
        ``(src, dst, t, feat)`` tuple). Used to inject a poisoned train stream
        while keeping val/test clean -- the standard poisoning-eval protocol."""
        obj = cls.__new__(cls)
        parts = []
        for s, d, t, f in (train, val, test):
            order = np.argsort(t, kind="stable")
            parts.append((s[order], d[order], t[order], f[order]))
        obj.src = np.concatenate([p[0] for p in parts]).astype(np.int64)
        obj.dst = np.concatenate([p[1] for p in parts]).astype(np.int64)
        obj.t = np.concatenate([p[2] for p in parts]).astype(np.float64)
        obj.feat = np.concatenate([p[3] for p in parts]).astype(np.float32)
        obj.num_nodes, obj.num_feats = int(num_nodes), int(num_feats)
        n_tr, n_va = len(parts[0][0]), len(parts[1][0])
        n_te = len(parts[2][0])
        obj.train_idx = np.arange(0, n_tr)
        obj.val_idx = np.arange(n_tr, n_tr + n_va)
        obj.test_idx = np.arange(n_tr + n_va, n_tr + n_va + n_te)
        return obj

    # -- convenience accessors -------------------------------------------------
    def split(self, which: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = {"train": self.train_idx, "val": self.val_idx, "test": self.test_idx}[which]
        return self.src[idx], self.dst[idx], self.t[idx], self.feat[idx]

    def num_train_events(self) -> int:
        return len(self.train_idx)

    def iter_batches(self, which: str, batch_size: int, device: str = "cpu"):
        """Yield chronological :class:`EdgeBatch` objects for a split."""
        s, d, t, f = self.split(which)
        for i in range(0, len(s), batch_size):
            sl = slice(i, i + batch_size)
            yield EdgeBatch(
                torch.as_tensor(s[sl], dtype=torch.long),
                torch.as_tensor(d[sl], dtype=torch.long),
                torch.as_tensor(t[sl], dtype=torch.float32),
                torch.as_tensor(f[sl], dtype=torch.float32),
            ).to(device)


def _read_jodie(path: str, max_events: int | None,
                bipartite: bool = True) -> tuple[np.ndarray, ...]:
    """Read a JODIE-layout CSV: ``user_id,item_id,timestamp,state_label,<feats>``.

    ``bipartite`` selects the node-id space, and getting it wrong is silent rather
    than loud. The JODIE datasets (Wikipedia, Reddit, MOOC, LastFM) have disjoint
    user and item spaces, so ids must be remapped into separate ranges. The
    single-node-type datasets distributed in the same layout (UCI, Enron, CanParl,
    UNtrade, ...) reuse one id space, and remapping them as bipartite would split
    every node into an unconnected "source copy" and "destination copy" --
    doubling the node count and destroying exactly the recurrence structure this
    project measures.
    """
    nrows = max_events if max_events else None
    df = pd.read_csv(path, skiprows=1, header=None, nrows=nrows)
    u = df.iloc[:, 0].to_numpy()
    i = df.iloc[:, 1].to_numpy()
    t = df.iloc[:, 2].to_numpy(dtype=np.float64)
    feat = df.iloc[:, 4:].to_numpy(dtype=np.float32) if df.shape[1] > 4 \
        else np.zeros((len(u), 1), dtype=np.float32)
    return u, i, t, feat, bipartite


def _read_dgb(path: str, max_events: int | None) -> tuple[np.ndarray, ...]:
    """Read a DGB/TGN-preprocessed ``ml_<name>.csv`` plus its edge-feature ``.npy``.

    Layout is ``<row index>,u,i,ts,label,idx`` -- an unnamed index column shifts
    every field one place relative to the raw JODIE layout, so reading it with the
    JODIE reader would silently take ``u`` as the timestamp. Edge features live in
    ``ml_<name>.npy``, whose row 0 is TGN's padding row and whose row ``idx``
    belongs to the edge with that ``idx`` (1-based).

    Several of these graphs carry no attributes at all and DGB stores them as
    zero matrices of a nominal width (UCI as 100 columns, Enron as 32). Those are
    collapsed to a single zero column: keeping them would advertise a feature
    dimensionality the data does not have, and would let a reader mistake an
    attribute-free graph for a feature-rich one.
    """
    df = pd.read_csv(path, nrows=max_events if max_events else None)
    u = df["u"].to_numpy()
    i = df["i"].to_numpy()
    t = df["ts"].to_numpy(dtype=np.float64)

    feat_path = os.path.splitext(path)[0] + ".npy"
    feat = None
    if os.path.isfile(feat_path):
        raw = np.load(feat_path)
        idx = df["idx"].to_numpy(dtype=np.int64)
        if idx.max() < len(raw):
            feat = raw[idx].astype(np.float32)
    if feat is None or np.allclose(feat, 0.0):
        feat = np.zeros((len(u), 1), dtype=np.float32)
    return u, i, t, feat, False        # DGB single-node-type graphs share one id space


def _read_bitcoinotc(path: str, max_events: int | None) -> tuple[np.ndarray, ...]:
    nrows = max_events if max_events else None
    df = pd.read_csv(path, header=None, nrows=nrows)
    u = df.iloc[:, 0].to_numpy()
    i = df.iloc[:, 1].to_numpy()
    rating = df.iloc[:, 2].to_numpy(dtype=np.float32).reshape(-1, 1)
    t = df.iloc[:, 3].to_numpy(dtype=np.float64)
    return u, i, t, rating, False  # bipartite=False (trust network)


def load_dataset(root: str, name: str, fmt: str = "jodie",
                 max_events: int | None = None,
                 val_ratio: float = 0.15, test_ratio: float = 0.15) -> TemporalGraphData:
    """Load a dataset by name from ``root`` and return a :class:`TemporalGraphData`."""
    if fmt == "dgb":                    # ships as <root>/<name>/ml_<name>.csv
        path = os.path.join(root, name, f"ml_{name}.csv")
    else:
        fname = {"bitcoinotc": "bitcoinotc.csv"}.get(name, f"{name}.csv")
        path = os.path.join(root, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"dataset file not found: {path}")

    if fmt == "dgb":
        u, i, t, feat, bipartite = _read_dgb(path, max_events)
    elif fmt == "bitcoinotc":
        u, i, t, feat, bipartite = _read_bitcoinotc(path, max_events)
    elif fmt == "jodie_unipartite":
        # same file layout, one shared node-id space (DGB single-node-type graphs)
        u, i, t, feat, bipartite = _read_jodie(path, max_events, bipartite=False)
    else:
        u, i, t, feat, bipartite = _read_jodie(path, max_events)

    # Remap to a contiguous node id space.
    if bipartite:
        u_ids = {x: k for k, x in enumerate(np.unique(u))}
        offset = len(u_ids)
        i_ids = {x: offset + k for k, x in enumerate(np.unique(i))}
        src = np.array([u_ids[x] for x in u], dtype=np.int64)
        dst = np.array([i_ids[x] for x in i], dtype=np.int64)
        num_nodes = offset + len(i_ids)
    else:
        all_ids = {x: k for k, x in enumerate(np.unique(np.concatenate([u, i])))}
        src = np.array([all_ids[x] for x in u], dtype=np.int64)
        dst = np.array([all_ids[x] for x in i], dtype=np.int64)
        num_nodes = len(all_ids)

    return TemporalGraphData(src, dst, t, feat, num_nodes, val_ratio, test_ratio)
