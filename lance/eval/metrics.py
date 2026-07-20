"""MRR and Hit@k, computed with the OGB/TGB tie-aware ranking convention."""
from __future__ import annotations

import numpy as np
import torch


def mrr_and_hits(pos_scores: torch.Tensor, neg_scores: torch.Tensor,
                 k: int = 10) -> dict[str, float]:
    """Compute MRR and Hit@k.

    Args:
        pos_scores: [N] score of each positive edge.
        neg_scores: [N, M] scores of M negatives for each positive.
        k: cutoff for Hit@k.

    Uses the tie-aware rank ``0.5*(optimistic + pessimistic) + 1`` (OGB/TGB).
    """
    pos = pos_scores.view(-1, 1)
    optimistic = (neg_scores > pos).sum(dim=1)
    pessimistic = (neg_scores >= pos).sum(dim=1)
    rank = 0.5 * (optimistic + pessimistic).float() + 1.0
    return {
        "mrr": float((1.0 / rank).mean().item()),
        f"hits@{k}": float((rank <= k).float().mean().item()),
    }


def average_precision_auc(pos_scores: torch.Tensor,
                          neg_scores: torch.Tensor) -> dict[str, float]:
    """Binary AP and ROC-AUC treating positives vs (flattened) negatives."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.concatenate([np.ones(pos_scores.numel()),
                        np.zeros(neg_scores.numel())])
    s = torch.cat([pos_scores.flatten(), neg_scores.flatten()]).cpu().numpy()
    if y.sum() == 0 or y.sum() == len(y):
        return {"ap": float("nan"), "auc": float("nan")}
    return {"ap": float(average_precision_score(y, s)),
            "auc": float(roc_auc_score(y, s))}


def detection_pr(suspicion: np.ndarray, adv_mask: np.ndarray,
                 q: float = 0.10) -> dict[str, float]:
    """Precision/recall of flagging injected edges.

    ``suspicion`` is a per-edge score (higher = more suspicious); the top-``q``
    fraction is "flagged". ``adv_mask`` marks the truly-injected edges. Measures
    whether a defense's screening actually finds the attack.
    """
    if len(suspicion) == 0 or adv_mask.sum() == 0:
        return {"det_precision": float("nan"), "det_recall": float("nan")}
    k = max(1, int(q * len(suspicion)))
    flagged = np.zeros(len(suspicion), bool)
    flagged[np.argsort(-suspicion)[:k]] = True
    tp = int((flagged & adv_mask).sum())
    return {"det_precision": tp / max(int(flagged.sum()), 1),
            "det_recall": tp / max(int(adv_mask.sum()), 1)}


def build_history(data) -> dict[int, np.ndarray]:
    """Map each source node to the destinations it linked to in train and val.

    These support historical negatives: a source's past partners other than the
    current positive. They are the hard negatives that deletions act on, since
    removing a true edge leaves its endpoint as a confusable past partner, and so
    the regime in which deletion damage is measurable.
    """
    from collections import defaultdict
    hist: dict[int, list] = defaultdict(list)
    for which in ("train", "val"):
        s, d, _, _ = data.split(which)
        for u, v in zip(s.tolist(), d.tolist()):
            hist[u].append(v)
    return {k: np.array(v, dtype=np.int64) for k, v in hist.items()}


def _neg_matrix(src_np, pos_dst_np, dst_pool, num_neg, hist_frac, history, rng):
    n_hist = int(num_neg * hist_frac)
    out = np.empty((len(src_np), num_neg), dtype=np.int64)
    for i, (s, pos_dst) in enumerate(zip(src_np, pos_dst_np)):
        h = history.get(int(s)) if history is not None else None
        hist_pool = h[h != pos_dst] if h is not None else None
        random_pool = dst_pool[dst_pool != pos_dst]
        if len(random_pool) == 0:
            random_pool = dst_pool
        hist = (rng.choice(hist_pool, size=n_hist)
                if (hist_pool is not None and len(hist_pool) and n_hist)
                else rng.choice(random_pool, size=n_hist))
        rand = rng.choice(random_pool, size=num_neg - n_hist)
        out[i] = np.concatenate([hist, rand])
    return out


@torch.no_grad()
def evaluate_link_prediction(model, data, which: str, neg_sampler,
                             num_neg: int = 100, k: int = 10, batch_size: int = 200,
                             device: str = "cpu", history=None,
                             hist_frac: float = 0.0) -> dict[str, float]:
    """Rank each positive edge against ``num_neg`` negatives (random and/or
    historical). Memory is advanced chronologically so eval is causal."""
    model.eval()
    rng = np.random.default_rng(12345)
    dst_pool = neg_sampler.unique_dst
    all_pos, all_neg = [], []
    for batch in data.iter_batches(which, batch_size, device=device):
        b = len(batch)
        if b == 0:
            continue
        if hist_frac > 0.0:
            neg = _neg_matrix(batch.src.cpu().numpy(), batch.dst.cpu().numpy(),
                              dst_pool, num_neg, hist_frac, history, rng)
        else:
            neg = neg_sampler.sample_matrix(
                b, num_neg, positive_dst=batch.dst.cpu().numpy())
        neg_t = torch.as_tensor(neg, dtype=torch.long, device=device)
        pos_s, neg_s = model.score_pos_neg(batch, neg_t)        # [b], [b, num_neg]
        all_pos.append(pos_s.detach().cpu())
        all_neg.append(neg_s.detach().cpu())
        model.advance_memory(batch)                             # causal update
    pos = torch.cat(all_pos) if all_pos else torch.zeros(0)
    neg = torch.cat(all_neg) if all_neg else torch.zeros(0, num_neg)
    if pos.numel() == 0:
        return {"mrr": 0.0, f"hits@{k}": 0.0}
    return mrr_and_hits(pos, neg, k=k)
