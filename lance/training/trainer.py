"""Training loop with truncated backpropagation through time (TBPTT).

Each batch is scored using memory produced only by earlier batches and memory is
then advanced, so no edge informs its own prediction. Losses are accumulated over
a window of ``bptt`` batches and backpropagated once, which gives the GRU memory
updater gradient without leakage.

An optional ``defense`` object supplies the DT-SHIELD components; when it is
``None`` the loop trains an undefended victim. The interface is duck-typed:
``precompute(data, model)`` for setup, ``weight_batch(model, batch)`` returning a
per-sample weight in [0, 1] (C1 and C2 reweighting), ``adv_negatives(epoch, data,
model)`` returning optional hard negatives (C3), and a ``smooth_lambda`` attribute
for the C2 stability term.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from lance.data import NegativeSampler
from lance.eval.metrics import evaluate_link_prediction, build_history
from lance.utils import get_logger

_LOG = get_logger()


def resolve_device(pref: str = "auto") -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


class Trainer:
    def __init__(self, model, cfg, device: str | None = None, bptt: int = 5):
        self.model = model
        self.cfg = cfg
        self.device = device or resolve_device(cfg.train.device)
        self.bptt = bptt
        self.model.to(self.device)

    # ------------------------------------------------------------------
    def _batch_loss(self, batch, neg_sampler, num_neg, weights=None,
                    adv_neg=None, extra_pos=None):
        neg = neg_sampler.sample_matrix(
            len(batch), num_neg, positive_dst=batch.dst.detach().cpu().numpy())
        neg_t = torch.as_tensor(neg, dtype=torch.long, device=self.device)
        pos, negs = self.model.score_pos_neg(batch, neg_t)             # [B], [B, num_neg]

        pos_loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos),
                                                       reduction="none")
        if weights is not None:
            pos_loss = pos_loss * weights
        loss = pos_loss.mean() + F.binary_cross_entropy_with_logits(
            negs, torch.zeros_like(negs))

        # C3: treat importance-guided injected edges as additional hard negatives.
        if adv_neg is not None and len(adv_neg) > 0:
            a = self.model.score_pairs(adv_neg.src, adv_neg.dst, adv_neg.t)
            loss = loss + F.binary_cross_entropy_with_logits(a, torch.zeros_like(a))

        # C1: imputed deletion-suspect edges, re-introduced as positives,
        # at a low weight (they are inferred, not observed) so they help recover
        # deletions under attack without polluting the clean objective.
        if extra_pos is not None and len(extra_pos) > 0:
            m = min(len(extra_pos), len(batch))
            idx = torch.randint(0, len(extra_pos), (m,), device=self.device)
            p = self.model.score_pairs(extra_pos.src[idx], extra_pos.dst[idx],
                                       extra_pos.t[idx])
            loss = loss + 0.25 * F.binary_cross_entropy_with_logits(p, torch.ones_like(p))
        return loss

    # ------------------------------------------------------------------
    def fit(self, data, defense=None, verbose: bool = True):
        c = self.cfg
        opt = torch.optim.Adam(self.model.parameters(), lr=c.train.lr,
                               weight_decay=c.train.weight_decay)
        train_dst = data.split("train")[1]
        neg_sampler = NegativeSampler(train_dst, seed=c.train.seed)
        eval_neg = NegativeSampler(train_dst, seed=c.train.seed + 1,
                                   historical_frac=c.eval.historical_neg_frac)
        smooth_lambda = getattr(defense, "smooth_lambda", 0.0) if defense else 0.0

        if defense is not None:
            defense.precompute(data, self.model)

        eval_hist = build_history(data)          # for historical-negative eval
        hist_frac = c.eval.historical_neg_frac
        history = []
        for epoch in range(1, c.train.epochs + 1):
            if defense is not None and hasattr(defense, "on_epoch_start"):
                defense.on_epoch_start(epoch, c.train.epochs)
            self.model.train()
            self.model.reset_state(self.device)
            adv_neg = defense.adv_negatives(epoch, data, self.model) if defense else None
            if adv_neg is not None:
                adv_neg = adv_neg.to(self.device)
            extra_pos = defense.extra_positives(epoch, data, self.model) if defense else None
            # restore training state after any defense routine that warmed memory
            self.model.train()
            self.model.reset_state(self.device)

            opt.zero_grad()
            window_loss, window_n, total = 0.0, 0, 0.0
            batches = list(data.iter_batches("train", c.train.batch_size, self.device))
            for bi, batch in enumerate(batches):
                if len(batch) == 0:
                    continue
                weights = None
                if defense is not None:
                    weights = defense.weight_batch(self.model, batch)   # no-grad inside
                loss = self._batch_loss(batch, neg_sampler, c.train.num_neg_train,
                                        weights=weights, adv_neg=adv_neg,
                                        extra_pos=extra_pos)
                if smooth_lambda > 0:                                    # C2 stability term
                    z = self.model._embed(batch.src, batch.t)
                    loss = loss + smooth_lambda * z.pow(2).mean()
                window_loss = window_loss + loss
                window_n += 1
                self.model.update_memory(batch)                         # advance state

                if (bi + 1) % self.bptt == 0 or bi == len(batches) - 1:
                    (window_loss / max(window_n, 1)).backward()
                    if c.train.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       c.train.grad_clip)
                    opt.step()
                    opt.zero_grad()
                    total += float(window_loss.detach()) / max(window_n, 1)
                    window_loss, window_n = 0.0, 0
                    self.model.detach_state()

            val = evaluate_link_prediction(self.model, data, "val", eval_neg,
                                           num_neg=c.eval.num_neg, k=c.eval.hits_k,
                                           batch_size=c.train.batch_size, device=self.device,
                                           history=eval_hist, hist_frac=hist_frac)
            history.append({"epoch": epoch, "loss": total, **val})
            if verbose:
                _LOG.info(f"epoch {epoch:02d} | loss {total:.4f} | "
                          f"val MRR {val['mrr']:.4f} | val Hit@{c.eval.hits_k} "
                          f"{val[f'hits@{c.eval.hits_k}']:.4f}")
        return history

    @torch.no_grad()
    def test(self, data, history=None, negative_dst_pool=None):
        """Evaluate on the test split. ``history`` (src -> past dsts) lets the
        caller supply a *fixed clean* history so historical negatives are
        identical across clean/attacked/defended runs (a fair comparison)."""
        c = self.cfg
        self.model.reset_state(self.device)
        # warm memory through train+val so test starts from a realistic state
        for split in ("train", "val"):
            for batch in data.iter_batches(split, c.train.batch_size, self.device):
                if len(batch):
                    self.model.advance_memory(batch)
        # A benchmark may supply the clean training destination pool so every
        # condition is ranked against the same candidate universe.  Without
        # this, deleting the only occurrence of a destination can silently
        # change the evaluation task for an attacked condition.
        dst_pool = data.split("train")[1] if negative_dst_pool is None else negative_dst_pool
        eval_neg = NegativeSampler(dst_pool, seed=c.train.seed + 2)
        if history is None:
            history = build_history(data)
        return evaluate_link_prediction(self.model, data, "test", eval_neg,
                                        num_neg=c.eval.num_neg, k=c.eval.hits_k,
                                        batch_size=c.train.batch_size, device=self.device,
                                        history=history, hist_frac=c.eval.historical_neg_frac)
