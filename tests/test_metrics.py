"""Unit tests for the ranking metrics (closed-form expectations)."""
import torch
import numpy as np

from lance.eval.metrics import mrr_and_hits
from lance.data.negatives import NegativeSampler


def test_perfect_ranker():
    pos = torch.tensor([3.0, 1.0, 5.0])
    neg = torch.full((3, 50), -10.0)
    m = mrr_and_hits(pos, neg, k=10)
    assert m["mrr"] == 1.0
    assert m["hits@10"] == 1.0


def test_worst_ranker():
    # positive below all 50 negatives -> rank 51 -> MRR = 1/51, not in top-10
    pos = torch.zeros(4)
    neg = torch.ones(4, 50)
    m = mrr_and_hits(pos, neg, k=10)
    assert abs(m["mrr"] - 1.0 / 51.0) < 1e-6
    assert m["hits@10"] == 0.0


def test_tie_aware_rank():
    # all tied: tie-aware rank = 0.5*(0 + M) + 1 = M/2 + 1
    pos = torch.zeros(5)
    neg = torch.zeros(5, 9)
    m = mrr_and_hits(pos, neg, k=10)
    assert abs(m["mrr"] - 1.0 / (9 / 2 + 1)) < 1e-6


def test_negative_sampler_excludes_current_positive():
    sampler = NegativeSampler(np.array([10, 11, 12]), seed=0)
    positives = np.array([10, 11, 12, 10])
    negatives = sampler.sample_matrix(len(positives), 100, positive_dst=positives)
    assert not np.any(negatives == positives[:, None])
