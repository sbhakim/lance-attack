"""Defense-component contracts: weights in [0,1], HIA perturbs, defense trains."""
import numpy as np

from lance.attack import compute_impact, hia_attack
from lance.defense import build_defense
from lance.defense.consistency import Consistency
from lance.models import TGNLite
from lance.training import Trainer


def test_hia_perturbs(tiny_data):
    s, d, t, f = tiny_data.split("train")
    imp = compute_impact(s, d, tiny_data.num_nodes, betweenness_k=20)
    rng = np.random.default_rng(0)
    score_fn = lambda a, b, c: rng.uniform(size=len(a))  # noqa: E731  (dummy surrogate)
    res = hia_attack(s, d, t, f, tiny_data.num_nodes, imp, score_fn, ptb_rate=0.2)
    assert res.n_deleted > 0 and res.n_injected >= 0
    assert len(res.src) == len(res.dst) == len(res.t)


def test_consistency_band_two_sided(tiny_data):
    m = TGNLite(tiny_data.num_nodes, tiny_data.num_feats, 16, 8, 16, 16)
    m.reset_state("cpu")
    c2 = Consistency(band_low=0.25, band_high=3.0).fit(tiny_data)
    batch = next(tiny_data.iter_batches("train", 32))
    m.advance_memory(batch)
    w = c2.weights(m, batch)
    assert w.shape == (len(batch),)
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0


def test_defense_weights_and_training(tiny_data, tiny_cfg):
    m = TGNLite(tiny_data.num_nodes, tiny_data.num_feats, 16, 8, 16, 16)
    defense = build_defense(tiny_cfg, device="cpu")
    defense.precompute(tiny_data, m)
    batch = next(tiny_data.iter_batches("train", 32))
    m.reset_state("cpu")
    w = defense.weight_batch(m, batch)
    assert w.shape == (len(batch),)
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0 + 1e-6
    # defended training runs to completion
    hist = Trainer(m, tiny_cfg, device="cpu").fit(tiny_data, defense=defense, verbose=False)
    assert len(hist) == tiny_cfg.train.epochs


def test_build_defense_none(tiny_cfg):
    tiny_cfg.defense.mode = "none"
    assert build_defense(tiny_cfg) is None
