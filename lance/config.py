"""Typed configuration objects and a YAML loader.

A single :class:`Config` aggregates data/model/training/attack/defense settings.
Configs are plain dataclasses so they are easy to construct in tests and to
serialize back out alongside experiment artifacts.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "../Dataset"          # directory containing the *.csv files
    name: str = "mooc"                # mooc | wikipedia | lastfm | bitcoinotc
    fmt: str = "jodie"                # jodie | bitcoinotc
    max_events: int | None = None     # cap #events (None = all); useful for smoke tests
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class ModelConfig:
    memory_dim: int = 100
    time_dim: int = 100
    embedding_dim: int = 100
    predictor_hidden: int = 80
    dropout: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    num_neg_train: int = 1
    grad_clip: float = 1.0
    device: str = "auto"              # auto | cpu | cuda
    seed: int = 0


@dataclass
class EvalConfig:
    num_neg: int = 100                # negatives per positive at eval time
    hits_k: int = 10
    historical_neg_frac: float = 0.0  # 0 => all-random; >0 mixes historical negatives


@dataclass
class AttackConfig:
    ptb_rate: float = 0.1             # budget delta = ptb_rate * |E_train|
    alpha: float = 0.6                # budget split: high-attraction vs bridge
    del_percentile: float = 85.0      # delete edges above this surrogate-likelihood pct
    inj_percentile: float = 10.0      # inject edges below this surrogate-likelihood pct
    impact_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)
    betweenness_k: int = 200          # k-sample approximation for betweenness
    high_impact_frac: float = 0.1     # fraction of nodes treated as "high impact"
    # --- LANCE extensions over HIA ---
    knowledge: str = "k1"             # k1 full | k2 limited-future | k3 streaming
    lk_cutoff_frac: float = 0.7       # K2/K3: surrogate sees only this prefix (events <= t_a)
    adaptive: bool = True             # adaptive (vs fixed 50/50) inject/delete budget split
    query_aware_injection: bool = False  # score injections with source-future/history proxies
    stream_windows: int = 4           # K3: number of online windows


@dataclass
class DefenseConfig:
    mode: str = "dtshield"            # none | tshield | cosine | dtshield
    # T-SHIELD baseline: single-tail, cosine-annealed low-score edge filter
    tshield_pct_s: float = 10.0       # start drop-percentile (of surrogate yhat)
    tshield_pct_e: float = 40.0       # end drop-percentile (annealed up over epochs)
    # GNNGuard-style cosine baseline: drop low embedding-affinity edges
    cosine_q: float = 0.10
    # C1: dual-tail importance-conditioned screening
    screen_q: float = 0.10            # fraction of injection-suspect edges to down-weight
    impute_q: float = 0.02            # fraction of deletion-suspect pairs to impute
    # C2: two-sided temporal-consistency band
    band_low: float = 0.25            # tau_ell (lower band multiplier)
    band_high: float = 3.0            # tau_h  (upper band multiplier)
    clean_prefix_frac: float = 0.2    # fraction of train stream treated as clean for band fit
    # C3: importance-guided adversarial training
    adv_every: int = 0                # 0 disables C3; else run inner attack every K epochs
    adv_ptb_rate: float = 0.05
    smooth_lambda: float = 0.02       # temporal-smoothness regularizer weight


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)

    # ------------------------------------------------------------------
    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path, "r") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        return Config.from_dict(raw)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "Config":
        def build(cls, key):
            sub = raw.get(key, {}) or {}
            allowed = {f.name for f in dataclasses.fields(cls)}
            return cls(**{k: v for k, v in sub.items() if k in allowed})

        return Config(
            data=build(DataConfig, "data"),
            model=build(ModelConfig, "model"),
            train=build(TrainConfig, "train"),
            eval=build(EvalConfig, "eval"),
            attack=build(AttackConfig, "attack"),
            defense=build(DefenseConfig, "defense"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
