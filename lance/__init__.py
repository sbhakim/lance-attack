"""LANCE: a Limited-knowledge, Adaptive, Node-importance, Continuous-time Edge-attack.

LANCE is a restricted black-box *poisoning attack* on Temporal Graph Neural
Networks (TGNNs) for link prediction. It extends the High Impact Attack (HIA) with
its three components --- a surrogate model, strategic node selection by an Impact
score, and a hybrid (inject + delete) perturbation --- plus limited-knowledge
operation (K1/K2/K3), adaptive-budget allocation, and a streaming variant.

DT-SHIELD (a deletion-aware, importance-guided defense) is included as the
six-month *extension plan*, not the primary deliverable.

The package is organised by responsibility:
  - ``lance.data``     : temporal-graph datasets and negative sampling
  - ``lance.models``   : a self-contained memory-based TGNN + link predictor
  - ``lance.attack``   : node-importance scoring + the HIA/LANCE poisoning attacks
  - ``lance.defense``  : the DT-SHIELD components (C1/C2/C3) -- extension plan
  - ``lance.training`` : training loops (clean victim/surrogate; optional defense)
  - ``lance.eval``     : ranking metrics (MRR, Hit@k) + attack/defense diagnostics
  - ``lance.utils``    : seeding and logging helpers
"""

__version__ = "0.1.0"
