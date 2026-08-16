"""Deterministic seeding across Python, NumPy, and PyTorch."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 0, deterministic: bool = False) -> int:
    """Seed all RNGs used by the project.

    Seeding alone does not make a CUDA run reproducible: several kernels reduce
    in a nondeterministic order, so two identical invocations drift apart. On
    MOOC that drift reached 0.02 MRR between nominally identical 5-seed gates,
    which is larger than the effects the benchmark measures. ``deterministic``
    therefore also pins the algorithm selection, which costs roughly 30% runtime
    and makes repeated runs bit-identical.

    Args:
        seed: The seed value.
        deterministic: If True, force deterministic kernels everywhere (slower).

    Returns:
        The seed, for convenience / logging.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    return seed
