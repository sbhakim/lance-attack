"""Deterministic seeding across Python, NumPy, and PyTorch."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 0, deterministic: bool = False) -> int:
    """Seed all RNGs used by the project.

    Args:
        seed: The seed value.
        deterministic: If True, force CuDNN into deterministic mode (slower).

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
    return seed
