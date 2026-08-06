"""Deterministic seeding across numpy, python and torch."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def seed_everything(seed: Optional[int]) -> int:
    """Seed all RNGs. Returns the seed actually used (random if ``None``)."""
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch optional at import time
        pass
    return seed
