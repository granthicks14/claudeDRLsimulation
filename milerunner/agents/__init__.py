"""RL agents and neural-network architectures."""
from __future__ import annotations

from .factory import ALL_ALGOS, OFF_POLICY, ON_POLICY, algo_family, build_agent
from .networks import EXTRACTORS, make_extractor_kwargs

__all__ = ["build_agent", "ALL_ALGOS", "ON_POLICY", "OFF_POLICY",
           "algo_family", "EXTRACTORS", "make_extractor_kwargs"]
