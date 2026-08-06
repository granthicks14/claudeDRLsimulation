"""Environments for the mile-running platform."""
from __future__ import annotations

from .mile_env import EnvConfig, MileRunEnv
from .rewards import RewardWeights
from .observations import observation_labels, observation_dim

try:
    from gymnasium.envs.registration import register

    register(id="MileRun-v0", entry_point="milerunner.envs.mile_env:MileRunEnv")
except Exception:  # pragma: no cover
    pass

__all__ = ["MileRunEnv", "EnvConfig", "RewardWeights",
           "observation_labels", "observation_dim"]
