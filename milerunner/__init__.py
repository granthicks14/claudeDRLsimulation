"""MileRunner: a research platform for discovering the fastest human mile via
deep reinforcement learning and evolutionary optimization.

Public API re-exports the pieces most users need. Heavy optional dependencies
(mujoco, torch, stable-baselines3) are imported lazily by submodules so that
lightweight utilities remain importable without them.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .biomech.params import BodyParams
from .physics.track import Weather
from .envs.rewards import RewardWeights
from .utils.config import Config, load_config

__all__ = ["BodyParams", "Weather", "RewardWeights", "Config", "load_config",
           "__version__"]
