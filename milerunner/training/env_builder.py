"""Build (optionally vectorised, parallel) environments for training/eval.

A genome carries evolvable reward weights, so each agent trains against an
environment configured with *its* weights. This module turns a genome + a base
config into single or vectorised environments, with optional domain
randomisation of weather and body type for robustness.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..biomech.params import BodyParams
from ..envs.mile_env import EnvConfig, MileRunEnv
from ..envs.rewards import RewardWeights
from ..physics.track import Weather

try:
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    _HAVE_SB3 = True
except Exception:  # pragma: no cover
    _HAVE_SB3 = False


def make_env_fn(reward_weights: Optional[RewardWeights] = None,
                body: Optional[BodyParams] = None,
                weather: Optional[Weather] = None,
                config: Optional[EnvConfig] = None,
                seed: int = 0, rank: int = 0) -> Callable[[], MileRunEnv]:
    def _init():
        env = MileRunEnv(body=body, weather=weather, reward_weights=reward_weights,
                         config=config, seed=seed + rank)
        if _HAVE_SB3:
            env = Monitor(env)
        return env
    return _init


def build_vec_env(n_envs: int = 1, *, reward_weights: Optional[RewardWeights] = None,
                  body: Optional[BodyParams] = None, weather: Optional[Weather] = None,
                  config: Optional[EnvConfig] = None, seed: int = 0,
                  subprocess: bool = False):
    """Return a (Dummy|Subproc)VecEnv of ``n_envs`` mile environments."""
    if not _HAVE_SB3:  # pragma: no cover
        raise RuntimeError("stable-baselines3 required for vectorised envs")
    fns = [make_env_fn(reward_weights, body, weather, config, seed, rank=i)
           for i in range(n_envs)]
    if subprocess and n_envs > 1:
        return SubprocVecEnv(fns, start_method="spawn")
    return DummyVecEnv(fns)


def build_single_env(reward_weights: Optional[RewardWeights] = None,
                     body: Optional[BodyParams] = None,
                     weather: Optional[Weather] = None,
                     config: Optional[EnvConfig] = None, seed: int = 0) -> MileRunEnv:
    return MileRunEnv(body=body, weather=weather, reward_weights=reward_weights,
                      config=config, seed=seed)
