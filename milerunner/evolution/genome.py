"""Genome: the evolvable specification of one runner-agent.

A genome encodes the RL algorithm, the network architecture and its size, the
optimisation hyperparameters, the exploration parameters, a training-schedule
gene, and the *reward weights*. Evolution mutates and recombines these across a
population. Note what is **not** here: any explicit stride pattern, cadence,
pacing schedule or breathing rhythm. Those behaviours are produced by the
trained policy under the physics — evolution only tunes the learning machinery
and the trade-off weights, never the strategy itself.
"""
from __future__ import annotations

import copy
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..agents.factory import ALL_ALGOS
from ..agents.networks import EXTRACTORS
from ..envs.rewards import RewardWeights

ARCHES = ["mlp", "cnn", "transformer", "gru"]

# (low, high, log-scale?) bounds for continuous hyperparameter genes.
CONT_BOUNDS: Dict[str, tuple] = {
    "learning_rate": (1e-5, 3e-3, True),
    "gamma": (0.95, 0.9997, False),
    "ent_coef": (1e-4, 0.05, True),
    "gae_lambda": (0.85, 0.99, False),
    "clip_range": (0.1, 0.4, False),
    "tau": (0.001, 0.02, True),
    "action_noise": (0.02, 0.4, True),
}
INT_BOUNDS: Dict[str, tuple] = {
    "batch_size": (64, 512),
    "n_steps": (256, 4096),
    "features_dim": (64, 512),
    "hidden": (64, 512),
    "depth": (1, 4),
    "gradient_steps": (1, 8),
}
ACTIVATIONS = ["relu", "tanh", "gelu"]


@dataclass
class Genome:
    algo: str = "ppo"
    arch: str = "mlp"
    activation: str = "relu"
    hyperparams: Dict[str, float] = field(default_factory=dict)
    reward_weights: Dict[str, float] = field(default_factory=lambda: RewardWeights().to_dict())
    # A "training schedule" gene: relative compute budget multiplier for PBT.
    train_budget_mult: float = 1.0
    genome_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0

    # ------------------------------------------------------------------ #
    def hp_for_agent(self) -> Dict[str, Any]:
        hp = dict(self.hyperparams)
        hp["activation"] = self.activation
        return hp

    def reward_weights_obj(self) -> RewardWeights:
        return RewardWeights.from_dict(self.reward_weights)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.__dict__)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Genome":
        return cls(**copy.deepcopy(d))


def _as_algo_list(algos) -> List[str]:
    if algos is None:
        return list(ALL_ALGOS)
    if isinstance(algos, str):
        return [algos]
    return list(algos)


def random_genome(rng: np.random.Generator, algos: Optional[List[str]] = None,
                  generation: int = 0) -> Genome:
    algos = _as_algo_list(algos)
    algo = str(rng.choice(algos))
    arch = str(rng.choice(ARCHES))
    hp: Dict[str, float] = {}
    for name, (lo, hi, log) in CONT_BOUNDS.items():
        if log:
            hp[name] = float(np.exp(rng.uniform(math.log(lo), math.log(hi))))
        else:
            hp[name] = float(rng.uniform(lo, hi))
    for name, (lo, hi) in INT_BOUNDS.items():
        hp[name] = int(rng.integers(lo, hi + 1))
    # reward weights start from defaults, jittered so the population is diverse
    rw = RewardWeights().to_dict()
    for k in rw:
        rw[k] = float(rw[k] * rng.uniform(0.7, 1.4))
    return Genome(
        algo=algo, arch=arch, activation=str(rng.choice(ACTIVATIONS)),
        hyperparams=hp, reward_weights=rw,
        train_budget_mult=float(np.exp(rng.uniform(math.log(0.6), math.log(1.8)))),
        generation=generation,
    )


def _mutate_continuous(value: float, lo: float, hi: float, log: bool,
                       rng: np.random.Generator, strength: float) -> float:
    if log:
        v = math.log(value)
        v += rng.normal(0, strength) * (math.log(hi) - math.log(lo))
        value = math.exp(v)
    else:
        value += rng.normal(0, strength) * (hi - lo)
    return float(np.clip(value, lo, hi))


def mutate(g: Genome, rng: np.random.Generator, strength: float = 0.15,
           algo_swap_prob: float = 0.1, arch_swap_prob: float = 0.1,
           algos: Optional[List[str]] = None, generation: Optional[int] = None) -> Genome:
    """Return a mutated copy of ``g``."""
    algos = _as_algo_list(algos)
    child = Genome.from_dict(g.to_dict())
    child.genome_id = uuid.uuid4().hex[:12]
    child.parent_ids = [g.genome_id]
    if generation is not None:
        child.generation = generation

    if rng.random() < algo_swap_prob:
        child.algo = str(rng.choice(algos))
    if rng.random() < arch_swap_prob:
        child.arch = str(rng.choice(ARCHES))
    if rng.random() < 0.1:
        child.activation = str(rng.choice(ACTIVATIONS))

    for name, (lo, hi, log) in CONT_BOUNDS.items():
        if name in child.hyperparams and rng.random() < 0.6:
            child.hyperparams[name] = _mutate_continuous(
                child.hyperparams[name], lo, hi, log, rng, strength)
    for name, (lo, hi) in INT_BOUNDS.items():
        if name in child.hyperparams and rng.random() < 0.5:
            span = hi - lo
            val = child.hyperparams[name] + int(round(rng.normal(0, strength) * span))
            child.hyperparams[name] = int(np.clip(val, lo, hi))
    # mutate reward weights (bounded, non-negative)
    for k in child.reward_weights:
        if rng.random() < 0.5:
            child.reward_weights[k] = float(max(0.0,
                child.reward_weights[k] * math.exp(rng.normal(0, strength))))
    child.train_budget_mult = float(np.clip(
        child.train_budget_mult * math.exp(rng.normal(0, strength)), 0.4, 3.0))
    return child


def crossover(a: Genome, b: Genome, rng: np.random.Generator,
              generation: Optional[int] = None) -> Genome:
    """Uniform-crossover recombination of two parents."""
    child = Genome.from_dict(a.to_dict())
    child.genome_id = uuid.uuid4().hex[:12]
    child.parent_ids = [a.genome_id, b.genome_id]
    if generation is not None:
        child.generation = generation
    child.algo = a.algo if rng.random() < 0.5 else b.algo
    child.arch = a.arch if rng.random() < 0.5 else b.arch
    child.activation = a.activation if rng.random() < 0.5 else b.activation
    for name in set(a.hyperparams) | set(b.hyperparams):
        av = a.hyperparams.get(name)
        bv = b.hyperparams.get(name)
        if av is None:
            child.hyperparams[name] = bv
        elif bv is None:
            child.hyperparams[name] = av
        else:
            child.hyperparams[name] = av if rng.random() < 0.5 else bv
    for k in set(a.reward_weights) | set(b.reward_weights):
        av = a.reward_weights.get(k, 0.0)
        bv = b.reward_weights.get(k, 0.0)
        child.reward_weights[k] = float(0.5 * (av + bv)) if rng.random() < 0.5 else \
            (av if rng.random() < 0.5 else bv)
    child.train_budget_mult = 0.5 * (a.train_budget_mult + b.train_budget_mult)
    return child
