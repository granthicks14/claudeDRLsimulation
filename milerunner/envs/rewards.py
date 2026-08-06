"""Reward shaping for the mile-run task.

The reward is a *weighted sum of biomechanically meaningful terms*. It rewards
covering distance quickly, staying upright and balanced, running economically,
and finishing; it penalises falling, over-exertion beyond physiological limits,
joint-limit violations and energy waste. The **weights are evolvable** (see
:mod:`milerunner.evolution`) — the platform does not hard-code a pacing policy,
it lets population-based training discover which trade-offs produce the fastest
legal mile.

Every term is defined so that "faster legal mile" is the dominant signal;
shaping terms are there to make the sparse finish reward learnable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class RewardWeights:
    progress: float = 1.0            # per metre advanced this step
    speed: float = 0.05             # sustained forward velocity
    alive: float = 0.05             # per-step survival bonus
    upright: float = 0.10           # posture / balance
    energy_economy: float = 0.02    # reward for low metabolic cost per metre
    finish_bonus: float = 60.0      # reward for completing the mile
    time_bonus_scale: float = 400.0 # bonus inversely proportional to finish time
    fall_penalty: float = 12.0
    overexertion_penalty: float = 0.05
    joint_violation_penalty: float = 0.5
    energy_waste_penalty: float = 0.01
    lateral_penalty: float = 0.05
    action_cost: float = 0.002

    def to_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "RewardWeights":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RewardBreakdown:
    total: float
    terms: Dict[str, float] = field(default_factory=dict)


def compute_reward(w: RewardWeights, *, dprogress: float, speed: float,
                   uprightness: float, lateral_speed: float,
                   metabolic_power: float, mass: float,
                   overexertion: float, joint_violation: float,
                   action_sq: float, fell: bool, finished: bool,
                   finish_time: float, dt: float,
                   exhausted: bool) -> RewardBreakdown:
    terms: Dict[str, float] = {}
    terms["progress"] = w.progress * dprogress
    terms["speed"] = w.speed * max(speed, 0.0) * dt
    terms["alive"] = w.alive * dt if not fell else 0.0
    terms["upright"] = w.upright * max(uprightness, 0.0) * dt
    # Economy: reward advancing far per unit metabolic energy spent.
    energy_per_m = metabolic_power / max(speed, 0.3)
    terms["energy_economy"] = w.energy_economy * dt * np.exp(-energy_per_m / (6.0 * mass))
    terms["lateral"] = -w.lateral_penalty * abs(lateral_speed) * dt
    terms["overexertion"] = -w.overexertion_penalty * overexertion * dt
    terms["joint_violation"] = -w.joint_violation_penalty * joint_violation * dt
    terms["action_cost"] = -w.action_cost * action_sq * dt
    terms["energy_waste"] = -w.energy_waste_penalty * max(metabolic_power - 400.0, 0.0) / 1000.0 * dt
    terms["fall"] = -w.fall_penalty if fell else 0.0
    if exhausted and not finished:
        terms["exhaustion"] = -2.0
    if finished:
        terms["finish"] = w.finish_bonus
        # Faster finish -> larger bonus. World-record mile ~226 s; scale so that
        # times near/under that are strongly rewarded without going unbounded.
        terms["time_bonus"] = w.time_bonus_scale / max(finish_time, 60.0)
    total = float(sum(terms.values()))
    return RewardBreakdown(total=total, terms=terms)
