"""Muscle-group model: activation, local fatigue and available torque.

Each named muscle group (mapped onto the humanoid's joints) has a peak torque
budget. Sustained high activation drains a *local* fatigue reserve, which lowers
the torque the group can currently produce; resting it recovers the reserve.
This is what forces the agent to allocate effort across muscle groups and over
time, rather than sprinting flat out. The model reports, per group, the
fraction of peak torque currently available — which the physics layer uses to
clip actuator forces, guaranteeing the body never exceeds human strength.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .params import BodyParams

# The muscle groups called out in the brief, mapped to the joint whose torque
# budget they govern. Feet/calves -> ankle; quads/hamstrings -> knee;
# glutes -> hip; core -> trunk; shoulders/arms -> shoulder/elbow; neck -> neck.
MUSCLE_GROUPS: List[str] = [
    "calf", "quad", "hamstring", "glute", "core",
    "shoulder", "arm", "neck",
]

GROUP_TO_JOINT: Dict[str, str] = {
    "calf": "ankle",
    "quad": "knee",
    "hamstring": "knee",
    "glute": "hip",
    "core": "trunk",
    "shoulder": "shoulder",
    "arm": "elbow",
    "neck": "neck",
}


@dataclass
class MuscleState:
    fatigue: Dict[str, float]           # 0 (fresh) .. 1 (spent) per group
    available_frac: Dict[str, float]    # 0..1 torque available per group
    activation: Dict[str, float]        # last commanded activation per group

    def vector(self, order: List[str]) -> np.ndarray:
        return np.array([self.fatigue[g] for g in order], dtype=np.float64)


class MuscleSystem:
    """Tracks per-group activation and fatigue over time."""

    def __init__(self, params: BodyParams):
        self.p = params
        self.groups = list(MUSCLE_GROUPS)
        self.reset()

    def reset(self) -> None:
        self.fatigue = {g: 0.0 for g in self.groups}
        self.activation = {g: 0.0 for g in self.groups}

    # ------------------------------------------------------------------ #
    def peak_torque(self, group: str) -> float:
        joint = GROUP_TO_JOINT[group]
        base = self.p.muscle_peak_torque[joint]
        # Two groups share the knee (quad/hamstring) and split its budget.
        if joint == "knee":
            return base * 0.6
        if group in ("shoulder", "arm"):
            return base
        return base

    def available_frac(self, group: str) -> float:
        """Torque available now as a fraction of peak (drops with fatigue)."""
        # A simple but standard fatigue->force relation: force capacity falls
        # roughly linearly with accumulated fatigue but never below 25%.
        return float(np.clip(1.0 - 0.75 * self.fatigue[group], 0.25, 1.0))

    # ------------------------------------------------------------------ #
    def step(self, dt: float, activations: Dict[str, float]) -> MuscleState:
        """Update fatigue from commanded activations (each in 0..1)."""
        for g in self.groups:
            a = float(np.clip(activations.get(g, 0.0), 0.0, 1.0))
            self.activation[g] = a
            # Fatigue accrues proportional to activation above a maintenance
            # level; recovers when activation is low. Time constants come from
            # the body params (local muscular endurance).
            if a > 0.15:
                rate = a / self.p.muscle_fatigue_tau_s
                self.fatigue[g] += rate * (1.0 - self.fatigue[g]) * dt
            else:
                rate = 1.0 / self.p.muscle_recovery_tau_s
                self.fatigue[g] -= rate * self.fatigue[g] * dt
            self.fatigue[g] = float(np.clip(self.fatigue[g], 0.0, 1.0))
        return self.state()

    def muscular_power(self, joint_torques: Dict[str, float],
                       joint_velocities: Dict[str, float]) -> float:
        """Positive mechanical power the muscles are producing (W).

        Used by the energy system to add a metabolic surcharge for muscular
        work beyond pure locomotion cost.
        """
        power = 0.0
        for joint, tau in joint_torques.items():
            omega = joint_velocities.get(joint, 0.0)
            power += abs(tau * omega)
        return power

    def torque_limit_vector(self, joint_order: List[str]) -> np.ndarray:
        """Per-joint available torque limit (N*m) given current fatigue.

        For joints driven by multiple groups (knee), takes the summed budget
        weighted by each group's remaining capacity.
        """
        limits = []
        for joint in joint_order:
            groups = [g for g, j in GROUP_TO_JOINT.items() if j == joint]
            total = 0.0
            for g in groups:
                total += self.peak_torque(g) * self.available_frac(g)
            if not groups:
                total = 50.0
            limits.append(total)
        return np.array(limits, dtype=np.float64)

    def state(self) -> MuscleState:
        return MuscleState(
            fatigue=dict(self.fatigue),
            available_frac={g: self.available_frac(g) for g in self.groups},
            activation=dict(self.activation),
        )
