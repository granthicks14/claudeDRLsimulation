"""Observation assembly.

Collects every quantity the brief requires an agent to observe (speed,
acceleration, heart rate, oxygen, muscle fatigue, joint angles, stride length,
cadence, distance remaining, balance, energy reserves, ground forces) plus the
proprioceptive joint velocities an RL policy needs to actually control the body.

The layout is exposed as :data:`OBS_LABELS` so the dashboard and analysis tools
can label each dimension.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..biomech.muscles import MUSCLE_GROUPS


@dataclass
class GaitStats:
    speed: float = 0.0
    acceleration: float = 0.0
    stride_length: float = 0.0
    cadence_hz: float = 0.0
    distance_covered: float = 0.0
    distance_remaining: float = 0.0
    time_elapsed: float = 0.0


def observation_labels(n_joints: int) -> List[str]:
    labels = [
        "speed", "acceleration", "heart_rate", "vo2_frac", "vo2_demand_frac",
    ]
    labels += [f"fatigue_{g}" for g in MUSCLE_GROUPS]
    labels += [f"joint_angle_{i}" for i in range(n_joints)]
    labels += [f"joint_vel_{i}" for i in range(n_joints)]
    labels += [
        "stride_length", "cadence", "distance_remaining_frac", "distance_covered_frac",
        "uprightness", "lean", "ang_vel_mag",
        "w_prime_frac", "glycogen_frac",
        "grf_right", "grf_left",
        "lactate", "core_temp", "contact_right", "contact_left",
        "time_frac", "breathing",
    ]
    return labels


def build_observation(humanoid, energy_state, muscle_state, gait: GaitStats,
                      total_distance: float, max_time: float, breathing: float,
                      body_mass: float, vo2max: float) -> np.ndarray:
    p = humanoid.params
    angles = humanoid.joint_angles()
    vels = humanoid.joint_velocities()
    grf = humanoid.ground_reaction_forces()
    contacts = humanoid.foot_contacts()
    bodyweight = body_mass * 9.81

    obs = [
        gait.speed,
        np.clip(gait.acceleration, -20, 20),
        (energy_state.heart_rate_bpm - 60.0) / 140.0,
        np.clip(energy_state.vo2_ml_kg_min / max(vo2max, 1e-6), 0.0, 1.3),
        np.clip(energy_state.vo2_demand_ml_kg_min / max(vo2max, 1e-6), 0.0, 2.5),
    ]
    obs += [muscle_state.fatigue[g] for g in MUSCLE_GROUPS]
    obs += list(angles)
    obs += list(np.clip(vels, -30, 30))
    obs += [
        gait.stride_length,
        gait.cadence_hz,
        np.clip(gait.distance_remaining / max(total_distance, 1e-6), 0.0, 1.0),
        np.clip(gait.distance_covered / max(total_distance, 1e-6), 0.0, 1.0),
        humanoid.orientation_uprightness(),
        humanoid.lean_angle(),
        float(np.linalg.norm(humanoid.data.qvel[3:6])),
        energy_state.w_prime_frac,
        energy_state.glycogen_frac,
        np.clip(grf["right"] / max(bodyweight, 1.0), 0.0, 4.0),
        np.clip(grf["left"] / max(bodyweight, 1.0), 0.0, 4.0),
        np.clip(energy_state.blood_lactate_mmol_l / 20.0, 0.0, 1.0),
        np.clip((energy_state.core_temp_c - 37.0) / 3.0, 0.0, 1.0),
        float(contacts["right"]),
        float(contacts["left"]),
        np.clip(gait.time_elapsed / max(max_time, 1e-6), 0.0, 1.0),
        breathing,
    ]
    return np.asarray(obs, dtype=np.float32)


def observation_dim(n_joints: int) -> int:
    return len(observation_labels(n_joints))
