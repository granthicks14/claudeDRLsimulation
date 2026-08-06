"""Evaluate a trained policy on the mile and compute fitness + telemetry.

Fitness is defined so that *finishing the mile fast* dominates, with partial
credit for distance covered when the agent cannot yet finish (essential early
in training when nobody completes a mile). The returned telemetry time-series
drives the dashboard's best-agent replay and the research analyses (speed
curve, cadence, HR, fatigue, energy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..envs.mile_env import MileRunEnv
from ..physics.track import MILE_M


@dataclass
class EvalResult:
    fitness: float
    finished: bool
    mile_time: Optional[float]
    distance: float
    mean_speed: float
    peak_speed: float
    mean_cadence: float
    mean_hr: float
    mean_stride: float
    total_reward: float
    steps: int
    telemetry: Dict[str, List[float]] = field(default_factory=dict)

    def to_metrics(self) -> Dict[str, Any]:
        return {
            "fitness": self.fitness,
            "finished": self.finished,
            "mile_time": self.mile_time,
            "distance": self.distance,
            "mean_speed": self.mean_speed,
            "peak_speed": self.peak_speed,
            "mean_cadence": self.mean_cadence,
            "mean_hr": self.mean_hr,
            "metrics": {
                "mean_stride": self.mean_stride,
                "total_reward": self.total_reward,
                "steps": self.steps,
            },
        }


def _predict(model, obs, state=None, deterministic=True):
    """Uniform predict for feedforward and recurrent SB3 policies."""
    try:
        return model.predict(obs, state=state, deterministic=deterministic)
    except TypeError:  # pragma: no cover
        action, _ = model.predict(obs, deterministic=deterministic)
        return action, None


def _downsample(tele: Dict[str, List[float]], max_points: int = 1500) -> Dict[str, List[float]]:
    """Uniformly thin telemetry lists so stored records stay small."""
    n = len(tele.get("t", []))
    if n <= max_points:
        return tele
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return {k: [v[i] for i in idx] if isinstance(v, list) and len(v) == n else v
            for k, v in tele.items()}


def evaluate_policy(model, env: Optional[MileRunEnv] = None, *,
                    reward_weights=None, body=None, weather=None, config=None,
                    record_telemetry: bool = True, record_skeleton: bool = False,
                    skeleton_frames: int = 80, seed: int = 12345) -> EvalResult:
    """Run one deterministic episode and summarise it.

    ``record_skeleton`` additionally captures sub-sampled 3D body positions for
    the dashboard's replay. Per-muscle-group fatigue timelines are captured too
    so the fatigue heat-map has data.
    """
    from ..biomech.muscles import MUSCLE_GROUPS
    own_env = env is None
    if own_env:
        env = MileRunEnv(body=body, weather=weather, reward_weights=reward_weights,
                         config=config, seed=seed)
    base_env = getattr(env, "unwrapped", env)
    obs, info = env.reset(seed=seed)
    done = False
    state = None
    speeds: List[float] = []
    hrs: List[float] = []
    cads: List[float] = []
    strides: List[float] = []
    keys = ["t", "distance", "speed", "cadence", "heart_rate", "vo2",
            "w_prime_frac", "lactate", "glycogen", "core_temp", "metabolic_power"]
    keys += [f"fatigue_{g}" for g in MUSCLE_GROUPS]
    tele: Dict[str, List[float]] = {k: [] for k in keys}
    skeleton: List[Dict[str, List[float]]] = []
    skel_t: List[float] = []
    total_reward = 0.0
    steps = 0
    while not done:
        action, state = _predict(model, obs, state=state, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        speeds.append(info["speed"])
        hrs.append(info["heart_rate"])
        cads.append(info["cadence_hz"])
        strides.append(info["stride_length"])
        if record_telemetry:
            tele["t"].append(info["time"])
            tele["distance"].append(info["distance"])
            tele["speed"].append(info["speed"])
            tele["cadence"].append(info["cadence_hz"])
            tele["heart_rate"].append(info["heart_rate"])
            tele["vo2"].append(info["vo2"])
            tele["w_prime_frac"].append(info["w_prime_frac"])
            tele["lactate"].append(info["lactate"])
            tele["glycogen"].append(info["glycogen"])
            tele["core_temp"].append(info["core_temp"])
            tele["metabolic_power"].append(info["metabolic_power"])
            fat = info.get("muscle_fatigue", {})
            for g in MUSCLE_GROUPS:
                tele[f"fatigue_{g}"].append(float(fat.get(g, 0.0)))
        if record_skeleton and (steps % 4 == 0):
            from ..dashboard.replay import BODIES
            skeleton.append({b: base_env.humanoid.data.body(b).xpos.tolist() for b in BODIES})
            skel_t.append(info["time"])

    if record_telemetry:
        tele = _downsample(tele)
    if record_skeleton and skeleton:
        # thin skeleton to the requested frame budget
        n = len(skeleton)
        idx = np.linspace(0, n - 1, min(skeleton_frames, n)).astype(int)
        tele["skeleton"] = [skeleton[i] for i in idx]
        tele["skeleton_t"] = [skel_t[i] for i in idx]

    finished = bool(info.get("finished"))
    mile_time = info.get("finish_time") if finished else None
    distance = float(info.get("distance", 0.0))

    # ----- fitness -----
    if finished and mile_time:
        # Reward faster finishes strongly; ~ inverse time on top of a base.
        fitness = 1000.0 + 3000.0 / max(mile_time, 60.0)
    else:
        # Partial credit for distance; encourages steady progress before anyone
        # can complete a full mile.
        fitness = 200.0 * (distance / MILE_M) + 0.1 * total_reward
    result = EvalResult(
        fitness=float(fitness), finished=finished, mile_time=mile_time,
        distance=distance,
        mean_speed=float(np.mean(speeds)) if speeds else 0.0,
        peak_speed=float(np.max(speeds)) if speeds else 0.0,
        mean_cadence=float(np.mean([c for c in cads if c > 0])) if any(c > 0 for c in cads) else 0.0,
        mean_hr=float(np.mean(hrs)) if hrs else 0.0,
        mean_stride=float(np.mean([s for s in strides if s > 0])) if any(s > 0 for s in strides) else 0.0,
        total_reward=float(total_reward), steps=steps,
        telemetry=tele if record_telemetry else {},
    )
    if own_env:
        env.close()
    return result
