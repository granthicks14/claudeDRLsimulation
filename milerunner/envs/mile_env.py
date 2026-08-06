"""The mile-run Gymnasium environment.

Couples the MuJoCo humanoid physics with the biomechanical energy and muscle
systems and the atmospheric track model. An agent applies joint torques (its
"muscle activations") plus a breathing command; the environment enforces human
strength limits (via fatigue-scaled torque clipping), human joint ranges (via
the model), and human energy limits (via the W'-balance / VO2max model), then
returns the full physiological + kinematic observation and a reward.

Nothing here encodes *how* to run. Stride pattern, cadence, pacing, arm swing,
lean and breathing rhythm are all emergent from the agent's control policy
under these physical constraints — exactly the discovery the project targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None
    spaces = None

from ..biomech.energy import EnergySystem
from ..biomech.muscles import GROUP_TO_JOINT, MUSCLE_GROUPS, MuscleSystem
from ..biomech.params import BodyParams
from ..physics.humanoid import Humanoid
from ..physics.track import MILE_M, Track, Weather
from .observations import (GaitStats, build_observation, observation_dim,
                          observation_labels)
from .rewards import RewardWeights, compute_reward


@dataclass
class EnvConfig:
    distance_m: float = MILE_M
    control_hz: float = 100.0
    physics_timestep: float = 0.001     # 1000 Hz physics -> 1000+ steps/sec
    max_time_s: float = 900.0           # generous cap; a mile even slow < 15 min
    terminate_on_fall: bool = True
    terminate_on_exhaustion: bool = False
    friction: float = 1.0
    reset_noise: float = 0.02
    randomize_weather: bool = False
    randomize_body: bool = False
    domain_randomization: bool = False


class MileRunEnv(gym.Env if gym is not None else object):
    """A single-runner mile environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(self, body: Optional[BodyParams] = None,
                 weather: Optional[Weather] = None,
                 reward_weights: Optional[RewardWeights] = None,
                 config: Optional[EnvConfig] = None,
                 seed: Optional[int] = None):
        self.cfg = config or EnvConfig()
        self.body = (body or BodyParams()).scaled()
        self.weather = weather or Weather()
        self.reward_weights = reward_weights or RewardWeights()
        self.rng = np.random.default_rng(seed)

        self.humanoid = Humanoid(self.body, friction=self.cfg.friction,
                                 timestep=self.cfg.physics_timestep)
        self.energy = EnergySystem(self.body)
        self.muscles = MuscleSystem(self.body)
        self.track = Track(self.weather, distance_m=self.cfg.distance_m)

        self.n_substeps = max(1, int(round((1.0 / self.cfg.control_hz) / self.cfg.physics_timestep)))
        self.dt = self.n_substeps * self.cfg.physics_timestep
        self.n_act = self.humanoid.n_actuators
        self.n_joints = self.n_act

        obs_dim = observation_dim(self.n_joints)
        if spaces is not None:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
            # actions: one torque per actuator + 1 breathing channel
            self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_act + 1,), dtype=np.float32)
        self.obs_labels = observation_labels(self.n_joints)

        self._reset_state()

    # ------------------------------------------------------------------ #
    def _reset_state(self) -> None:
        self.time_s = 0.0
        self.distance = 0.0
        self.start_x = 0.0
        self.prev_speed = 0.0
        self.prev_x = 0.0
        self.finished = False
        self.finish_time = None
        self.last_contacts = {"right": False, "left": False}
        self.last_strike_time = {"right": 0.0, "left": 0.0}
        self.last_strike_x = {"right": 0.0, "left": 0.0}
        self.cadence_hz = 0.0
        self.stride_length = 0.0
        self._recent_strides = []
        self._recent_cadence = []
        self.telemetry: Dict[str, list] = {}
        self.episode_reward = 0.0
        self.peak_speed = 0.0

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.cfg.randomize_weather:
            self.weather = Weather(
                temperature_c=float(self.rng.uniform(5, 30)),
                wind_mps=float(self.rng.uniform(-3, 3)),
                humidity=float(self.rng.uniform(0.2, 0.9)),
                altitude_m=float(self.rng.choice([0, 0, 0, 1000, 2000])),
            )
            self.track = Track(self.weather, distance_m=self.cfg.distance_m)
        self.energy.reset()
        self.muscles.reset()
        self.humanoid.reset(qpos_noise=self.cfg.reset_noise, rng=self.rng)
        self._reset_state()
        self.prev_x = float(self.humanoid.data.qpos[0])
        self.start_x = self.prev_x
        obs = self._build_obs()
        return obs, self._info(np.zeros(self.n_act + 1))

    # ------------------------------------------------------------------ #
    def _decode_action(self, action: np.ndarray) -> Tuple[np.ndarray, float]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        torques = action[: self.n_act]
        breathing = float((np.tanh(action[self.n_act]) + 1.0) * 0.5) if action.shape[0] > self.n_act else 0.5
        return torques, breathing

    def _muscle_torque_scale(self) -> np.ndarray:
        """Per-actuator torque limit fraction from current muscle fatigue."""
        scale = np.ones(self.n_act)
        for i, jname in enumerate(self.humanoid.joint_names):
            joint_key = self.humanoid.joint_muscle[jname]
            groups = [g for g, j in GROUP_TO_JOINT.items() if j == joint_key]
            if groups:
                scale[i] = float(np.mean([self.muscles.available_frac(g) for g in groups]))
        return scale

    def _activations_from_torque(self, torques: np.ndarray) -> Dict[str, float]:
        """Map applied joint torques back to per-muscle-group activation (0..1)."""
        acts = {g: 0.0 for g in MUSCLE_GROUPS}
        counts = {g: 0 for g in MUSCLE_GROUPS}
        for i, jname in enumerate(self.humanoid.joint_names):
            joint_key = self.humanoid.joint_muscle[jname]
            for g in MUSCLE_GROUPS:
                if GROUP_TO_JOINT[g] == joint_key:
                    acts[g] += abs(float(np.clip(torques[i], -1, 1)))
                    counts[g] += 1
        for g in MUSCLE_GROUPS:
            if counts[g] > 0:
                acts[g] /= counts[g]
        return acts

    # ------------------------------------------------------------------ #
    def step(self, action: np.ndarray):
        torques, breathing = self._decode_action(action)
        torque_scale = self._muscle_torque_scale()

        self.humanoid.set_ctrl(torques, torque_scale=torque_scale)
        self.humanoid.step(self.n_substeps)
        self.time_s += self.dt

        x = float(self.humanoid.data.qpos[0])
        dx = x - self.prev_x
        speed = dx / self.dt
        accel = (speed - self.prev_speed) / self.dt
        self.distance = x - self.start_x
        self.peak_speed = max(self.peak_speed, speed)

        # --- gait bookkeeping: cadence + stride length from foot strikes ---
        self._update_gait(x)

        # --- muscle fatigue update ---
        activations = self._activations_from_torque(torques)
        self.muscles.step(self.dt, activations)

        # --- physiology update ---
        curve = self.track.curve_cost_factor(self.distance)
        muscular_power = self.humanoid.muscular_power()
        drag_power = self.track.drag_power(max(speed, 0.0))
        total_extra_power = muscular_power * (curve - 1.0) + drag_power
        vo2max_factor = self.weather.vo2max_altitude_factor()
        estate = self.energy.step(
            self.dt, speed_mps=max(speed, 0.0),
            muscular_power_w=total_extra_power,
            cadence_hz=self.cadence_hz if self.cadence_hz > 0 else self.body.tendon_natural_cadence_hz,
            ambient_temp_c=self.weather.temperature_c,
            breathing=breathing, vo2max_factor=vo2max_factor,
            humidity=self.weather.humidity,
        )
        mstate = self.muscles.state()

        # --- termination checks ---
        fell = self.humanoid.has_fallen()
        joint_viol = self.humanoid.joint_limit_violation()
        finished_now = (not self.finished) and (self.distance >= self.cfg.distance_m)
        if finished_now:
            self.finished = True
            self.finish_time = self.time_s

        overexert = max(estate.vo2_demand_ml_kg_min - self.body.vo2max_ml_kg_min, 0.0) / \
            max(self.body.vo2max_ml_kg_min, 1e-6)
        lateral_speed = float(self.humanoid.data.qvel[1])
        action_sq = float(np.mean(np.square(np.clip(torques, -1, 1))))

        rb = compute_reward(
            self.reward_weights,
            dprogress=dx, speed=speed, uprightness=self.humanoid.orientation_uprightness(),
            lateral_speed=lateral_speed, metabolic_power=estate.metabolic_power_w,
            mass=self.body.mass_kg, overexertion=overexert, joint_violation=joint_viol,
            action_sq=action_sq, fell=fell, finished=finished_now,
            finish_time=self.finish_time or self.time_s, dt=self.dt,
            exhausted=estate.exhausted,
        )
        reward = rb.total
        self.episode_reward += reward

        terminated = False
        truncated = False
        if fell and self.cfg.terminate_on_fall:
            terminated = True
        if estate.exhausted and self.cfg.terminate_on_exhaustion:
            terminated = True
        if self.finished:
            terminated = True
        if self.time_s >= self.cfg.max_time_s:
            truncated = True

        self.prev_speed = speed
        self.prev_x = x

        obs = self._build_obs(estate, mstate, speed, accel)
        info = self._info(action, rb=rb, estate=estate, mstate=mstate,
                          speed=speed, fell=fell)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _update_gait(self, x: float) -> None:
        contacts = self.humanoid.foot_contacts()
        for side in ("right", "left"):
            if contacts[side] and not self.last_contacts[side]:
                # new foot strike
                dt_stride = self.time_s - self.last_strike_time[side]
                dx_stride = x - self.last_strike_x[side]
                if dt_stride > 0.05:
                    self._recent_cadence.append(1.0 / dt_stride)
                    self._recent_strides.append(abs(dx_stride))
                    self._recent_cadence = self._recent_cadence[-6:]
                    self._recent_strides = self._recent_strides[-6:]
                self.last_strike_time[side] = self.time_s
                self.last_strike_x[side] = x
        self.last_contacts = contacts
        if self._recent_cadence:
            # steps per second across both feet ~= 2 * per-foot stride rate
            self.cadence_hz = float(np.mean(self._recent_cadence)) * 2.0 / 2.0
        if self._recent_strides:
            self.stride_length = float(np.mean(self._recent_strides))

    def _gait_stats(self, speed: float = 0.0, accel: float = 0.0) -> GaitStats:
        return GaitStats(
            speed=speed, acceleration=accel,
            stride_length=self.stride_length, cadence_hz=self.cadence_hz,
            distance_covered=self.distance,
            distance_remaining=max(self.cfg.distance_m - self.distance, 0.0),
            time_elapsed=self.time_s,
        )

    def _build_obs(self, estate=None, mstate=None, speed: float = 0.0, accel: float = 0.0) -> np.ndarray:
        estate = estate or self.energy.state()
        mstate = mstate or self.muscles.state()
        gait = self._gait_stats(speed, accel)
        breathing = 0.5
        return build_observation(
            self.humanoid, estate, mstate, gait,
            total_distance=self.cfg.distance_m, max_time=self.cfg.max_time_s,
            breathing=breathing, body_mass=self.body.mass_kg,
            vo2max=self.body.vo2max_ml_kg_min,
        )

    def _info(self, action, rb=None, estate=None, mstate=None,
              speed: float = 0.0, fell: bool = False) -> Dict[str, Any]:
        estate = estate or self.energy.state()
        info = {
            "distance": self.distance,
            "time": self.time_s,
            "speed": speed,
            "peak_speed": self.peak_speed,
            "cadence_hz": self.cadence_hz,
            "stride_length": self.stride_length,
            "heart_rate": estate.heart_rate_bpm,
            "vo2": estate.vo2_ml_kg_min,
            "w_prime_frac": estate.w_prime_frac,
            "lactate": estate.blood_lactate_mmol_l,
            "glycogen": estate.glycogen_frac,
            "core_temp": estate.core_temp_c,
            "metabolic_power": estate.metabolic_power_w,
            "exhausted": estate.exhausted,
            "finished": self.finished,
            "finish_time": self.finish_time,
            "fell": fell,
            "episode_reward": self.episode_reward,
        }
        if rb is not None:
            info["reward_terms"] = rb.terms
        if mstate is not None:
            info["muscle_fatigue"] = dict(mstate.fatigue)
        return info

    def render(self):  # pragma: no cover - offscreen rendering optional
        return None

    def close(self):  # pragma: no cover
        pass
