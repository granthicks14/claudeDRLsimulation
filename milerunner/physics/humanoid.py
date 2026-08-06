"""Thin wrapper around a compiled MuJoCo humanoid model.

Owns the ``MjModel``/``MjData`` pair and exposes the biomechanically relevant
quantities the environment needs: joint angles/velocities, foot ground-reaction
forces, centre-of-mass velocity, balance/orientation and per-actuator torque
clipping driven by muscle fatigue.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    import mujoco
except Exception:  # pragma: no cover - allow import without mujoco present
    mujoco = None

from ..biomech.params import BodyParams
from .body_builder import (actuated_joint_names, build_humanoid_mjcf,
                           joint_muscle_map)


class Humanoid:
    """A single simulated runner body."""

    def __init__(self, params: BodyParams, friction: float = 1.0,
                 timestep: float = 0.001):
        if mujoco is None:  # pragma: no cover
            raise RuntimeError("mujoco is required to instantiate Humanoid")
        self.params = params
        self.timestep = timestep
        self.friction = friction
        self.xml = build_humanoid_mjcf(params, friction=friction, timestep=timestep)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.joint_names = actuated_joint_names()
        self.joint_muscle = joint_muscle_map()
        self._act_ids = [self.model.actuator(f"act_{n}").id for n in self.joint_names]
        self._joint_ids = [self.model.joint(n).id for n in self.joint_names]
        self._qpos_adr = [self.model.jnt_qposadr[j] for j in self._joint_ids]
        self._qvel_adr = [self.model.jnt_dofadr[j] for j in self._joint_ids]
        self._gears = np.array([self.model.actuator_gear[i, 0] for i in self._act_ids])
        self._foot_geoms = {
            "right": self.model.geom("foot_r_g").id,
            "left": self.model.geom("foot_l_g").id,
        }
        self._floor_geom = self.model.geom("floor").id
        self.n_actuators = len(self._act_ids)

    # ------------------------------------------------------------------ #
    def reset(self, qpos_noise: float = 0.0, rng: Optional[np.random.Generator] = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        if qpos_noise > 0 and rng is not None:
            self.data.qpos[7:] += rng.uniform(-qpos_noise, qpos_noise, size=self.model.nq - 7)
            self.data.qvel[:] += rng.uniform(-qpos_noise, qpos_noise, size=self.model.nv) * 0.1
        mujoco.mj_forward(self.model, self.data)

    def set_ctrl(self, action: np.ndarray, torque_scale: Optional[np.ndarray] = None) -> None:
        """Apply an action in [-1, 1] per actuator.

        ``torque_scale`` (0..1 per actuator) comes from the muscle fatigue
        model and shrinks the usable control authority so torques can never
        exceed the *current* human strength of that group.
        """
        a = np.clip(action, -1.0, 1.0)
        if torque_scale is not None:
            a = a * np.clip(torque_scale, 0.0, 1.0)
        self.data.ctrl[self._act_ids] = a

    def step(self, n_substeps: int = 1) -> None:
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------ #
    def joint_angles(self) -> np.ndarray:
        return self.data.qpos[self._qpos_adr].copy()

    def joint_velocities(self) -> np.ndarray:
        return self.data.qvel[self._qvel_adr].copy()

    def joint_torques(self) -> np.ndarray:
        """Actual actuator torque (N*m) currently applied per joint."""
        return (self.data.ctrl[self._act_ids] * self._gears).copy()

    def com_position(self) -> np.ndarray:
        return self.data.body("pelvis").xpos.copy()

    def com_velocity(self) -> np.ndarray:
        # subtree linear velocity of the whole body at the pelvis
        return self.data.qvel[0:3].copy()

    def forward_speed(self) -> float:
        return float(self.data.qvel[0])

    def torso_height(self) -> float:
        return float(self.data.body("torso").xpos[2])

    def pelvis_height(self) -> float:
        return float(self.data.qpos[2])

    def orientation_uprightness(self) -> float:
        """Cosine of tilt from vertical of the pelvis z-axis (1 = upright)."""
        # rotation matrix of pelvis
        xmat = self.data.body("pelvis").xmat.reshape(3, 3)
        up = xmat[:, 2]
        return float(up[2])

    def lean_angle(self) -> float:
        """Forward body-lean angle (radians); positive = leaning forward."""
        xmat = self.data.body("torso").xmat.reshape(3, 3)
        forward = xmat[:, 0]
        return float(np.arctan2(forward[0] - 0.0, forward[2] + 1e-8))

    def ground_reaction_forces(self) -> Dict[str, float]:
        """Vertical GRF (N) under each foot, summed over active contacts."""
        grf = {"right": 0.0, "left": 0.0}
        forces = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            for side, fg in self._foot_geoms.items():
                if (g1 == fg and g2 == self._floor_geom) or (g2 == fg and g1 == self._floor_geom):
                    mujoco.mj_contactForce(self.model, self.data, i, forces)
                    grf[side] += abs(float(forces[0]))  # normal component
        return grf

    def foot_contacts(self) -> Dict[str, bool]:
        grf = self.ground_reaction_forces()
        return {k: v > 1.0 for k, v in grf.items()}

    def has_fallen(self, min_pelvis_height: float = 0.55) -> bool:
        return self.pelvis_height() < min_pelvis_height or self.orientation_uprightness() < 0.3

    def joint_limit_violation(self) -> float:
        """Fraction of actuated joints currently pressed against their limit."""
        n_viol = 0
        for jid, adr in zip(self._joint_ids, self._qpos_adr):
            lo, hi = self.model.jnt_range[jid]
            q = self.data.qpos[adr]
            span = hi - lo
            if span <= 0:
                continue
            if q <= lo + 0.02 * span or q >= hi - 0.02 * span:
                n_viol += 1
        return n_viol / max(len(self._joint_ids), 1)

    def muscular_power(self) -> float:
        """Instantaneous positive mechanical power across all actuators (W)."""
        tau = self.joint_torques()
        omega = self.joint_velocities()
        return float(np.sum(np.abs(tau * omega)))

    def state_vectors(self) -> Dict[str, np.ndarray]:
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
        }

    def load_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
