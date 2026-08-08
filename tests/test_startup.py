"""Startup diagnostic tests — the chain the live dashboard depends on.

These guard against the 'Waiting for runner' class of bug: the runner must be
able to build, step physics, run the RL env, produce telemetry, and the app must
surface trainer crashes instead of hanging.
"""
import os

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
mujoco = pytest.importorskip("mujoco")

from milerunner.biomech.params import BodyParams
from milerunner.config_build import build_body, build_env_config
from milerunner.physics.humanoid import Humanoid
from milerunner.training.env_builder import build_single_env
from milerunner.utils.config import load_config


def test_humanoid_builds_and_steps_without_gl():
    # The trainer needs NO OpenGL; ensure it builds/steps regardless of MUJOCO_GL.
    h = Humanoid(BodyParams())
    h.reset()
    h.step(100)
    assert h.pelvis_height() > 0.5


def test_env_reset_step_produces_state():
    env = build_single_env(config=build_env_config(load_config("hosted")), seed=0)
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        obs, r, term, trunc, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert np.all(np.isfinite(obs))
    assert "distance" in info


def test_evaluate_produces_telemetry_for_dashboard():
    from stable_baselines3 import PPO
    from milerunner.training.evaluation import evaluate_policy
    env = build_single_env(config=build_env_config(load_config("hosted")), seed=0)
    model = PPO("MlpPolicy", env, n_steps=64, device="cpu")
    res = evaluate_policy(model, env=env, record_telemetry=True,
                          record_skeleton=True, seed=0)
    assert res.steps > 0
    assert len(res.telemetry.get("speed", [])) > 0
    assert len(res.telemetry.get("skeleton", [])) > 0     # feeds the runner views


def test_app_surfaces_trainer_error_instead_of_hanging(monkeypatch):
    os.environ["MILE_NO_AUTOSTART"] = "1"      # don't spawn a real trainer on import
    import app
    # Simulate a trainer that crashed and exhausted its restarts.
    app._trainer_proc = None
    app._trainer_error = "Traceback: RuntimeError: mujoco is required to instantiate Humanoid"
    app._trainer_restarts = app._MAX_RESTARTS + 1
    msg = app._startup_message({}, exp_id=1)
    assert "error" in msg.lower()
    assert "mujoco is required" in msg           # the real error is shown to the user


def test_app_startup_message_is_friendly_while_running(monkeypatch):
    os.environ["MILE_NO_AUTOSTART"] = "1"
    import app
    app._trainer_error = None
    app._trainer_restarts = 0

    class _Alive:
        def poll(self):
            return None
    app._trainer_proc = _Alive()
    msg = app._startup_message({"generation": 0, "total_timesteps": 500}, exp_id=1)
    assert "starting" in msg.lower() and "running" in msg.lower()
