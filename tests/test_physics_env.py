"""Tests for the physics body, track, and the mile environment."""
import numpy as np
import pytest

from milerunner.biomech.params import BodyParams
from milerunner.physics.body_builder import (actuated_joint_names,
                                            build_humanoid_mjcf)
from milerunner.physics.track import MILE_M, Track, Weather

mujoco = pytest.importorskip("mujoco")
gym = pytest.importorskip("gymnasium")

from milerunner.envs.mile_env import EnvConfig, MileRunEnv  # noqa: E402
from milerunner.physics.humanoid import Humanoid  # noqa: E402


def test_model_total_mass_matches_body():
    p = BodyParams(mass_kg=77.0)
    m = mujoco.MjModel.from_xml_string(build_humanoid_mjcf(p))
    assert float(sum(m.body_mass)) == pytest.approx(77.0, abs=0.5)
    assert m.nu == len(actuated_joint_names())


def test_humanoid_stands_under_gravity():
    hum = Humanoid(BodyParams())
    hum.reset()
    z0 = hum.pelvis_height()
    for _ in range(500):
        hum.step(1)
    # No control -> should settle onto feet, not collapse.
    assert hum.pelvis_height() > 0.7
    assert abs(hum.pelvis_height() - z0) < 0.2


def test_weather_air_density_and_altitude():
    cool = Weather(temperature_c=10).air_density()
    hot = Weather(temperature_c=35).air_density()
    assert cool > hot  # cold air is denser
    assert Weather(altitude_m=2500).vo2max_altitude_factor() < 1.0
    assert Weather(altitude_m=0).vo2max_altitude_factor() == 1.0


def test_drag_opposes_and_scales_with_speed():
    tr = Track(Weather(wind_mps=0))
    assert tr.drag_force(5.0) > tr.drag_force(2.0) > 0
    head = Track(Weather(wind_mps=-4)).drag_force(5.0)
    tail = Track(Weather(wind_mps=4)).drag_force(5.0)
    assert head > tail  # headwind costs more


def test_env_reset_and_step_shapes():
    env = MileRunEnv(config=EnvConfig(max_time_s=5))
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, r, term, trunc, info = env.step(a)
    assert np.all(np.isfinite(obs))
    assert isinstance(r, float)
    for k in ("distance", "time", "speed", "heart_rate", "w_prime_frac"):
        assert k in info


def test_env_physics_rate_meets_requirement():
    env = MileRunEnv(config=EnvConfig(physics_timestep=0.001, control_hz=100))
    # 100 Hz control * 10 substeps = 1000 physics steps per simulated second.
    assert env.n_substeps * env.cfg.control_hz >= 1000


def test_env_determinism_same_seed():
    a = MileRunEnv(config=EnvConfig(max_time_s=3))
    b = MileRunEnv(config=EnvConfig(max_time_s=3))
    oa, _ = a.reset(seed=123)
    ob, _ = b.reset(seed=123)
    assert np.allclose(oa, ob)
    act = np.full(a.action_space.shape, 0.05, dtype=np.float32)
    for _ in range(20):
        oa, ra, ta, tra, _ = a.step(act)
        ob, rb, tb, trb, _ = b.step(act)
    assert np.allclose(oa, ob)
    assert ra == pytest.approx(rb)


def test_falling_terminates_and_penalizes():
    env = MileRunEnv(config=EnvConfig(max_time_s=30, terminate_on_fall=True))
    env.reset(seed=1)
    # large random torques -> should fall and terminate
    term = False
    rng = np.random.default_rng(0)
    for _ in range(400):
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, env.action_space.shape).astype(np.float32))
        if term:
            break
    assert term and info["fell"]
