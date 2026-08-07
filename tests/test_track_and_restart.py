"""Tests for the track-view mapping, auto-restart, and the elite-miler build."""
import math

import numpy as np
import pytest

from milerunner.dashboard.track_view import LAP, track_position


def test_track_lap_length_is_400m():
    assert LAP == pytest.approx(400.0, abs=1.0)


def test_track_start_and_wraparound():
    x0, y0 = track_position(0.0)
    xlap, ylap = track_position(LAP)
    # a full lap returns to the start
    assert (x0, y0) == pytest.approx((xlap, ylap))
    # start is on the (bottom) home straight
    assert y0 < 0


def test_track_mile_is_about_four_laps():
    mile = 1609.344
    assert mile / LAP == pytest.approx(4.02, abs=0.05)
    # position stays within the oval's bounding box at all distances
    for d in np.linspace(0, mile, 50):
        x, y = track_position(float(d))
        assert abs(x) < 130 and abs(y) < 60


def test_track_figure_builds_from_telemetry():
    fig = track_position  # ensure import path stable
    from milerunner.dashboard.track_view import track_figure
    tele = {"distance": list(np.linspace(0, 800, 200)),
            "t": list(np.linspace(0, 160, 200))}
    d = track_figure(tele).to_dict()
    assert len(d["data"]) >= 3          # outline + runner
    assert len(d.get("frames", [])) > 0  # animated


gym = pytest.importorskip("gymnasium")
mujoco = pytest.importorskip("mujoco")
from milerunner.envs.mile_env import EnvConfig, MileRunEnv  # noqa: E402


def test_stall_timeout_triggers_restart():
    env = MileRunEnv(config=EnvConfig(max_time_s=60, stall_timeout_s=2.0,
                                      terminate_on_fall=False))
    env.reset(seed=0)
    truncated = False
    info = {}
    for _ in range(2000):
        obs, r, term, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32))
        if term or truncated:
            break
    assert truncated
    assert info["restarted_reason"] == "stalled"
    assert info["time"] <= 3.0


def test_pace_deadline_triggers_restart():
    env = MileRunEnv(config=EnvConfig(max_time_s=60, pace_deadline_s=1.0,
                                      pace_deadline_m=500.0, terminate_on_fall=False,
                                      stall_timeout_s=0.0))
    env.reset(seed=1)
    truncated = False
    info = {}
    for _ in range(3000):
        obs, r, term, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32))
        if term or truncated:
            break
    # A stationary runner is far short of 500 m at the 1 s deadline -> restart.
    assert truncated and info["restarted_reason"] == "behind_pace"


def test_restart_disabled_by_default():
    cfg = EnvConfig()
    assert cfg.stall_timeout_s == 0.0 and cfg.pace_deadline_s == 0.0


def test_elite_miler_build_is_consistent():
    from milerunner.biomech.energy import EnergySystem
    from milerunner.config_build import build_body
    from milerunner.utils.config import load_config

    body = build_body(load_config("hosted"))
    assert body.name == "elite_miler"
    es = EnergySystem(body.scaled())
    # critical speed must remain below VO2max pace (physiologically valid)
    assert es.effective_critical_speed() <= es.vvo2max_speed()
    # and it should be a genuinely fast build (sub-4:30 theoretical floor)
    floor = (1609.344 - body.d_prime_m) / es.effective_critical_speed()
    assert floor < 270.0
