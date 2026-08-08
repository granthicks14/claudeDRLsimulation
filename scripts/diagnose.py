#!/usr/bin/env python3
"""Startup diagnostic — verifies every stage the live dashboard depends on.

Run this if the dashboard is stuck on "Waiting for runner". It checks, in order:
runner/humanoid creation, MuJoCo init, a physics step, an RL env + one step,
producing telemetry, writing/reading the telemetry sidecar, and (optionally) the
OSMesa/EGL renderer. Each line prints OK / SKIP / FAIL with the real error, so
you can see exactly where it breaks.

    python scripts/diagnose.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK, FAIL, SKIP = "✅ OK", "❌ FAIL", "⚠️  SKIP"
_failed = 0


def check(name, fn, optional=False):
    global _failed
    try:
        detail = fn()
        print(f"{OK}  {name}" + (f" — {detail}" if detail else ""))
        return True
    except Exception as e:
        if optional:
            print(f"{SKIP}  {name} — {e}")
            return True
        print(f"{FAIL}  {name} — {type(e).__name__}: {e}")
        traceback.print_exc()
        _failed += 1
        return False


def main():
    config = os.environ.get("MILE_CONFIG", "hosted")
    print(f"MileRunner startup diagnostic  (config={config})")
    print(f"  MUJOCO_GL = {os.environ.get('MUJOCO_GL', '(unset — correct for the trainer)')}\n")

    from milerunner.config_build import build_body, build_env_config
    from milerunner.utils.config import load_config
    cfg = load_config(config)
    body = build_body(cfg)
    env_cfg = build_env_config(cfg)

    check("import mujoco", lambda: __import__("mujoco").__version__ and "imported")

    def build_humanoid():
        from milerunner.physics.humanoid import Humanoid
        h = Humanoid(body.scaled(), timestep=env_cfg.physics_timestep)
        return f"{body.name}, {sum(h.model.body_mass):.0f} kg, {h.n_actuators} actuators"
    if not check("build humanoid (MuJoCo model)", build_humanoid):
        print("\n>>> The trainer cannot start. Fix the humanoid/MuJoCo error above.")
        sys.exit(1)

    def physics_step():
        from milerunner.physics.humanoid import Humanoid
        h = Humanoid(body.scaled(), timestep=env_cfg.physics_timestep)
        h.reset(); h.step(200)
        return f"pelvis z = {h.pelvis_height():.2f} m after 200 steps"
    check("physics stepping", physics_step)

    holder = {}

    def build_env_step():
        import numpy as np
        from milerunner.training.env_builder import build_single_env
        env = build_single_env(body=body, config=env_cfg, seed=0)
        obs, _ = env.reset(seed=0)
        for _ in range(5):
            obs, r, term, trunc, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32))
        holder["env"] = env
        return f"obs dim {obs.shape[0]}, reward {r:+.3f}, distance {info['distance']:.2f} m"
    check("RL environment + step", build_env_step)

    def evaluate_telemetry():
        from stable_baselines3 import PPO
        from milerunner.training.evaluation import evaluate_policy
        env = holder["env"]
        model = PPO("MlpPolicy", env, n_steps=64, device="cpu")
        res = evaluate_policy(model, env=env, record_telemetry=True,
                              record_skeleton=True, seed=0)
        holder["tele"] = res.telemetry
        return (f"{res.steps} steps, {len(res.telemetry.get('speed', []))} telemetry pts, "
                f"{len(res.telemetry.get('skeleton', []))} skeleton frames")
    check("evaluate (produce telemetry)", evaluate_telemetry)

    def sidecar_roundtrip():
        tele = holder.get("tele", {})
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(tele, fh, default=str)
            path = fh.name
        loaded = json.load(open(path))
        os.unlink(path)
        return f"wrote+read {len(json.dumps(loaded, default=str))} bytes of telemetry"
    check("telemetry sidecar write/read (UI feed)", sidecar_roundtrip)

    def dashboard_build():
        from milerunner.dashboard import figures as F
        from milerunner.dashboard.track_view import track_figure
        tele = holder.get("tele", {})
        F.speed_curve(tele).to_dict()
        track_figure(tele, best_distance=tele.get("distance", [0])[-1]).to_dict()
        return "figures build from telemetry"
    check("dashboard figures from telemetry", dashboard_build)

    def renderer():
        os.environ.setdefault("MUJOCO_GL",
                              "egl" if os.path.exists("/proc/driver/nvidia/version") else "osmesa")
        from milerunner.dashboard.render3d import gl_available
        if not gl_available():
            raise RuntimeError(f"no GL backend (MUJOCO_GL={os.environ.get('MUJOCO_GL')}) "
                               "— video render disabled, everything else still works")
        return f"OSMesa/EGL renderer works (MUJOCO_GL={os.environ.get('MUJOCO_GL')})"
    check("3D video renderer (optional)", renderer, optional=True)

    print()
    if _failed:
        print(f">>> {_failed} check(s) FAILED — the dashboard will show this error. Fix the above.")
        sys.exit(1)
    print(">>> All required checks passed. The runner works; the dashboard will "
          "populate once generation 0 finishes (~2–3 min on a free CPU).")


if __name__ == "__main__":
    main()
