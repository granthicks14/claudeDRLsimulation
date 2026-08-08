#!/usr/bin/env python3
"""Launch continuous, autonomous mile-running training.

Training starts automatically and runs until you stop it (Ctrl-C) or a
generation cap is reached. Progress, checkpoints and the best models are saved
continuously, so you can stop and re-run this script to resume exactly where you
left off — the longer it runs, the faster the discovered mile becomes.

Examples
--------
    python scripts/run.py                       # default config, run forever
    python scripts/run.py --config smoke        # quick demo (3 short gens)
    python scripts/run.py --config cluster      # large-scale server config
    python scripts/run.py --experiment hot --set weather.temperature_c=32
    python scripts/run.py --generations 50      # stop after 50 generations
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from milerunner.config_build import build_all
from milerunner.training.trainer import ContinuousTrainer
from milerunner.utils.config import load_config
from milerunner.utils.logging import configure_logging, get_logger
from milerunner.utils.seeding import seed_everything


def parse_set(pairs):
    overrides = {}
    for p in pairs or []:
        if "=" not in p:
            continue
        key, val = p.split("=", 1)
        node = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        # best-effort typing
        try:
            if val.lower() in ("true", "false"):
                val = val.lower() == "true"
            elif "." in val:
                val = float(val)
            else:
                val = int(val)
        except ValueError:
            pass
        node[parts[-1]] = val
    return overrides


def main():
    ap = argparse.ArgumentParser(description="Continuous mile-running RL training")
    ap.add_argument("--config", default="default", help="config name or path")
    ap.add_argument("--experiment", default=None, help="experiment name (overrides config)")
    ap.add_argument("--generations", type=int, default=None,
                    help="stop after N generations (default: run forever)")
    ap.add_argument("--set", nargs="*", dest="overrides",
                    help="override config values, e.g. population.size=64")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    configure_logging(args.log_level, logfile="experiments/train.log")
    log = get_logger("milerunner.run")

    log.info("[1] Starting MileRunner trainer")
    cfg = load_config(args.config, overrides=parse_set(args.overrides))
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    log.info("[2] Loaded config '%s'", args.config)

    from milerunner.config_build import build_body
    tcfg, pcfg, ecfg = build_all(cfg, experiment_name=args.experiment)
    body = build_body(cfg)
    log.info("[3] Body: %s (%.0f kg) | population=%d | algos=%s | device=%s | n_envs=%d",
             body.name, body.mass_kg, pcfg.size, pcfg.algos, pcfg.device, pcfg.n_envs)

    # [4] Fail fast (with a clear message) if MuJoCo / the humanoid can't build —
    # otherwise the crash would happen deep inside training and be hard to read.
    log.info("[4] Initializing MuJoCo humanoid + physics …")
    try:
        from milerunner.physics.humanoid import Humanoid
        _probe = Humanoid(body.scaled(), friction=ecfg.friction,
                          timestep=ecfg.physics_timestep)
        _probe.reset()
        _probe.step(1)
        log.info("[4] MuJoCo OK — %.0f kg humanoid, %d actuators, pelvis z=%.2f m",
                 sum(_probe.model.body_mass), _probe.n_actuators, _probe.pelvis_height())
    except Exception as e:
        log.error("[4] MuJoCo/humanoid FAILED to initialize: %s", e, exc_info=True)
        log.error("    -> The trainer cannot run. This is usually a bad MUJOCO_GL "
                  "backend; the trainer needs NO OpenGL, so ensure MUJOCO_GL is unset.")
        raise

    log.info("[5] Physics: %d Hz (%.0f steps/sim-sec) | control: %.0f Hz | distance: %.0f m",
             int(1.0 / ecfg.physics_timestep), 1.0 / ecfg.physics_timestep,
             ecfg.control_hz, ecfg.distance_m)
    trainer = ContinuousTrainer(tcfg, pcfg, ecfg, full_config=cfg.to_dict(), body=body)
    log.info("[6] Trainer + experiment DB ready — starting population")
    log.info("[7] Training generation 0 … (first dashboard data in ~2–3 min on a free CPU)")
    log.info("Press Ctrl-C to pause; re-run to resume.")
    trainer.run(max_generations=args.generations)


if __name__ == "__main__":
    main()
