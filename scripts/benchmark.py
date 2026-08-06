#!/usr/bin/env python3
"""Benchmarking suite.

Three benchmarks:

1. **Physics throughput** — verifies the simulator runs at 1000+ physics
   steps per second (the brief's hard requirement) and reports control-step and
   environment throughput, single- and multi-env.
2. **Biomechanical model validation** — checks the energy model produces
   physiologically plausible time-to-exhaustion curves from the critical-power
   relationship (a sanity test of the hardest part of the platform).
3. **Best-agent conditions sweep** — evaluates the best discovered agent across
   the weather presets to quantify condition sensitivity.

    python scripts/benchmark.py --physics
    python scripts/benchmark.py --biomech
    python scripts/benchmark.py --all
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from milerunner.biomech.energy import EnergySystem
from milerunner.biomech.params import BodyParams
from milerunner.envs.mile_env import EnvConfig, MileRunEnv
from milerunner.utils.logging import configure_logging, get_logger

log = get_logger("milerunner.benchmark")


def bench_physics(n_steps: int = 3000):
    log.info("=== Physics throughput ===")
    for n_envs in (1, 4):
        from milerunner.training.env_builder import build_vec_env
        venv = build_vec_env(n_envs=n_envs, config=EnvConfig(max_time_s=1e6))
        venv.reset()
        acts = np.stack([venv.action_space.sample() for _ in range(n_envs)]) * 0.1
        t0 = time.time()
        control_steps = 0
        for _ in range(n_steps):
            venv.step(acts)
            control_steps += n_envs
        dt = time.time() - t0
        env0 = venv.get_attr("n_substeps")[0]
        phys = control_steps * env0 / dt
        log.info("n_envs=%d: %.0f control-steps/s, %.0f PHYSICS-steps/s  [req >=1000]  %s",
                 n_envs, control_steps / dt, phys, "OK" if phys >= 1000 else "LOW")
        venv.close()


def bench_biomech():
    log.info("=== Biomechanical model validation (critical-power) ===")
    p = BodyParams()
    es = EnergySystem(p)
    cs = p.critical_speed_mps
    log.info("critical speed = %.2f m/s | D' = %.0f m | VO2max = %.0f ml/kg/min",
             cs, p.d_prime_m, p.vo2max_ml_kg_min)
    log.info("Predicted time-to-exhaustion at constant speeds above CS:")
    for speed in (cs + 0.3, cs + 0.6, cs + 1.0, cs + 1.5):
        es.reset()
        t = 0.0
        dt = 0.1
        while not es.exhausted and t < 2000:
            es.step(dt, speed_mps=speed, cadence_hz=p.tendon_natural_cadence_hz)
            t += dt
        dist = speed * t
        log.info("  %.2f m/s (+%.2f over CS): exhausted at %6.1f s  (%.0f m)  HR~%.0f",
                 speed, speed - cs, t, dist, es.hr)
    # A pace at/below CS should be ~sustainable (not exhaust quickly).
    es.reset()
    t = 0.0
    while not es.exhausted and t < 3600:
        es.step(0.2, speed_mps=cs - 0.2, cadence_hz=p.tendon_natural_cadence_hz)
        t += 0.2
    log.info("  at CS-0.2 m/s: %s after %.0f s (should be sustainable)",
             "exhausted" if es.exhausted else "still going", t)


def bench_conditions(experiment: str, db_path: str, ckpt_root: str):
    log.info("=== Best-agent conditions sweep ===")
    from milerunner.agents.factory import build_agent
    from milerunner.database.experiment_db import ExperimentDB
    from milerunner.physics.track import Weather
    from milerunner.training.checkpoint import load_genome
    from milerunner.training.env_builder import build_single_env
    from milerunner.training.evaluation import evaluate_policy

    db = ExperimentDB(db_path)
    exps = {e["name"]: e for e in db.list_experiments()}
    if experiment not in exps:
        log.warning("No experiment '%s' yet — train first.", experiment)
        return
    rec = db.best_mile_time(exps[experiment]["id"])
    if not rec:
        log.warning("No finished-mile record yet.")
        return
    genome = load_genome(ckpt_root, experiment, rec["genome_id"])
    model_path = os.path.join(ckpt_root, experiment, rec["genome_id"], "model.zip")
    conditions = {
        "cool_calm": Weather(temperature_c=12, wind_mps=0),
        "hot_humid": Weather(temperature_c=32, wind_mps=0, humidity=0.85),
        "headwind": Weather(temperature_c=15, wind_mps=-4),
        "tailwind": Weather(temperature_c=15, wind_mps=3),
        "altitude": Weather(temperature_c=15, altitude_m=2200),
    }
    for name, w in conditions.items():
        env = build_single_env(reward_weights=genome.reward_weights_obj(),
                               weather=w, config=EnvConfig(), seed=7)
        model = build_agent(genome.algo, env, hyperparams=genome.hp_for_agent(),
                            arch=genome.arch, device="cpu")
        if os.path.exists(model_path):
            model.set_parameters(model_path, exact_match=False)
        r = evaluate_policy(model, env=env, record_telemetry=False)
        log.info("  %-10s: %s  mean speed %.2f m/s",
                 name, f"{r.mile_time:.1f}s" if r.finished else "DNF", r.mean_speed)
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--physics", action="store_true")
    ap.add_argument("--biomech", action="store_true")
    ap.add_argument("--conditions", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--experiment", default="default")
    ap.add_argument("--db", default="experiments/milerunner.db")
    ap.add_argument("--checkpoint-root", default="checkpoints")
    args = ap.parse_args()
    configure_logging("INFO")

    if args.all or args.physics:
        bench_physics()
    if args.all or args.biomech:
        bench_biomech()
    if args.all or args.conditions:
        bench_conditions(args.experiment, args.db, args.checkpoint_root)
    if not any([args.physics, args.biomech, args.conditions, args.all]):
        bench_physics()
        bench_biomech()


if __name__ == "__main__":
    main()
