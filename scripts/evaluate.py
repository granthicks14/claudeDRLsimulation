#!/usr/bin/env python3
"""Evaluate the best discovered agent and export its replay.

Loads the fastest agent recorded in the experiment database, runs a full mile,
prints a physiological + kinematic summary, saves the telemetry JSON, and (if a
GL backend is available) exports a MuJoCo video. Always writes an interactive
HTML 3D skeleton replay, which needs no OpenGL.

    python scripts/evaluate.py --experiment default
    python scripts/evaluate.py --experiment default --video out/best.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from milerunner.agents.factory import build_agent
from milerunner.config_build import build_env_config
from milerunner.database.experiment_db import ExperimentDB
from milerunner.evolution.genome import Genome
from milerunner.training.checkpoint import load_genome, load_search_state
from milerunner.training.env_builder import build_single_env
from milerunner.training.evaluation import evaluate_policy
from milerunner.utils.config import load_config
from milerunner.utils.logging import configure_logging, get_logger


def find_best(db: ExperimentDB, experiment: str, state_dir: str, ckpt_root: str):
    """Return (genome, model_path). Prefer the current best agent whose
    checkpoint still exists (from the search-state file); fall back to the DB.
    """
    # 1) Search-state file: current population with live checkpoints.
    state = load_search_state(os.path.join(state_dir, f"{experiment}_state.json"))
    if state:
        cands = [i for i in state["individuals"]
                 if i.get("model_path") and os.path.exists(i["model_path"])]
        if state.get("best_record") and os.path.exists(
                os.path.join(ckpt_root, experiment,
                             state["best_record"]["genome_id"], "model.zip")):
            gid = state["best_record"]["genome_id"]
            return load_genome(ckpt_root, experiment, gid), \
                os.path.join(ckpt_root, experiment, gid, "model.zip")
        if cands:
            best = max(cands, key=lambda i: i.get("fitness", -1e9))
            return Genome.from_dict(best["genome"]), best["model_path"]
    # 2) DB fallback.
    exps = {e["name"]: e for e in db.list_experiments()}
    if experiment not in exps:
        raise SystemExit(f"No experiment '{experiment}'. Have: {list(exps)}")
    exp_id = exps[experiment]["id"]
    rec = db.best_mile_time(exp_id) or db.best_by_fitness(exp_id)
    if not rec:
        raise SystemExit("No evaluations recorded yet — train first.")
    gid = rec["genome_id"]
    return load_genome(ckpt_root, experiment, gid), \
        os.path.join(ckpt_root, experiment, gid, "model.zip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default")
    ap.add_argument("--experiment", default="default")
    ap.add_argument("--checkpoint-root", default="checkpoints")
    ap.add_argument("--db", default="experiments/milerunner.db")
    ap.add_argument("--out", default="experiments/best_eval.json")
    ap.add_argument("--html", default="experiments/best_replay.html")
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    configure_logging("INFO")
    log = get_logger("milerunner.evaluate")
    cfg = load_config(args.config)
    env_cfg = build_env_config(cfg)

    db = ExperimentDB(args.db)
    genome, model_path = find_best(db, args.experiment, "experiments",
                                   args.checkpoint_root)
    log.info("Best agent %s | algo=%s arch=%s", genome.genome_id, genome.algo, genome.arch)

    env = build_single_env(reward_weights=genome.reward_weights_obj(),
                           config=env_cfg, seed=999)
    model = build_agent(genome.algo, env, hyperparams=genome.hp_for_agent(),
                        arch=genome.arch, device="cpu")
    if os.path.exists(model_path):
        model.set_parameters(model_path, exact_match=False)

    result = evaluate_policy(model, env=env, record_telemetry=True,
                             record_skeleton=True, seed=999)
    log.info("Mile: %s | distance %.1f m | mean speed %.2f m/s | peak %.2f m/s | "
             "cadence %.2f/s | HR %.0f bpm",
             f"{result.mile_time:.2f}s" if result.finished else "did not finish",
             result.distance, result.mean_speed, result.peak_speed,
             result.mean_cadence, result.mean_hr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result.to_metrics(), fh, indent=2)
    log.info("Wrote summary -> %s", args.out)

    # interactive HTML 3D replay (no GL required)
    from milerunner.dashboard.app import _replay_from_telemetry
    from milerunner.dashboard.replay import animated_skeleton_figure
    replay = _replay_from_telemetry(result.telemetry)
    fig = animated_skeleton_figure(replay)
    fig.write_html(args.html)
    log.info("Wrote 3D replay -> %s", args.html)

    if args.video:
        from milerunner.dashboard.replay import render_mujoco_video
        env2 = build_single_env(reward_weights=genome.reward_weights_obj(),
                                config=env_cfg, seed=999)
        path = render_mujoco_video(model, env2, args.video)
        if path:
            log.info("Wrote video -> %s", path)
        else:
            log.warning("No GL backend available; skipped MuJoCo video "
                        "(HTML replay still written).")


if __name__ == "__main__":
    main()
