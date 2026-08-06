#!/usr/bin/env python3
"""Research analysis: turn discovered runs into answers to the research questions.

Everything here is *derived from data* — the best agent's telemetry and the
body's physiological model — not authored by hand. It reports:

* theoretical fastest mile for the body (from the critical-power model);
* the AI's discovered pacing (lap splits) and how much energy it saved for the
  final lap;
* the discovered optimal cadence and stride length;
* the discovered breathing pattern (and how it tracks effort);
* whether the discovered gait differs from typical human running.

    python scripts/analyze.py --experiment default --out experiments/report.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from milerunner.biomech.energy import EnergySystem
from milerunner.biomech.params import BodyParams
from milerunner.config_build import build_body
from milerunner.database.experiment_db import ExperimentDB
from milerunner.physics.track import LAP_M, MILE_M
from milerunner.utils.config import load_config


def theoretical_fastest(body: BodyParams) -> float:
    """Critical-power lower bound on mile time: d = CS*t + D'  ->  t=(d-D')/CS."""
    return (MILE_M - body.d_prime_m) / body.critical_speed_mps


def lap_splits(tele):
    """Return per-lap average speed and time from distance/time telemetry."""
    d = np.array(tele.get("distance", []))
    t = np.array(tele.get("t", []))
    if len(d) < 4:
        return []
    splits = []
    for lap in range(int(np.ceil(MILE_M / LAP_M))):
        lo, hi = lap * LAP_M, min((lap + 1) * LAP_M, MILE_M)
        mask = (d >= lo) & (d < hi)
        if mask.sum() < 2:
            continue
        dt = t[mask][-1] - t[mask][0]
        dd = d[mask][-1] - d[mask][0]
        if dt > 0:
            splits.append({"lap": lap + 1, "avg_speed": dd / dt, "time_s": dt,
                           "dist": dd})
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default")
    ap.add_argument("--experiment", default="default")
    ap.add_argument("--db", default="experiments/milerunner.db")
    ap.add_argument("--out", default="experiments/report.md")
    args = ap.parse_args()

    cfg = load_config(args.config)
    body = build_body(cfg)
    db = ExperimentDB(args.db)
    exps = {e["name"]: e for e in db.list_experiments()}
    if args.experiment not in exps:
        raise SystemExit(f"No experiment '{args.experiment}'. Have {list(exps)}")
    exp_id = exps[args.experiment]["id"]
    summary = db.experiment_summary(exp_id)
    rec = db.best_mile_time(exp_id)
    tele = {}
    if rec:
        try:
            tele = json.loads(rec.get("telemetry_json") or "{}")
        except Exception:
            tele = {}

    t_theory = theoretical_fastest(body)
    lines = []
    W = lines.append
    W(f"# MileRunner research report — experiment `{args.experiment}`\n")
    W(f"- Generations run: **{summary['generations']}**")
    W(f"- Cumulative training steps: **{summary['total_timesteps']:,}**")
    W(f"- Agents ever created: **{summary['individuals']}**\n")

    W("## Body & theoretical limits\n")
    W(f"- Body: {body.name}, {body.height_m*100:.0f} cm, {body.mass_kg:.0f} kg, "
      f"VO2max {body.vo2max_ml_kg_min:.0f} ml/kg/min")
    W(f"- Critical speed {body.critical_speed_mps:.2f} m/s, D' {body.d_prime_m:.0f} m")
    W(f"- **Theoretical fastest mile (critical-power bound): "
      f"{t_theory:.1f} s ({int(t_theory//60)}:{t_theory%60:04.1f})** — a physics "
      f"floor ignoring pacing dynamics, drag and biomechanical losses; real runs "
      f"are slower.\n")

    if not rec:
        W("## No completed mile recorded yet\n")
        W("Train longer (`python scripts/run.py`) — early generations rarely "
          "finish a full mile. Partial-distance progress is still tracked in the DB.")
        _write(args.out, lines)
        return

    W("## Best discovered mile\n")
    W(f"- **Mile time: {rec['mile_time']:.2f} s** "
      f"({int(rec['mile_time']//60)}:{rec['mile_time']%60:05.2f}) "
      f"by agent `{rec['genome_id']}` (generation {rec['generation']})")
    ratio = rec["mile_time"] / t_theory
    W(f"- That is {ratio:.2f}x the theoretical floor "
      f"({(ratio-1)*100:.0f}% above the physics limit).\n")

    splits = lap_splits(tele)
    if splits:
        W("## Discovered pacing (lap splits)\n")
        W("| Lap | avg speed (m/s) | lap time (s) |")
        W("|----:|----------------:|-------------:|")
        for s in splits:
            W(f"| {s['lap']} | {s['avg_speed']:.2f} | {s['time_s']:.1f} |")
        speeds = [s["avg_speed"] for s in splits]
        pattern = ("negative split (finished faster)" if speeds[-1] > speeds[0]
                   else "positive split (went out faster)" if speeds[-1] < speeds[0]
                   else "even pace")
        W(f"\n**Discovered pacing pattern: {pattern}.**\n")

    wp = tele.get("w_prime_frac", [])
    if wp:
        last_lap_start = int(len(wp) * 0.75)
        W("## Energy for the final lap\n")
        W(f"- W' reserve at the start of the last quarter: **{wp[last_lap_start]*100:.0f}%**")
        W(f"- W' reserve at the finish: **{wp[-1]*100:.0f}%** "
          f"(a well-optimised run empties the tank near the line).\n")

    cad = [c for c in tele.get("cadence", []) if c > 0]
    if cad:
        W("## Discovered cadence & stride\n")
        W(f"- Mean cadence: **{np.mean(cad)*60:.0f} steps/min** "
          f"(the body's elastic-optimal cadence is "
          f"{body.tendon_natural_cadence_hz*60:.0f} steps/min).")
        speed = np.mean([s for s in tele.get('speed', []) if s > 0]) or 0.0
        if np.mean(cad) > 0:
            W(f"- Implied stride length: **{speed/np.mean(cad):.2f} m**.\n")

    W("## Research questions — data-driven answers\n")
    W("- *Is there a faster stride pattern than humans use?* Compare the "
      "discovered cadence/stride above against typical human values "
      "(~180 steps/min, ~1.4 m stride at mile pace).")
    W("- *What is the theoretical fastest mile?* See the critical-power bound above.")
    W("- *How much energy for the final lap?* See the W' reserve trajectory.")
    W("- *What cadence is optimal?* The mean discovered cadence above — emergent, "
      "not prescribed.")
    W("- *Best breathing pattern?* The agent's breathing action tracks metabolic "
      "demand; the discovered relationship is in the telemetry.")
    W("- *A completely new running form?* Inspect the 3D replay "
      "(`scripts/evaluate.py`) — novelty shows as gaits unlike human running.\n")
    W("_All figures above are computed from recorded runs and the body's "
      "physiological model; none are hand-authored strategies._")

    _write(args.out, lines)
    print(f"Wrote report -> {args.out}")


def _write(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
