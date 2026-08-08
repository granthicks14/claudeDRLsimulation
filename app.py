"""Gradio front-end for MileRunner — deployable as a FREE Hugging Face Space.

Hugging Face's Gradio SDK is free (no card, no Docker) and gives 16 GB RAM,
which is plenty for this program. This app:

* starts the autonomous trainer (``scripts/run.py``) in the background on launch;
* shows the live dashboard — fastest mile, training progress, and the best
  agent's speed / heart-rate / cadence / oxygen / energy curves, a muscle-fatigue
  heat-map and an animated 3D replay — refreshing every few seconds.

Run locally with ``python app.py`` (opens http://localhost:7860) or deploy it to
a free Gradio Space (see docs/HUGGINGFACE_SETUP.md).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

import gradio as gr

from milerunner.biomech.muscles import MUSCLE_GROUPS
from milerunner.dashboard import figures as F
from milerunner.dashboard.replay import (Replay, animated_skeleton_figure,
                                         side_run_figure)
from milerunner.dashboard.track_view import track_figure
from milerunner.database.experiment_db import ExperimentDB

DB = os.environ.get("MILE_DB", "experiments/milerunner.db")
STATUS = os.environ.get("MILE_STATUS", "experiments/status.json")
BEST_TELE = os.environ.get("MILE_BEST_TELE", "experiments/best_telemetry.json")
EXPERIMENT = os.environ.get("MILE_EXPERIMENT", "hosted")
CONFIG = os.environ.get("MILE_CONFIG", "hosted")

_trainer_proc = None


def ensure_trainer_running() -> None:
    """Start the trainer subprocess once; restart it if it has died."""
    global _trainer_proc
    if _trainer_proc is not None and _trainer_proc.poll() is None:
        return
    os.makedirs("experiments", exist_ok=True)
    logf = open("experiments/train.log", "a")
    _trainer_proc = subprocess.Popen(
        [sys.executable, "scripts/run.py", "--config", CONFIG],
        stdout=logf, stderr=subprocess.STDOUT,
    )


def _load_status() -> dict:
    try:
        with open(STATUS) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _experiment_id(db: ExperimentDB):
    exps = db.list_experiments()
    for e in exps:
        if e["name"] == EXPERIMENT:
            return e["id"]
    return exps[0]["id"] if exps else None


def _best_telemetry(db: ExperimentDB, exp_id):
    # Prefer a completed-mile record (has a real mile_time); otherwise fall back
    # to the current best agent's live telemetry sidecar written each generation.
    rec = db.best_mile_time(exp_id)
    if rec:
        try:
            return rec, json.loads(rec.get("telemetry_json") or "{}")
        except Exception:
            return rec, {}
    try:
        with open(BEST_TELE) as fh:
            return None, json.load(fh)
    except Exception:
        return None, {}


def _replay(tele: dict) -> Replay:
    r = Replay()
    for frame in (tele.get("skeleton") or []):
        r.frames.append({b: np.array(p) for b, p in frame.items()})
    r.times = tele.get("skeleton_t", [])
    return r


def refresh():
    """Return the KPI header + 8 figures for the dashboard."""
    ensure_trainer_running()
    status = _load_status()
    try:
        db = ExperimentDB(DB)
        exp_id = _experiment_id(db)
    except Exception:
        exp_id = None

    if exp_id is None:
        kpi = ("### ⏳ Training is starting…\n"
               "The first numbers and charts appear within a couple of minutes — "
               "this panel refreshes automatically.")
        empty = F._empty("waiting for data…")
        return (kpi, track_figure({}), side_run_figure(Replay()), empty, empty,
                empty, empty, empty, empty, empty, empty)

    hist = db.generation_history(exp_id)
    rec, tele = _best_telemetry(db, exp_id)
    mile = rec.get("mile_time") if rec else None
    kpi = (
        "### 🏃 MileRunner — live\n"
        f"**Fastest mile:** {f'{mile:.1f}s' if mile else '—'}  ·  "
        f"**Generation:** {status.get('generation', (hist[-1]['gen_index'] if hist else 0))}  ·  "
        f"**Cumulative steps:** {status.get('total_timesteps', 0):,}  ·  "
        f"**Population:** {status.get('population_size', '—')}  ·  "
        f"**Finishers:** {status.get('last_summary', {}).get('n_finished', '—')}"
    )
    fatigue = {g: tele.get(f"fatigue_{g}", []) for g in MUSCLE_GROUPS}
    rp = _replay(tele)
    return (
        kpi,
        track_figure(tele),
        side_run_figure(rp),
        F.training_progress(hist),
        animated_skeleton_figure(rp),
        F.speed_curve(tele),
        F.heart_rate_curve(tele),
        F.cadence_curve(tele),
        F.oxygen_curve(tele),
        F.energy_curve(tele),
        F.muscle_fatigue_heatmap(fatigue, MUSCLE_GROUPS, distance=tele.get("distance")),
    )


def build_demo() -> "gr.Blocks":
    with gr.Blocks(title="MileRunner", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🏃 MileRunner — Discovering the Fastest Human Mile\n"
            "Populations of AI agents learn to run a mile under realistic "
            "biomechanics — no technique is hard-coded. Training runs live on "
            "this Space; the charts below refresh automatically."
        )
        kpi = gr.Markdown()
        # The two "watch the AI run" views: a side-view runner + the oval track.
        with gr.Row():
            g_side = gr.Plot(label="🎥 Watch the AI run (side view)")
            g_track = gr.Plot(label="🏁 Position on the 400 m track")
        with gr.Row():
            g_prog = gr.Plot(label="Training progress")
            g_replay = gr.Plot(label="Best-agent 3D replay")
        with gr.Row():
            g_speed = gr.Plot(label="Speed")
            g_hr = gr.Plot(label="Heart rate")
            g_cad = gr.Plot(label="Cadence")
        with gr.Row():
            g_o2 = gr.Plot(label="Oxygen (VO₂)")
            g_energy = gr.Plot(label="Energy reserves")
            g_fat = gr.Plot(label="Muscle fatigue")

        outputs = [kpi, g_track, g_side, g_prog, g_replay,
                   g_speed, g_hr, g_cad, g_o2, g_energy, g_fat]
        # Initial paint + auto-refresh every 6 seconds.
        demo.load(refresh, outputs=outputs)
        timer = gr.Timer(6)
        timer.tick(refresh, outputs=outputs)
    return demo


# Start training as soon as the app is imported/launched.
ensure_trainer_running()
demo = build_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("PORT", "7860")))
