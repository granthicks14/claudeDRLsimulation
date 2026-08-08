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
import shutil
import subprocess
import sys
import threading

import numpy as np

# IMPORTANT: we do NOT set MUJOCO_GL at import time. Setting it makes
# `import mujoco` eagerly load an OpenGL backend, which CRASHES the import when
# OSMesa/EGL isn't perfectly available — and the trainer subprocess (which does
# physics only and needs no GL) would inherit that env and die silently.
# Instead the render path (only) selects a backend just before rendering, in the
# app process, isolated from the trainer.

def _has_gpu() -> bool:
    return (os.path.exists("/proc/driver/nvidia/version")
            or shutil.which("nvidia-smi") is not None)

import gradio as gr

from milerunner.biomech.muscles import MUSCLE_GROUPS
from milerunner.dashboard import figures as F
from milerunner.dashboard.replay import (Replay, animated_skeleton_figure,
                                         side_run_figure)
from milerunner.dashboard.track_view import race_figure, track_figure
from milerunner.database.experiment_db import ExperimentDB

DB = os.environ.get("MILE_DB", "experiments/milerunner.db")
STATUS = os.environ.get("MILE_STATUS", "experiments/status.json")
BEST_TELE = os.environ.get("MILE_BEST_TELE", "experiments/best_telemetry.json")
RACE = os.environ.get("MILE_RACE", "experiments/race.json")
EXPERIMENT = os.environ.get("MILE_EXPERIMENT", "hosted")
CONFIG = os.environ.get("MILE_CONFIG", "hosted")


def _load_race():
    try:
        with open(RACE) as fh:
            return json.load(fh)
    except Exception:
        return []


# --- Optional photorealistic MuJoCo render (auto-enabled when a GPU is present).
_VIDEO_PATH = None
_LAST_RENDER_KEY = None
_RENDER_LOCK = threading.Lock()


def _maybe_render_video():
    """If a GL backend exists, render the latest best run to a video in the
    background (throttled to once per new best telemetry). No-op on CPU."""
    global _VIDEO_PATH, _LAST_RENDER_KEY
    try:
        # Choose the render backend HERE (never at import time): GPU->EGL, else
        # OSMesa (software GL on CPU). This only affects the app process, not the
        # trainer. If OSMesa/EGL isn't available the import fails -> caught below.
        os.environ.setdefault("MUJOCO_GL", "egl" if _has_gpu() else "osmesa")
        from milerunner.dashboard.render3d import gl_available
        if not gl_available():
            return
        key = os.path.getmtime(BEST_TELE)
    except Exception:
        return
    if key == _LAST_RENDER_KEY or not _RENDER_LOCK.acquire(blocking=False):
        return

    def _work():
        global _VIDEO_PATH, _LAST_RENDER_KEY
        try:
            with open(BEST_TELE) as fh:
                tele = json.load(fh)
            from milerunner.config_build import build_body
            from milerunner.dashboard.render3d import render_best_video
            from milerunner.utils.config import load_config
            body = build_body(load_config(CONFIG))
            path = render_best_video(body, tele, "experiments/best_render.mp4")
            if path:
                _VIDEO_PATH = path
            _LAST_RENDER_KEY = key
        except Exception:
            pass
        finally:
            _RENDER_LOCK.release()

    threading.Thread(target=_work, daemon=True).start()

_trainer_proc = None
_trainer_restarts = 0
_trainer_error = None
_MAX_RESTARTS = 3


def _log_tail(n_chars: int = 2500) -> str:
    try:
        with open("experiments/train.log") as fh:
            return fh.read()[-n_chars:]
    except Exception:
        return ""


def ensure_trainer_running() -> None:
    """Start the trainer subprocess; restart on death up to a limit, and capture
    the error so the dashboard can show it instead of hanging on 'waiting'."""
    global _trainer_proc, _trainer_restarts, _trainer_error
    if _trainer_proc is not None and _trainer_proc.poll() is None:
        return                                   # already running
    if _trainer_proc is not None:                # it died since last check
        _trainer_error = _log_tail()
        _trainer_restarts += 1
        if _trainer_restarts > _MAX_RESTARTS:
            return                               # give up; leave the error for the UI
        print(f"[app] trainer exited — restart {_trainer_restarts}/{_MAX_RESTARTS}", flush=True)

    os.makedirs("experiments", exist_ok=True)
    # CRITICAL: the trainer does physics only and needs NO OpenGL. Strip
    # MUJOCO_GL from its environment so `import mujoco` stays lazy about GL and
    # can't crash on a missing OSMesa/EGL backend (that was the
    # 'Waiting for runner' bug).
    env = dict(os.environ)
    env.pop("MUJOCO_GL", None)
    logf = open("experiments/train.log", "a")
    print(f"[app] launching trainer: {sys.executable} scripts/run.py --config {CONFIG}", flush=True)
    _trainer_proc = subprocess.Popen(
        [sys.executable, "scripts/run.py", "--config", CONFIG],
        stdout=logf, stderr=subprocess.STDOUT, env=env,
    )
    print(f"[app] trainer started (pid {_trainer_proc.pid})", flush=True)


def trainer_alive() -> bool:
    return _trainer_proc is not None and _trainer_proc.poll() is None


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


MILE_M = 1609.344


def _stats_markdown(status: dict, tele: dict, furthest: float) -> str:
    """Plain-language progress stats for the sidebar."""
    speeds = [s for s in (tele.get("speed") or []) if s == s]
    top = max(speeds) if speeds else 0.0
    hr = tele.get("heart_rate") or []
    avg_hr = sum(hr) / len(hr) if hr else 0.0
    cad = [c for c in (tele.get("cadence") or []) if c > 0]
    avg_cad = (sum(cad) / len(cad) * 60.0) if cad else 0.0     # steps per minute
    pct = min(100.0, furthest / MILE_M * 100.0)
    filled = int(round(pct / 10.0))
    bar = "█" * filled + "░" * (10 - filled)
    best = status.get("best_record") or {}
    mile_txt = f"{best['mile_time']:.1f} s" if best.get("mile_time") else "not yet"
    gen = status.get("generation", "—")
    steps = status.get("total_timesteps", 0)
    runners = status.get("population_size", "—")
    n_fin = status.get("last_summary", {}).get("n_finished", 0)
    laps = furthest / 400.0
    return (
        "## 📊 How the AIs are doing\n"
        f"**🏁 Best mile time:** {mile_txt}\n\n"
        f"**📏 Furthest reached:** {furthest:.0f} m  ({pct:.0f}% of the mile)\n\n"
        f"`{bar}`  ~{laps:.1f} / 4 laps\n\n"
        f"**⚡ Top speed:** {top:.1f} m/s  ({top * 2.237:.1f} mph)\n\n"
        f"**🦵 Cadence:** {avg_cad:.0f} steps/min\n\n"
        f"**❤️ Heart rate:** {avg_hr:.0f} bpm\n\n"
        "---\n"
        f"**🧬 Generation:** {gen}\n\n"
        f"**🏃 Runners in the race:** {runners}\n\n"
        f"**⏱️ Experience:** {steps:,} steps\n\n"
        f"**✅ Full miles completed:** {n_fin}\n"
    )


def _startup_message(status: dict, exp_id) -> str:
    """Friendly startup status — or the actual trainer error if it crashed."""
    alive = trainer_alive()
    if _trainer_error and not alive and _trainer_restarts > _MAX_RESTARTS:
        return (
            "## ❌ The trainer stopped with an error\n"
            "The runner process crashed, so no live data can appear. Here are the "
            "last log lines (also in `experiments/train.log`):\n\n"
            "```\n" + (_trainer_error[-1600:] or "(no log captured)") + "\n```\n"
            "Fix the error above, then re-run the launch cell."
        )
    gen = status.get("generation")
    steps = status.get("total_timesteps", 0)
    detail = (f"Generation {gen} in progress · {steps:,} steps so far"
              if gen is not None else
              ("experiment created — training generation 0" if exp_id
               else "starting the trainer…"))
    return (
        "## ⏳ Training is starting…\n"
        "The runner and live data appear when **generation 0 finishes** — this "
        "takes about **2–3 minutes on a free CPU**. Nothing is broken while you "
        "see this; the panel refreshes automatically.\n\n"
        f"_Trainer:_ **{'running ✅' if alive else 'launching…'}**  ·  {detail}\n\n"
        "_(Watch progress in the Colab output, or run "
        "`!tail -n 40 experiments/train.log` in a new cell.)_"
    )


def refresh():
    """Return the stats sidebar + all dashboard figures."""
    ensure_trainer_running()
    status = _load_status()
    try:
        db = ExperimentDB(DB)
        exp_id = _experiment_id(db)
    except Exception:
        exp_id = None

    rec, tele = _best_telemetry(db, exp_id) if exp_id is not None else (None, {})
    has_runner_data = bool(tele.get("speed"))

    # Until the first generation produces telemetry, show a precise startup/error
    # status instead of an indefinite "waiting" — and never hide a trainer crash.
    if not has_runner_data:
        furthest = db.max_distance(exp_id) if exp_id is not None else 0.0
        stats = _startup_message(status, exp_id)
        empty = F._empty("waiting for the first generation to finish…")
        return (stats, _VIDEO_PATH, side_run_figure(Replay()),
                track_figure({}, best_distance=furthest), empty, race_figure(_load_race()),
                empty, empty, empty, empty, empty, empty, empty, empty)

    hist = db.generation_history(exp_id)
    furthest = db.max_distance(exp_id)
    dist_rows = db.distance_by_generation(exp_id)
    stats = _stats_markdown(status, tele, furthest)
    fatigue = {g: tele.get(f"fatigue_{g}", []) for g in MUSCLE_GROUPS}
    rp = _replay(tele)
    _maybe_render_video()          # renders a real 3D video in the background if a GPU is present
    return (
        stats,
        _VIDEO_PATH,
        side_run_figure(rp),                               # hero: the AI running
        track_figure(tele, best_distance=furthest),        # sidebar: on the track
        F.distance_progress(dist_rows),                    # sidebar: progress trend
        race_figure(_load_race()),
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
        gr.Markdown("# 🏃 MileRunner — watch an AI learn to run a mile")

        # ---- Hero row: the runner takes most of the screen; stats on the side ----
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):        # MOST OF THE SCREEN — the AI running
                g_video = gr.Video(
                    label="🎬 Photorealistic 3D render (renders on CPU or GPU — free)",
                    autoplay=True, interactive=False, height=300)
                g_side = gr.Plot(label="🎥 The AI running — press ▶ to watch")
            with gr.Column(scale=1, min_width=300):    # SIDEBAR
                stats = gr.Markdown()
                g_track = gr.Plot(label="🏁 On the track (⚑ = furthest ever)")
                g_dist = gr.Plot(label="📈 Progress each generation")

        # The whole population racing — one model per lane.
        g_race = gr.Plot(label="🏁 The squad racing — a model in every lane")

        # Detailed physiology & training charts, tucked away for a clean top view.
        with gr.Accordion("📊 More detailed charts (physiology & training)", open=False):
            with gr.Row():
                g_prog = gr.Plot(label="Training progress (fitness)")
                g_replay = gr.Plot(label="3D replay")
            with gr.Row():
                g_speed = gr.Plot(label="Speed")
                g_hr = gr.Plot(label="Heart rate")
                g_cad = gr.Plot(label="Cadence")
            with gr.Row():
                g_o2 = gr.Plot(label="Oxygen (VO₂)")
                g_energy = gr.Plot(label="Energy reserves")
                g_fat = gr.Plot(label="Muscle fatigue")

        outputs = [stats, g_video, g_side, g_track, g_dist, g_race, g_prog,
                   g_replay, g_speed, g_hr, g_cad, g_o2, g_energy, g_fat]
        # Initial paint + auto-refresh every 6 seconds.
        demo.load(refresh, outputs=outputs)
        timer = gr.Timer(6)
        timer.tick(refresh, outputs=outputs)
    return demo


# Start training as soon as the app is imported/launched (unless disabled, e.g.
# in tests). Prints go to the Colab output so startup is visible.
print("[app] MileRunner starting — launching trainer + building dashboard", flush=True)
if os.environ.get("MILE_NO_AUTOSTART") != "1":
    ensure_trainer_running()
demo = build_demo()
print("[app] Dashboard ready. Training runs in the background; live data appears "
      "after generation 0 (~2–3 min on a free CPU).", flush=True)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("PORT", "7860")))
