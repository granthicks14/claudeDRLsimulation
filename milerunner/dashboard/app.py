"""Live analytics dashboard (Dash + Plotly).

Runs alongside the trainer and refreshes on an interval, reading the trainer's
``status.json`` and the experiment database. Shows the fastest mile, training
progress, and — for the best agent discovered so far — the speed / cadence /
heart-rate / oxygen / energy / lactate curves, a muscle-fatigue heat-map, an
animated 3D skeleton replay, and per-algorithm/architecture comparisons.

Start it with ``python scripts/dashboard.py`` (optionally while
``python scripts/run.py`` trains in the background).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ..biomech.muscles import MUSCLE_GROUPS
from ..database.experiment_db import ExperimentDB
from . import figures as F
from .replay import Replay, animated_skeleton_figure
from .track_view import track_figure


def _load_status(path: str) -> Dict[str, Any]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _best_telemetry(db: ExperimentDB, exp_id: int,
                    best_tele_path: str = "experiments/best_telemetry.json") -> Dict[str, Any]:
    rec = db.best_mile_time(exp_id)
    if rec:
        try:
            return {"record": rec, "telemetry": json.loads(rec.get("telemetry_json") or "{}")}
        except Exception:
            return {"record": rec, "telemetry": {}}
    # No finished mile yet — use the live best-agent telemetry sidecar.
    try:
        with open(best_tele_path) as fh:
            return {"record": None, "telemetry": json.load(fh)}
    except Exception:
        return {}


def _replay_from_telemetry(tele: Dict[str, Any]) -> Replay:
    import numpy as np
    r = Replay()
    frames = tele.get("skeleton") or []
    for f in frames:
        r.frames.append({b: np.array(p) for b, p in f.items()})
    r.times = tele.get("skeleton_t", list(range(len(frames))))
    return r


def create_app(db_path: str = "experiments/milerunner.db",
               status_path: str = "experiments/status.json",
               experiment: Optional[str] = None):
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output

    db = ExperimentDB(db_path)

    def resolve_exp_id() -> Optional[int]:
        exps = db.list_experiments()
        if experiment:
            for e in exps:
                if e["name"] == experiment:
                    return e["id"]
        return exps[0]["id"] if exps else None

    app = dash.Dash(__name__, title="MileRunner — AI Mile Discovery")

    card_style = {"backgroundColor": "#111827", "borderRadius": "12px",
                  "padding": "14px", "margin": "8px", "flex": "1",
                  "boxShadow": "0 2px 8px rgba(0,0,0,0.4)"}
    kpi_style = {**card_style, "textAlign": "center"}

    def kpi(title, value_id):
        return html.Div([
            html.Div(title, style={"color": "#9ca3af", "fontSize": "13px"}),
            html.Div(id=value_id, style={"color": "#22d3ee", "fontSize": "26px",
                                         "fontWeight": "700"}),
        ], style=kpi_style)

    app.layout = html.Div(style={"backgroundColor": "#0b1220", "minHeight": "100vh",
                                 "fontFamily": "Inter, system-ui, sans-serif",
                                 "color": "#e5e7eb", "padding": "12px"}, children=[
        html.H2("🏃 MileRunner — Discovering the Fastest Human Mile",
                style={"margin": "6px 12px"}),
        html.Div("Population-based deep-RL + evolution. Training runs autonomously; "
                 "this view refreshes live.", style={"margin": "0 12px 8px",
                                                      "color": "#9ca3af"}),
        html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
            kpi("Fastest mile", "kpi-mile"),
            kpi("Generation", "kpi-gen"),
            kpi("Cumulative steps", "kpi-steps"),
            kpi("Population", "kpi-pop"),
            kpi("Finishers", "kpi-finish"),
        ]),
        dcc.Interval(id="tick", interval=5000, n_intervals=0),
        html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
            html.Div(dcc.Graph(id="g-track"), style={**card_style, "minWidth": "460px", "flex": "2"}),
            html.Div(dcc.Graph(id="g-replay"), style={**card_style, "minWidth": "420px"}),
        ]),
        html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
            html.Div(dcc.Graph(id="g-progress"), style=card_style),
        ]),
        html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
            html.Div(dcc.Graph(id="g-speed"), style=card_style),
            html.Div(dcc.Graph(id="g-cadence"), style=card_style),
            html.Div(dcc.Graph(id="g-hr"), style=card_style),
        ]),
        html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
            html.Div(dcc.Graph(id="g-oxygen"), style=card_style),
            html.Div(dcc.Graph(id="g-energy"), style=card_style),
            html.Div(dcc.Graph(id="g-lactate"), style=card_style),
        ]),
        html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
            html.Div(dcc.Graph(id="g-fatigue"), style={**card_style, "flex": "2"}),
            html.Div(dcc.Graph(id="g-algo"), style=card_style),
        ]),
        html.Div(style={"display": "flex"}, children=[
            html.Div(dcc.Graph(id="g-leader"), style={**card_style, "flex": "1"}),
        ]),
    ])

    @app.callback(
        [Output("kpi-mile", "children"), Output("kpi-gen", "children"),
         Output("kpi-steps", "children"), Output("kpi-pop", "children"),
         Output("kpi-finish", "children"),
         Output("g-track", "figure"),
         Output("g-progress", "figure"), Output("g-replay", "figure"),
         Output("g-speed", "figure"), Output("g-cadence", "figure"),
         Output("g-hr", "figure"), Output("g-oxygen", "figure"),
         Output("g-energy", "figure"), Output("g-lactate", "figure"),
         Output("g-fatigue", "figure"), Output("g-algo", "figure"),
         Output("g-leader", "figure")],
        [Input("tick", "n_intervals")],
    )
    def refresh(_n):
        status = _load_status(status_path)
        exp_id = resolve_exp_id()
        history = db.generation_history(exp_id) if exp_id else []
        best = _best_telemetry(db, exp_id) if exp_id else {}
        tele = best.get("telemetry", {})
        rec = best.get("record", {})

        mile = rec.get("mile_time")
        mile_txt = f"{mile:.1f}s" if mile else "—"
        gen_txt = str(status.get("generation", history[-1]["gen_index"] if history else 0))
        steps_txt = f"{status.get('total_timesteps', 0):,}"
        pop_txt = str(status.get("population_size", "—"))
        finish_txt = str(status.get("last_summary", {}).get("n_finished", "—"))

        replay = _replay_from_telemetry(tele)
        fig_replay = animated_skeleton_figure(replay)
        fig_replay.update_layout(title="Best-agent 3D replay")

        fatigue_timeline = {g: tele.get(f"fatigue_{g}", []) for g in MUSCLE_GROUPS}
        fig_fatigue = F.muscle_fatigue_heatmap(
            fatigue_timeline, MUSCLE_GROUPS, distance=tele.get("distance"))

        by_algo = status.get("by_algo", {})
        leaderboard = status.get("leaderboard", [])

        return (
            mile_txt, gen_txt, steps_txt, pop_txt, finish_txt,
            track_figure(tele),
            F.training_progress(history), fig_replay,
            F.speed_curve(tele), F.cadence_curve(tele), F.heart_rate_curve(tele),
            F.oxygen_curve(tele), F.energy_curve(tele), F.lactate_temp_curve(tele),
            fig_fatigue, F.algo_comparison(by_algo, "best_fitness"),
            F.leaderboard_table(leaderboard),
        )

    return app


def run_dashboard(db_path: str = "experiments/milerunner.db",
                  status_path: str = "experiments/status.json",
                  experiment: Optional[str] = None,
                  host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
    app = create_app(db_path, status_path, experiment)
    app.run(host=host, port=port, debug=debug)
