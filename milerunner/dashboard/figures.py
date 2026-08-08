"""Plotly figure builders for the analytics dashboard.

Every figure the brief asks for: fastest-mile / training progress, speed curve,
cadence, heart rate, oxygen consumption, fatigue, muscle activation heat-map,
energy usage, and per-algorithm/architecture comparisons. All are pure
functions of data pulled from the database or a telemetry dict, so they can be
reused in notebooks and the benchmarking suite too.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

_TEMPLATE = "plotly_dark"
_ACCENT = "#22d3ee"
_ACCENT2 = "#f59e0b"
_ACCENT3 = "#a78bfa"


def _empty(title: str):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.update_layout(template=_TEMPLATE, title=title,
                      margin=dict(l=40, r=20, t=40, b=30))
    fig.add_annotation(text="waiting for data…", showarrow=False,
                       xref="paper", yref="paper", x=0.5, y=0.5)
    return fig


def training_progress(history: List[Dict[str, Any]]):
    """Best & mean fitness and best mile time over generations."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    if not history:
        return _empty("Training progress")
    gens = [h["gen_index"] for h in history]
    best = [h.get("best_fitness") for h in history]
    mean = [h.get("mean_fitness") for h in history]
    mile = [h.get("best_mile_time") for h in history]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=gens, y=best, name="best fitness",
                             line=dict(color=_ACCENT, width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=gens, y=mean, name="mean fitness",
                             line=dict(color=_ACCENT3, width=1, dash="dot")), secondary_y=False)
    if any(m is not None for m in mile):
        fig.add_trace(go.Scatter(x=gens, y=mile, name="best mile (s)",
                                 line=dict(color=_ACCENT2, width=2)), secondary_y=True)
    fig.update_layout(template=_TEMPLATE, title="Training progress",
                      margin=dict(l=40, r=40, t=40, b=30), legend=dict(orientation="h"))
    fig.update_yaxes(title_text="fitness", secondary_y=False)
    fig.update_yaxes(title_text="mile time (s)", secondary_y=True)
    fig.update_xaxes(title_text="generation")
    return fig


def _line(telemetry, ykey, title, ylabel, color, xkey="distance"):
    import plotly.graph_objects as go
    if not telemetry or not telemetry.get(ykey):
        return _empty(title)
    x = telemetry.get(xkey) or telemetry.get("t")
    y = telemetry[ykey]
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=2)))
    fig.update_layout(template=_TEMPLATE, title=title,
                      margin=dict(l=45, r=20, t=40, b=35))
    fig.update_xaxes(title_text=xkey)
    fig.update_yaxes(title_text=ylabel)
    return fig


def speed_curve(telemetry):
    return _line(telemetry, "speed", "Speed vs distance", "speed (m/s)", _ACCENT)


def cadence_curve(telemetry):
    return _line(telemetry, "cadence", "Cadence vs distance", "steps/s", _ACCENT2)


def heart_rate_curve(telemetry):
    return _line(telemetry, "heart_rate", "Heart rate vs distance", "bpm", "#ef4444")


def oxygen_curve(telemetry):
    return _line(telemetry, "vo2", "Oxygen uptake (VO2) vs distance", "ml/kg/min", "#34d399")


def energy_curve(telemetry):
    """W' reserve, glycogen and metabolic power together."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    if not telemetry or not telemetry.get("w_prime_frac"):
        return _empty("Energy reserves vs distance")
    x = telemetry.get("distance") or telemetry.get("t")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=x, y=telemetry["w_prime_frac"], name="W' reserve",
                             line=dict(color=_ACCENT, width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=telemetry.get("glycogen", []), name="glycogen",
                             line=dict(color=_ACCENT3, width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=telemetry.get("metabolic_power", []), name="metabolic power (W)",
                             line=dict(color=_ACCENT2, width=1, dash="dot")), secondary_y=True)
    fig.update_layout(template=_TEMPLATE, title="Energy reserves & power vs distance",
                      margin=dict(l=45, r=45, t=40, b=35), legend=dict(orientation="h"))
    fig.update_yaxes(title_text="fraction remaining", secondary_y=False, range=[0, 1.05])
    fig.update_yaxes(title_text="power (W)", secondary_y=True)
    return fig


def lactate_temp_curve(telemetry):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    if not telemetry or not telemetry.get("lactate"):
        return _empty("Lactate & core temperature")
    x = telemetry.get("distance") or telemetry.get("t")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=x, y=telemetry["lactate"], name="lactate (mmol/L)",
                             line=dict(color=_ACCENT2, width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=telemetry.get("core_temp", []), name="core temp (°C)",
                             line=dict(color="#ef4444", width=2)), secondary_y=True)
    fig.update_layout(template=_TEMPLATE, title="Blood lactate & core temperature",
                      margin=dict(l=45, r=45, t=40, b=35), legend=dict(orientation="h"))
    fig.update_yaxes(title_text="mmol/L", secondary_y=False)
    fig.update_yaxes(title_text="°C", secondary_y=True)
    return fig


def muscle_fatigue_heatmap(fatigue_timeline: Dict[str, List[float]], groups: List[str],
                           distance: Optional[List[float]] = None):
    """Heat-map of per-muscle-group fatigue over the run."""
    import plotly.graph_objects as go
    if not fatigue_timeline:
        return _empty("Muscle fatigue heat-map")
    z = np.array([fatigue_timeline[g] for g in groups])
    x = distance if distance is not None else list(range(z.shape[1]))
    fig = go.Figure(go.Heatmap(z=z, x=x, y=groups, colorscale="Inferno",
                               zmin=0, zmax=1, colorbar=dict(title="fatigue")))
    fig.update_layout(template=_TEMPLATE, title="Muscle-group fatigue over the run",
                      margin=dict(l=70, r=20, t=40, b=35))
    fig.update_xaxes(title_text="distance (m)")
    return fig


def distance_progress(rows: List[Dict[str, Any]], mile_m: float = 1609.344):
    """How far the AIs get each generation — the clearest 'are they improving?' view."""
    import plotly.graph_objects as go
    if not rows:
        return _empty("Furthest distance per generation")
    gens = [r["generation"] for r in rows]
    dist = [r.get("distance") or 0 for r in rows]
    fig = go.Figure(go.Scatter(x=gens, y=dist, mode="lines+markers", fill="tozeroy",
                               line=dict(color=_ACCENT, width=2),
                               marker=dict(size=5), name="furthest"))
    fig.add_hline(y=mile_m, line=dict(color=_ACCENT2, dash="dash"),
                  annotation_text="the mile (1609 m)",
                  annotation_font_color=_ACCENT2)
    fig.update_layout(template=_TEMPLATE, title="Furthest distance reached (m)",
                      margin=dict(l=45, r=15, t=40, b=30))
    fig.update_xaxes(title_text="generation")
    fig.update_yaxes(title_text="metres")
    return fig


def algo_comparison(by_algo: Dict[str, Dict[str, float]], metric: str = "best_fitness"):
    """Bar chart comparing RL algorithms (or architectures)."""
    import plotly.graph_objects as go
    if not by_algo:
        return _empty("Algorithm comparison")
    names = list(by_algo.keys())
    vals = [by_algo[n].get(metric) or 0 for n in names]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=_ACCENT))
    fig.update_layout(template=_TEMPLATE, title=f"By algorithm: {metric}",
                      margin=dict(l=45, r=20, t=40, b=35))
    return fig


def leaderboard_table(rows: List[Dict[str, Any]]):
    import plotly.graph_objects as go
    if not rows:
        return _empty("Leaderboard")
    header = ["#", "genome", "algo", "arch", "fitness", "mile (s)", "speed", "steps"]
    cells = [[], [], [], [], [], [], [], []]
    for r in rows:
        cells[0].append(r.get("rank"))
        cells[1].append(r.get("genome_id"))
        cells[2].append(r.get("algo"))
        cells[3].append(r.get("arch"))
        cells[4].append(f"{r.get('fitness', 0):.1f}")
        cells[5].append(f"{r.get('mile_time'):.1f}" if r.get("mile_time") else "-")
        cells[6].append(f"{r.get('mean_speed', 0):.2f}")
        cells[7].append(r.get("total_timesteps"))
    fig = go.Figure(go.Table(
        header=dict(values=header, fill_color="#1f2937", font=dict(color="white")),
        cells=dict(values=cells, fill_color="#111827", font=dict(color="#e5e7eb")),
    ))
    fig.update_layout(template=_TEMPLATE, margin=dict(l=5, r=5, t=25, b=5),
                      title="Leaderboard")
    return fig
