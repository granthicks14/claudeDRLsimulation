"""400 m oval track view — watch the runner go around the track.

Maps the runner's distance-along-the-mile onto a standard 400 m oval and draws
an animated icon travelling around it (≈4 laps for a mile), with a trail, lap
counter and a start/finish line. This is the "watch it like a real mile" view.

The geometry is a standard track: two 84.39 m straights joined by two
semicircular bends, ≈400 m around.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

STRAIGHT = 84.39          # length of each straight (m)
RADIUS = 36.80            # bend radius (m) -> total perimeter ~400 m
LAP = 2 * STRAIGHT + 2 * math.pi * RADIUS   # ~400.0 m
_HALF = STRAIGHT / 2.0
_ARC = math.pi * RADIUS   # length of one semicircle


def track_position(distance_m: float) -> Tuple[float, float]:
    """Map metres run to an (x, y) point on the oval (start = home straight)."""
    s = distance_m % LAP
    # 1) bottom (home) straight: left -> right
    if s <= STRAIGHT:
        return (-_HALF + s, -RADIUS)
    s -= STRAIGHT
    # 2) right bend: bottom -> top (theta -90° -> +90°)
    if s <= _ARC:
        theta = -math.pi / 2 + s / RADIUS
        return (_HALF + RADIUS * math.cos(theta), RADIUS * math.sin(theta))
    s -= _ARC
    # 3) top (back) straight: right -> left
    if s <= STRAIGHT:
        return (_HALF - s, RADIUS)
    s -= STRAIGHT
    # 4) left bend: top -> bottom (theta +90° -> +270°)
    theta = math.pi / 2 + s / RADIUS
    return (-_HALF + RADIUS * math.cos(theta), RADIUS * math.sin(theta))


def _oval_outline(n: int = 240) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for i in range(n + 1):
        x, y = track_position(LAP * i / n)
        xs.append(x)
        ys.append(y)
    return xs, ys


def _base_traces(distances: List[float], go):
    """Static track outline + lane fill + start/finish line."""
    ox, oy = _oval_outline()
    outer_x, outer_y = _oval_outline_scaled(1.16)
    traces = [
        go.Scatter(x=outer_x, y=outer_y, mode="lines",
                   line=dict(color="#b45309", width=1), hoverinfo="skip", showlegend=False),
        go.Scatter(x=ox, y=oy, mode="lines",
                   line=dict(color="#9ca3af", width=2, dash="dot"),
                   hoverinfo="skip", showlegend=False),
    ]
    # start/finish line (at the start of the home straight)
    sx, sy = track_position(0.0)
    traces.append(go.Scatter(x=[sx, sx], y=[sy - 4, sy + 4], mode="lines",
                             line=dict(color="#ffffff", width=3),
                             hoverinfo="skip", showlegend=False))
    return traces


def _oval_outline_scaled(scale: float, n: int = 240):
    xs, ys = [], []
    for i in range(n + 1):
        x, y = track_position(LAP * i / n)
        xs.append(x * scale)
        ys.append(y * scale)
    return xs, ys


def track_figure(telemetry: Dict, max_frames: int = 120):
    """Animated oval with the runner's icon; press ▶ to watch the mile."""
    import plotly.graph_objects as go

    dist = telemetry.get("distance") or []
    times = telemetry.get("t") or []
    if not dist:
        fig = go.Figure(_base_traces([], go))
        _layout(fig, go)
        fig.add_annotation(text="waiting for the runner…", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(color="#9ca3af"))
        return fig

    n = len(dist)
    idx = np.linspace(0, n - 1, min(max_frames, n)).astype(int)

    def runner_trace(k):
        d = dist[k]
        x, y = track_position(d)
        # trail: last part of the path already covered
        lo = max(0, k - 18)
        tx, ty = zip(*[track_position(dist[j]) for j in range(lo, k + 1)]) if k > 0 else ([x], [y])
        return [
            go.Scatter(x=list(tx), y=list(ty), mode="lines",
                       line=dict(color="#22d3ee", width=4), hoverinfo="skip",
                       showlegend=False),
            go.Scatter(x=[x], y=[y], mode="markers+text", text=["🏃"],
                       textposition="middle center", textfont=dict(size=22),
                       marker=dict(size=16, color="#f59e0b"),
                       hoverinfo="skip", showlegend=False),
        ]

    base = _base_traces(dist, go)
    fig = go.Figure(data=base + runner_trace(idx[0]))
    frames = []
    for k in idx:
        lap = int(dist[k] // LAP) + 1
        t = times[k] if k < len(times) else 0
        frames.append(go.Frame(
            data=base + runner_trace(k),
            layout=go.Layout(title=f"🏁 Lap {min(lap,4)}/4  ·  {dist[k]:.0f} m  ·  {t:.0f} s"),
        ))
    fig.frames = frames
    _layout(fig, go)
    fig.update_layout(
        title=f"🏁 Lap 1/4  ·  {dist[idx[0]]:.0f} m",
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.05,
                          xanchor="left",
                          buttons=[dict(label="▶ Watch the mile", method="animate",
                                        args=[None, dict(frame=dict(duration=60, redraw=True),
                                                         fromcurrent=True)])])],
    )
    return fig


_RACE_COLORS = ["#22d3ee", "#f59e0b", "#a78bfa", "#34d399", "#ef4444",
                "#f472b6", "#facc15", "#60a5fa", "#fb923c", "#4ade80"]


def _lane_pos(distance_m: float, lane: int) -> Tuple[float, float]:
    """Position on a concentric lane (outer lanes drawn slightly wider)."""
    x, y = track_position(distance_m)
    s = 1.0 + lane * 0.055
    return x * s, y * s


def race_figure(race: List[Dict], max_frames: int = 90):
    """Animate the whole population racing the mile — one runner per lane.

    ``race`` is a list of per-agent dicts with ``times`` and ``distances``. Each
    agent gets its own lane and colour; press ▶ to watch them race. They all
    learn together (population-based training shares discoveries via evolution),
    and this shows the squad's progress side by side.
    """
    import plotly.graph_objects as go

    race = [r for r in (race or []) if r.get("distances")][:len(_RACE_COLORS)]
    if not race:
        fig = go.Figure(_base_traces([], go))
        _layout(fig, go)
        fig.update_layout(title="🏁 The squad racing")
        fig.add_annotation(text="waiting for the runners…", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(color="#9ca3af"))
        return fig

    max_t = max((r["times"][-1] if r.get("times") else 0.0) for r in race) or 1.0
    tg = np.linspace(0, max_t, max_frames)
    series = []
    for r in race:
        t = np.asarray(r.get("times") or [0.0], dtype=float)
        d = np.asarray(r.get("distances") or [0.0], dtype=float)
        if len(t) < 2:
            series.append(np.full(max_frames, float(d[-1]) if len(d) else 0.0))
        else:
            series.append(np.interp(tg, t, d, right=float(d[-1])))

    # faint concentric lane outlines
    base = []
    for i in range(len(race)):
        lx, ly = _oval_outline_scaled(1.0 + i * 0.055)
        base.append(go.Scatter(x=lx, y=ly, mode="lines",
                               line=dict(color="#374151", width=1),
                               hoverinfo="skip", showlegend=False))
    sx, sy = track_position(0.0)
    base.append(go.Scatter(x=[sx, sx * 1.6], y=[sy, sy * 1.6], mode="lines",
                          line=dict(color="#ffffff", width=2), hoverinfo="skip",
                          showlegend=False))

    def frame_traces(k):
        traces = list(base)
        for i, dd in enumerate(series):
            x, y = _lane_pos(dd[k], i)
            traces.append(go.Scatter(
                x=[x], y=[y], mode="markers",
                marker=dict(size=13, color=_RACE_COLORS[i % len(_RACE_COLORS)],
                            line=dict(color="#0b1220", width=1)),
                hoverinfo="skip", showlegend=False))
        return traces

    fig = go.Figure(data=frame_traces(0))
    frames = []
    for k in range(max_frames):
        leader = max(s[k] for s in series)
        frames.append(go.Frame(
            data=frame_traces(k),
            layout=go.Layout(title=f"🏁 The squad racing  ·  {tg[k]:.0f} s  ·  "
                                   f"leader {leader:.0f} m ({len(race)} runners)")))
    fig.frames = frames
    _layout(fig, go)
    fig.update_layout(
        title=f"🏁 The squad racing  ·  {len(race)} runners",
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.05,
                          xanchor="left",
                          buttons=[dict(label="▶ Race", method="animate",
                                        args=[None, dict(frame=dict(duration=55, redraw=True),
                                                         fromcurrent=True)])])],
    )
    return fig


def _layout(fig, go):
    fig.update_layout(
        template="plotly_dark", showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1,
                   range=[-(_HALF + RADIUS) * 1.3, (_HALF + RADIUS) * 1.3]),
        yaxis=dict(visible=False, range=[-RADIUS * 1.5, RADIUS * 1.5]),
        plot_bgcolor="#0b1220",
    )
