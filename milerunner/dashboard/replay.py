"""Best-agent replay and 3D visualisation.

The "3D runner" is reconstructed from the humanoid's forward kinematics
(``mj_forward`` populates body world positions) — this needs **no** OpenGL, so
it works on any machine, headless servers included. The skeleton is drawn with
Plotly. When a GL backend *is* available (a desktop with a display or a GPU
server), :func:`render_mujoco_video` additionally exports a photorealistic
MuJoCo video.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Skeleton bone connectivity (parent body -> child body) for stick figure.
BONES = [
    ("pelvis", "torso"), ("torso", "head"),
    ("pelvis", "thigh_right"), ("thigh_right", "shank_right"), ("shank_right", "foot_right"),
    ("pelvis", "thigh_left"), ("thigh_left", "shank_left"), ("shank_left", "foot_left"),
    ("torso", "upper_arm_right"), ("upper_arm_right", "lower_arm_right"),
    ("torso", "upper_arm_left"), ("upper_arm_left", "lower_arm_left"),
]
BODIES = ["pelvis", "torso", "head", "thigh_right", "shank_right", "foot_right",
          "thigh_left", "shank_left", "foot_left", "upper_arm_right",
          "lower_arm_right", "upper_arm_left", "lower_arm_left"]


@dataclass
class Replay:
    frames: List[Dict[str, np.ndarray]] = field(default_factory=list)  # body -> xyz
    times: List[float] = field(default_factory=list)
    speeds: List[float] = field(default_factory=list)
    info: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "times": self.times,
            "speeds": self.speeds,
            "info": self.info,
            "frames": [{b: p.tolist() for b, p in f.items()} for f in self.frames],
        }


def record_replay(model, env, stride: int = 3, max_steps: int = 20000,
                  deterministic: bool = True) -> Replay:
    """Roll out a policy, capturing the skeleton every ``stride`` control steps."""
    import mujoco  # local import; only needed at replay time

    obs, info = env.reset()
    base_env = getattr(env, "unwrapped", env)
    hum = base_env.humanoid
    replay = Replay()
    done = False
    state = None
    step = 0
    while not done and step < max_steps:
        try:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
        except TypeError:  # pragma: no cover
            action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if step % stride == 0:
            frame = {b: hum.data.body(b).xpos.copy() for b in BODIES}
            replay.frames.append(frame)
            replay.times.append(info["time"])
            replay.speeds.append(info["speed"])
        step += 1
    replay.info = {
        "finished": bool(info.get("finished")),
        "finish_time": info.get("finish_time"),
        "distance": info.get("distance"),
        "peak_speed": info.get("peak_speed"),
    }
    return replay


def skeleton_figure(replay: Replay, frame_idx: int = -1):
    """A single Plotly 3D skeleton snapshot (for the dashboard)."""
    import plotly.graph_objects as go

    if not replay.frames:
        return go.Figure()
    f = replay.frames[frame_idx]
    edges_x, edges_y, edges_z = [], [], []
    for a, b in BONES:
        pa, pb = f[a], f[b]
        edges_x += [pa[0], pb[0], None]
        edges_y += [pa[1], pb[1], None]
        edges_z += [pa[2], pb[2], None]
    pts = np.array([f[b] for b in BODIES])
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=edges_x, y=edges_y, z=edges_z, mode="lines",
                               line=dict(width=8, color="#22d3ee"), name="bones"))
    fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                               marker=dict(size=5, color="#f59e0b"), name="joints"))
    cx = float(f["pelvis"][0])
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[cx - 1.2, cx + 1.2], title="x (m)"),
            yaxis=dict(range=[-1.2, 1.2], title="y (m)"),
            zaxis=dict(range=[0, 2.2], title="z (m)"),
            aspectmode="manual", aspectratio=dict(x=2.4, y=2.4, z=2.2),
        ),
        margin=dict(l=0, r=0, t=20, b=0), showlegend=False,
        template="plotly_dark",
    )
    return fig


def animated_skeleton_figure(replay: Replay, max_frames: int = 120):
    """An animated Plotly 3D replay of the run (play button)."""
    import plotly.graph_objects as go

    frames = replay.frames
    if not frames:
        return go.Figure()
    idxs = np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).astype(int)

    def make_traces(f):
        ex, ey, ez = [], [], []
        for a, b in BONES:
            pa, pb = f[a], f[b]
            ex += [pa[0], pb[0], None]
            ey += [pa[1], pb[1], None]
            ez += [pa[2], pb[2], None]
        pts = np.array([f[b] for b in BODIES])
        return [
            go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                         line=dict(width=8, color="#22d3ee")),
            go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                         marker=dict(size=5, color="#f59e0b")),
        ]

    fig = go.Figure(data=make_traces(frames[idxs[0]]))
    plotly_frames = []
    for i in idxs:
        f = frames[i]
        cx = float(f["pelvis"][0])
        plotly_frames.append(go.Frame(
            data=make_traces(f),
            layout=go.Layout(scene=dict(
                xaxis=dict(range=[cx - 1.4, cx + 1.4]))),
        ))
    fig.frames = plotly_frames
    fig.update_layout(
        template="plotly_dark", showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(yaxis=dict(range=[-1.2, 1.2]), zaxis=dict(range=[0, 2.2]),
                   aspectmode="manual", aspectratio=dict(x=2.6, y=2.4, z=2.2)),
        updatemenus=[dict(type="buttons", showactive=False, y=1, x=0.05,
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=40, redraw=True),
                                                         fromcurrent=True)])])],
    )
    return fig


def render_mujoco_video(model, env, out_path: str, stride: int = 2,
                        width: int = 640, height: int = 480, fps: int = 50) -> Optional[str]:
    """Export a photorealistic MuJoCo video. Requires a working GL backend.

    Returns the output path on success, ``None`` if no renderer is available.
    """
    try:
        import mujoco
        import imageio  # optional
    except Exception:  # pragma: no cover
        return None
    base_env = getattr(env, "unwrapped", env)
    hum = base_env.humanoid
    try:
        renderer = mujoco.Renderer(hum.model, height=height, width=width)
    except Exception:  # pragma: no cover - no GL backend
        return None
    obs, info = env.reset()
    frames, done, step, state = [], False, 0, None
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 4.0
    cam.elevation = -12
    while not done and step < 40000:
        try:
            action, state = model.predict(obs, state=state, deterministic=True)
        except TypeError:  # pragma: no cover
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if step % stride == 0:
            cam.lookat[:] = hum.data.body("pelvis").xpos
            renderer.update_scene(hum.data, camera=cam)
            frames.append(renderer.render())
        step += 1
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    return out_path
