"""CPU runner video — draws the AI running as an actual video file, no OpenGL.

The photorealistic MuJoCo render needs an OpenGL backend (OSMesa/EGL) which
isn't always available. This module renders the runner from the captured body
positions with **matplotlib** (pure CPU) and writes an MP4 that autoplays in the
dashboard's video box. It always works — so you can watch the runner on any free
host, no GPU and no system GL libraries required.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

# Bones for the side-view stick/'body' figure (forward = x, height = z).
_BONES = [
    ("pelvis", "torso"), ("torso", "head"),
    ("pelvis", "thigh_right"), ("thigh_right", "shank_right"), ("shank_right", "foot_right"),
    ("pelvis", "thigh_left"), ("thigh_left", "shank_left"), ("shank_left", "foot_left"),
    ("torso", "upper_arm_right"), ("upper_arm_right", "lower_arm_right"),
    ("torso", "upper_arm_left"), ("upper_arm_left", "lower_arm_left"),
]


def render_run_video(telemetry: dict, out_path: str, width: int = 480,
                     height: int = 270, fps: int = 15, max_frames: int = 90,
                     window_m: float = 3.2) -> Optional[str]:
    """Render the side-view runner to an MP4 (or GIF) with matplotlib. No GL.

    Returns the output path, or ``None`` if there are no skeleton frames yet.
    """
    frames = telemetry.get("skeleton") or []
    if not frames:
        return None
    times = telemetry.get("skeleton_t") or list(range(len(frames)))
    dist = telemetry.get("distance") or []

    # Use the object-oriented Agg API (NOT pyplot) — this is thread-safe, so it
    # works in the dashboard's background render thread.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    # Sub-sample to keep rendering quick on a free CPU.
    idx = np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).astype(int)
    dpi = 100.0
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    half = window_m / 2.0
    imgs: List[np.ndarray] = []

    for k in idx:
        f = frames[k]
        cx = float(f["pelvis"][0])
        ax.clear()
        ax.set_facecolor("#0b1220")
        ax.set_xlim(-half, half)
        ax.set_ylim(-0.25, 2.15)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        # ground + distance ticks scrolling past
        ax.axhline(0.0, color="#6b7280", linewidth=2)
        lo, hi = int(np.floor(cx - half)), int(np.ceil(cx + half))
        for m in range(lo, hi + 1):
            ax.plot([m - cx, m - cx], [0.0, -0.12], color="#4b5563", linewidth=1)
        # body limbs
        for a, b in _BONES:
            pa, pb = f[a], f[b]
            ax.plot([pa[0] - cx, pb[0] - cx], [pa[2], pb[2]],
                    color="#22d3ee", linewidth=4, solid_capstyle="round")
        # head
        hx, hz = float(f["head"][0] - cx), float(f["head"][2])
        ax.plot([hx], [hz], marker="o", markersize=10, color="#f59e0b")
        d = dist[k] if k < len(dist) else cx
        t = times[k] if k < len(times) else 0.0
        ax.text(0.02, 0.94, f"{d:5.0f} m   {t:4.0f} s", transform=ax.transAxes,
                color="#e5e7eb", fontsize=9, family="monospace", va="top")

        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        img = buf.reshape(canvas.get_width_height()[::-1] + (4,))[..., :3]
        imgs.append(img.copy())

    if not imgs:
        return None

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    import imageio.v2 as imageio
    try:
        imageio.mimsave(out_path, imgs, fps=fps, macro_block_size=None)
        return out_path
    except Exception:
        gif = os.path.splitext(out_path)[0] + ".gif"
        imageio.mimsave(gif, imgs, duration=1.0 / fps)
        return gif
