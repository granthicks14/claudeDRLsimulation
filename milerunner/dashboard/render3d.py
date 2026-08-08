"""Photorealistic MuJoCo rendering of the best runner — auto-enabled on a GPU.

MuJoCo's offscreen renderer needs an OpenGL backend (EGL on a headless GPU, e.g.
Colab's GPU runtime). This module detects whether one is available and, if so,
replays the best agent's recorded ``qpos`` frames through the humanoid model and
writes a real 3D video. Without a GPU it returns ``None`` and the dashboard just
uses the always-available Plotly views.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

_GL_CHECK: Optional[bool] = None


def gl_available() -> bool:
    """True if a MuJoCo OpenGL renderer can be created (cached)."""
    global _GL_CHECK
    if _GL_CHECK is not None:
        return _GL_CHECK
    ok = False
    try:
        import mujoco

        from ..biomech.params import BodyParams
        from ..physics.body_builder import build_humanoid_mjcf
        model = mujoco.MjModel.from_xml_string(build_humanoid_mjcf(BodyParams()))
        renderer = mujoco.Renderer(model, height=64, width=64)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        renderer.render()
        ok = True
    except Exception:
        ok = False
    _GL_CHECK = ok
    return ok


def render_best_video(body, telemetry: dict, out_path: str,
                      width: int = 512, height: int = 384, fps: int = 30) -> Optional[str]:
    """Render the recorded ``qpos`` frames to an mp4. Returns the path or None.

    Requires a GL backend (see :func:`gl_available`) and ``imageio``. The camera
    tracks the runner from the side, like the AI-learns-to-run videos.
    """
    frames_q: List[list] = telemetry.get("qpos") or []
    if not frames_q:
        return None
    try:
        import imageio.v2 as imageio
        import mujoco

        from ..physics.humanoid import Humanoid
    except Exception:
        return None
    try:
        hum = Humanoid(body)
        renderer = mujoco.Renderer(hum.model, height=height, width=width)
    except Exception:
        return None

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 4.5
    cam.elevation = -8
    cam.azimuth = 90            # side-on view
    nq = hum.model.nq
    imgs = []
    for q in frames_q:
        q = np.asarray(q, dtype=float)
        if q.shape[0] != nq:
            continue
        hum.data.qpos[:] = q
        hum.data.qvel[:] = 0.0
        mujoco.mj_forward(hum.model, hum.data)
        cam.lookat[:] = hum.data.body("pelvis").xpos
        renderer.update_scene(hum.data, camera=cam)
        imgs.append(renderer.render())
    if not imgs:
        return None
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        imageio.mimsave(out_path, imgs, fps=fps)
    except Exception:
        # mp4 needs imageio-ffmpeg; fall back to an animated GIF.
        gif = os.path.splitext(out_path)[0] + ".gif"
        imageio.mimsave(gif, imgs, duration=1.0 / fps)
        return gif
    return out_path
