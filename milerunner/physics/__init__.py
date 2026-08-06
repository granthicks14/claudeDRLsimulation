"""Physics: humanoid body model and track/atmosphere."""
from __future__ import annotations

from .body_builder import (actuated_joint_names, build_humanoid_mjcf,
                          joint_muscle_map)
from .track import MILE_M, LAP_M, Track, Weather

__all__ = ["build_humanoid_mjcf", "actuated_joint_names", "joint_muscle_map",
           "Track", "Weather", "MILE_M", "LAP_M"]

try:
    from .humanoid import Humanoid  # requires mujoco
    __all__.append("Humanoid")
except Exception:  # pragma: no cover
    pass
