"""Track and atmospheric environment.

Models the standard 400 m oval, air density as a function of temperature and
altitude, aerodynamic drag (headwind/tailwind), and reports where on the track
a runner is (used for lap/mile bookkeeping and lane-curvature effects).

For the biomechanics we run the humanoid on a straight treadmill of ground and
fold the track's curvature into a small extra energetic cost, which keeps the
physics simulation fast and stable while still accounting for the ~2.5% cost of
running the bends of an oval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MILE_M = 1609.344
LAP_M = 400.0


@dataclass
class Weather:
    temperature_c: float = 15.0
    wind_mps: float = 0.0          # + = tailwind, - = headwind (along running dir)
    humidity: float = 0.5
    altitude_m: float = 0.0
    pressure_kpa: float = 101.325

    def air_density(self) -> float:
        """Air density (kg/m^3) from temperature, pressure and altitude."""
        # Barometric pressure drop with altitude.
        p0 = self.pressure_kpa * 1000.0
        pressure = p0 * np.exp(-self.altitude_m / 8434.0)
        t_kelvin = self.temperature_c + 273.15
        # Ideal gas with humidity correction (dry-air approximation dominates).
        return pressure / (287.05 * t_kelvin)

    def vo2max_altitude_factor(self) -> float:
        """VO2max declines ~6-7% per 1000 m above ~1500 m altitude."""
        if self.altitude_m <= 1500.0:
            return 1.0
        return max(0.6, 1.0 - 0.065 * (self.altitude_m - 1500.0) / 1000.0)


class Track:
    """400 m oval bookkeeping + aerodynamic drag model."""

    DRAG_AREA = 0.45          # frontal area (m^2), average adult
    DRAG_CD = 0.9             # drag coefficient of a runner

    def __init__(self, weather: Weather | None = None, distance_m: float = MILE_M,
                 lane: int = 1):
        self.weather = weather or Weather()
        self.distance_m = distance_m
        self.lane = lane

    # ------------------------------------------------------------------ #
    def drag_force(self, runner_speed_mps: float) -> float:
        """Aerodynamic drag force (N) opposing motion.

        Relative air speed accounts for wind: a headwind (negative wind)
        increases relative speed and therefore drag.
        """
        rho = self.weather.air_density()
        rel = runner_speed_mps - self.weather.wind_mps
        return 0.5 * rho * self.DRAG_CD * self.DRAG_AREA * rel * abs(rel)

    def drag_power(self, runner_speed_mps: float) -> float:
        """Metabolic-equivalent power (W) needed to overcome drag."""
        return self.drag_force(runner_speed_mps) * max(runner_speed_mps, 0.0)

    def curve_cost_factor(self, distance_covered_m: float) -> float:
        """Extra cost multiplier from running the oval's two bends.

        Roughly half of each lap is curve; running a curve costs a little more
        (~2-3%) due to centripetal demand, more in inner lanes.
        """
        pos = distance_covered_m % LAP_M
        on_curve = (100.0 <= pos < 200.0) or (300.0 <= pos < LAP_M)
        if on_curve:
            lane_penalty = 1.0 + 0.004 * (self.lane - 1)
            return 1.025 * lane_penalty
        return 1.0

    def laps(self) -> float:
        return self.distance_m / LAP_M
