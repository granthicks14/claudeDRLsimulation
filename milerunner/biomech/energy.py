"""Physiological energy system.

This module turns *effort* (running speed and muscular power demand) into the
physiological state variables the agent must manage: oxygen uptake, heart rate,
the anaerobic ``W'`` (D') reserve, blood lactate, glycogen and core temperature.

The backbone is the **critical-power / W'-balance model**, which is the best
validated framework for the power–duration relationship and therefore for
pacing. Above critical speed the finite ``D'`` reserve is spent; below it, the
reserve recovers with a time constant. Exhaustion happens when the reserve is
gone. Crucially, the model contains *no* pacing policy — it only says what the
body permits. The agent must discover how to spend the reserve.

References: Monod & Scherrer (1965); Skiba et al. (2012) W'-balance;
ACSM metabolic equations; Margaria running energetics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import BodyParams

# Aerobic energy yield per litre of O2 (~5 kcal, RER dependent). Joules.
O2_ENERGY_J_PER_ML = 20.9  # J per ml O2


@dataclass
class EnergyState:
    vo2_ml_kg_min: float           # instantaneous oxygen uptake
    vo2_demand_ml_kg_min: float    # oxygen the current effort *asks* for
    heart_rate_bpm: float
    w_prime_balance_j: float       # remaining anaerobic reserve (Joules)
    w_prime_frac: float            # 0..1 fraction remaining
    blood_lactate_mmol_l: float
    glycogen_frac: float           # 0..1 remaining muscle+liver glycogen
    core_temp_c: float
    exhausted: bool
    metabolic_power_w: float

    def as_array(self) -> np.ndarray:
        return np.array([
            self.vo2_ml_kg_min,
            self.vo2_demand_ml_kg_min,
            self.heart_rate_bpm,
            self.w_prime_frac,
            self.blood_lactate_mmol_l,
            self.glycogen_frac,
            self.core_temp_c,
            float(self.exhausted),
        ], dtype=np.float64)


class EnergySystem:
    """Stateful, time-stepped human bioenergetic model for one runner."""

    def __init__(self, params: BodyParams):
        self.p = params
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        p = self.p
        self.vo2 = p.vo2_rest_ml_kg_min
        self.hr = p.hr_rest_bpm
        # Anaerobic reserve tracked in metres of "distance above critical speed"
        # (classic D' model). This keeps the critical-power relationship exact.
        self.d_balance = p.d_prime_m
        self.lactate = 1.0                # resting blood lactate mmol/L
        self.glycogen = 1.0
        self.core_temp = p.core_temp_c
        self.exhausted = False
        self._last_metabolic_power = 0.0
        self._time_s = 0.0

    # ------------------------------------------------------------------ #
    def _d_prime_energy_j(self) -> float:
        """Convert the D' distance reserve to an energy budget in Joules.

        D' is a distance above critical speed; multiplying by the metabolic
        cost of transport and body mass yields the anaerobic energy store.
        """
        return self.p.d_prime_m * self.p.cost_of_transport_j_kg_m * self.p.mass_kg

    def vo2max_power_w(self) -> float:
        net_vo2 = (self.p.vo2max_ml_kg_min - self.p.vo2_rest_ml_kg_min)
        return net_vo2 * self.p.mass_kg / 60.0 * O2_ENERGY_J_PER_ML

    def vvo2max_speed(self) -> float:
        """Running speed whose aerobic demand equals VO2max (m/s)."""
        return self.vo2max_power_w() / max(self.p.cost_of_transport_j_kg_m * self.p.mass_kg, 1e-6)

    def effective_critical_speed(self) -> float:
        """Critical speed, guarded to stay physiologically below VO2max pace.

        Critical/lactate-threshold speed must sit below the speed at VO2max; a
        mis-specified config that violates this is clamped to 95% of vVO2max so
        the model can never make supra-maximal running "sustainable".
        """
        return float(min(self.p.critical_speed_mps, 0.95 * self.vvo2max_speed()))

    def critical_power_w(self) -> float:
        """Metabolic power at critical speed (W)."""
        return self.effective_critical_speed() * self.p.cost_of_transport_j_kg_m * self.p.mass_kg

    # ------------------------------------------------------------------ #
    def rest_power_w(self) -> float:
        return self.p.vo2_rest_ml_kg_min * self.p.mass_kg / 60.0 * O2_ENERGY_J_PER_ML

    def metabolic_demand(self, speed_mps: float, muscular_power_w: float,
                         grade: float = 0.0, cost_mult: float = 1.0) -> float:
        """Total metabolic power demanded (W) by locomotion + muscular work.

        The baseline cost of transport already includes normal tendon recoil;
        ``cost_mult`` (>=1) applies an *economy penalty* for running off the
        body's natural cadence (see :meth:`step`). ``muscular_power_w`` carries
        environment surcharges (drag, curve, wasted co-contraction).
        """
        p = self.p
        locomotion = p.cost_of_transport_j_kg_m * p.mass_kg * max(speed_mps, 0.0) * cost_mult
        grade_cost = 9.81 * p.mass_kg * grade * max(speed_mps, 0.0)  # PE against gravity
        extra = max(muscular_power_w, 0.0) * 0.25   # inefficiency surcharge
        return locomotion + max(grade_cost, 0.0) + self.rest_power_w() + extra

    # ------------------------------------------------------------------ #
    def step(self, dt: float, speed_mps: float, muscular_power_w: float = 0.0,
             cadence_hz: float = 2.9, ambient_temp_c: float = 15.0,
             grade: float = 0.0, breathing: float | None = None,
             vo2max_factor: float = 1.0, humidity: float = 0.5) -> EnergyState:
        """Advance the physiology by ``dt`` seconds and return the new state.

        ``breathing`` (0..1), when supplied, is the agent's respiratory effort.
        Oxygen delivery is best when breathing matches metabolic demand;
        breathing too little starves the aerobic system, breathing too hard
        wastes energy on respiratory muscles. The optimum is therefore *not*
        given to the agent — it must be discovered. ``vo2max_factor`` lets the
        environment apply altitude effects.
        """
        p = self.p
        self._time_s += dt
        vo2max = p.vo2max_ml_kg_min * vo2max_factor

        # --- Cadence economy: the baseline cost of transport already includes
        #     normal tendon recoil; running *off* the body's natural cadence
        #     loses that recoil and costs more. So this is a penalty for
        #     bad cadence, not a bonus — which is what lets the agent discover
        #     the optimal cadence without double-counting economy. ---
        band = p.tendon_cadence_bandwidth_hz
        cadence_eff = np.exp(-0.5 * ((cadence_hz - p.tendon_natural_cadence_hz) / band) ** 2)
        cost_mult = 1.0 + p.tendon_elastic_return * (1.0 - cadence_eff)
        metabolic_power = self.metabolic_demand(speed_mps, muscular_power_w, grade,
                                                cost_mult=cost_mult)

        # --- Oxygen-uptake kinetics: VO2 chases demand with a lag, capped. ---
        demand_vo2 = metabolic_power / (p.mass_kg * O2_ENERGY_J_PER_ML) * 60.0
        demand_vo2 = max(demand_vo2, p.vo2_rest_ml_kg_min)
        target_vo2 = min(demand_vo2, vo2max)

        # --- Breathing: delivery efficiency + respiratory cost. ---
        if breathing is not None:
            intensity_frac = float(np.clip(
                (demand_vo2 - p.vo2_rest_ml_kg_min) /
                max(vo2max - p.vo2_rest_ml_kg_min, 1e-6), 0.0, 1.0))
            delivery_eff = float(np.clip(1.0 - 0.6 * (breathing - intensity_frac) ** 2, 0.45, 1.0))
            target_vo2 *= delivery_eff
            respiratory_cost = 10.0 * (breathing ** 2) * (p.mass_kg / 77.0)
            metabolic_power += respiratory_cost
        self._last_metabolic_power = metabolic_power

        self.vo2 += (target_vo2 - self.vo2) * (1.0 - np.exp(-dt / p.vo2_time_constant_s))

        # --- Anaerobic reserve in distance space (critical-speed / D' model).
        #     Convert the locomotor demand to the equivalent flat-running speed
        #     it represents; any excess over critical speed drains D', a deficit
        #     recovers it. Time-to-exhaustion above CS is then exactly
        #     D' / (v_equiv - CS), the classic hyperbolic power-duration law. ---
        cost_per_m = p.cost_of_transport_j_kg_m * p.mass_kg
        locomotor_power = max(metabolic_power - self.rest_power_w(), 0.0)
        equiv_speed = locomotor_power / max(cost_per_m, 1e-6)
        deficit_speed = equiv_speed - self.effective_critical_speed()
        if deficit_speed > 0:
            self.d_balance -= deficit_speed * dt
        else:
            self.d_balance += (p.d_prime_m - self.d_balance) * (1.0 - np.exp(-dt / p.w_prime_tau_s))
        self.d_balance = float(np.clip(self.d_balance, 0.0, p.d_prime_m))
        if self.d_balance <= 0.0 and deficit_speed > 0:
            self.exhausted = True

        # --- Blood lactate accumulates above the lactate-threshold power. ---
        lt_power = p.lactate_threshold_frac * self.vo2max_power_w()
        over = max(metabolic_power - self.rest_power_w() - lt_power, 0.0)
        lactate_production = over / max(self.vo2max_power_w(), 1.0) * 3.5
        lactate_clearance = 0.7 * max(self.lactate - 1.0, 0.0)
        self.lactate += (lactate_production - lactate_clearance) * dt
        self.lactate = float(np.clip(self.lactate, 0.6, 25.0))

        # --- Heart rate follows metabolic intensity with a lag. ---
        intensity = np.clip((self.vo2 - p.vo2_rest_ml_kg_min) /
                            max(p.vo2max_ml_kg_min - p.vo2_rest_ml_kg_min, 1e-6), 0.0, 1.2)
        # Cardiac drift: rising core temperature and lactate push HR up.
        drift = 6.0 * max(self.core_temp - 37.0, 0.0) + 1.2 * max(self.lactate - 2.0, 0.0)
        target_hr = p.hr_rest_bpm + intensity * (p.hr_max_bpm - p.hr_rest_bpm) + drift
        target_hr = min(target_hr, p.hr_max_bpm + 5.0)
        self.hr += (target_hr - self.hr) * (1.0 - np.exp(-dt / p.hr_time_constant_s))

        # --- Glycogen depletion scales with carbohydrate-fuelled power. ---
        glyc_burn = metabolic_power * (0.5 + 0.5 * intensity) * dt
        glyc_capacity_j = 7500.0 * 1000.0  # ~7500 kJ usable glycogen store
        self.glycogen -= glyc_burn / glyc_capacity_j
        self.glycogen = float(np.clip(self.glycogen, 0.0, 1.0))

        # --- Thermoregulation: metabolic heat in vs convection + evaporation.
        #     ~80% of metabolic power is heat. Convective/radiative loss scales
        #     with the skin-to-air gradient; evaporative (sweat) loss is the
        #     dominant term for a running human but is throttled by humidity. ---
        heat_in = 0.80 * metabolic_power
        convection = 14.0 * max(self.core_temp - ambient_temp_c, 0.0)
        evap_gain = 300.0 * (1.0 - 0.6 * float(np.clip(humidity, 0.0, 1.0)))
        evaporation = min(evap_gain * max(self.core_temp - 36.8, 0.0), 900.0)
        net_heat = heat_in - convection - evaporation
        self.core_temp += net_heat * dt / p.heat_capacity_j_per_c
        self.core_temp = float(np.clip(self.core_temp, 36.0, 42.0))
        if self.core_temp >= p.core_temp_critical_c:
            self.exhausted = True
        if self.glycogen <= 0.0:
            self.exhausted = True

        return self.state()

    # ------------------------------------------------------------------ #
    def state(self) -> EnergyState:
        d_prime = max(self.p.d_prime_m, 1e-6)
        frac = self.d_balance / d_prime
        return EnergyState(
            vo2_ml_kg_min=self.vo2,
            vo2_demand_ml_kg_min=self._last_metabolic_power /
                (self.p.mass_kg * O2_ENERGY_J_PER_ML) * 60.0,
            heart_rate_bpm=self.hr,
            w_prime_balance_j=self.d_balance * self.p.cost_of_transport_j_kg_m * self.p.mass_kg,
            w_prime_frac=float(np.clip(frac, 0.0, 1.0)),
            blood_lactate_mmol_l=self.lactate,
            glycogen_frac=self.glycogen,
            core_temp_c=self.core_temp,
            exhausted=self.exhausted,
            metabolic_power_w=self._last_metabolic_power,
        )
