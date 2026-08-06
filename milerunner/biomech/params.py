"""Anthropometric and physiological parameters for a human body.

Default values describe the "average male" specified by the project brief
(175 cm, 77 kg). Segment mass fractions follow Dempster/Winter cadaver-derived
anthropometry; physiological constants follow standard exercise-physiology
references (ACSM, critical-power literature). None of these encode a *running
strategy* — they describe the body the agent must learn to move.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


# Segment mass as a fraction of total body mass (Winter, Biomechanics and
# Motor Control of Human Movement). Left/right limbs share the per-side value.
SEGMENT_MASS_FRACTION: Dict[str, float] = {
    "head_neck": 0.081,
    "trunk": 0.497,
    "upper_arm": 0.028,   # per side
    "forearm": 0.016,     # per side
    "hand": 0.006,        # per side
    "thigh": 0.100,       # per side
    "shank": 0.0465,      # per side
    "foot": 0.0145,       # per side
}

# Segment length as a fraction of standing height (Winter).
SEGMENT_LENGTH_FRACTION: Dict[str, float] = {
    "head_neck": 0.130,
    "trunk": 0.288,
    "upper_arm": 0.186,
    "forearm": 0.146,
    "hand": 0.108,
    "thigh": 0.245,
    "shank": 0.246,
    "foot": 0.152,
}


@dataclass
class BodyParams:
    """Physical + physiological description of a runner's body."""

    name: str = "average_male"
    height_m: float = 1.75
    mass_kg: float = 77.0
    age_years: float = 30.0
    sex: str = "male"

    # --- Aerobic system ---
    # VO2max in ml O2 per kg per minute. ~48 is average for a 30yo male.
    vo2max_ml_kg_min: float = 48.0
    vo2_rest_ml_kg_min: float = 3.5              # 1 MET
    vo2_time_constant_s: float = 25.0            # oxygen-uptake kinetics lag
    # Fraction of VO2max sustainable at lactate threshold.
    lactate_threshold_frac: float = 0.85

    # --- Anaerobic system (critical-power / W' model) ---
    # Critical speed: the highest speed sustainable (quasi) indefinitely. Must
    # sit below VO2max pace (~4.0 m/s for the default body); ~90% of it here.
    critical_speed_mps: float = 3.6
    # D': finite distance capacity available above critical speed (metres of
    # "anaerobic reserve"). Roughly analogous to W'.
    d_prime_m: float = 220.0
    # Tau for W'-balance recovery when running below critical speed (Skiba).
    w_prime_tau_s: float = 300.0

    # --- Cardiac ---
    hr_rest_bpm: float = 60.0
    hr_max_bpm: float = 190.0                    # ~ 220 - age
    hr_time_constant_s: float = 20.0

    # --- Thermoregulation ---
    core_temp_c: float = 37.0
    core_temp_critical_c: float = 40.0           # heat-stroke / forced stop
    heat_capacity_j_per_c: float = 3500.0 * 77.0 # ~3.5 kJ/(kg*C) * mass
    sweat_cooling_gain: float = 12.0             # W of cooling per (C over 37)

    # --- Running energetics ---
    # Metabolic cost of transport (J per kg per metre) on flat ground. ~3.6-4.2.
    cost_of_transport_j_kg_m: float = 3.9
    # Fraction of mechanical work recovered elastically by tendons at the body's
    # natural stride frequency. Off-frequency striding loses this bonus.
    tendon_elastic_return: float = 0.45
    tendon_natural_cadence_hz: float = 2.9       # ~174 steps/min, emergent optimum
    tendon_cadence_bandwidth_hz: float = 0.9

    # --- Muscle system: per-group peak isometric-equivalent torque budget ---
    # Values are peak joint torques (N*m) the group can contribute, scaled with
    # body mass. These bound the physics actuators.
    muscle_peak_torque: Dict[str, float] = field(default_factory=lambda: {
        "ankle": 140.0,      # calves / soleus-gastrocnemius + tibialis
        "knee": 240.0,       # quadriceps / hamstrings
        "hip": 180.0,        # glutes / hip flexors
        "trunk": 200.0,      # core
        "shoulder": 60.0,    # shoulders / arm swing drivers
        "elbow": 40.0,       # arms
        "neck": 20.0,        # neck stabilisation
    })
    # Local muscular endurance: seconds of maximal activation before a group is
    # fully fatigued, and the recovery time constant when rested.
    muscle_fatigue_tau_s: float = 55.0
    muscle_recovery_tau_s: float = 90.0

    def scaled(self) -> "BodyParams":
        """Return a copy with mass-dependent quantities rescaled to mass_kg.

        Allows callers to build a 60 kg or 90 kg runner and keep torques /
        heat capacity physically consistent without editing every field.
        """
        ref = 77.0
        scale = self.mass_kg / ref
        new = BodyParams(**{k: v for k, v in self.__dict__.items()})
        new.heat_capacity_j_per_c = 3500.0 * self.mass_kg
        new.muscle_peak_torque = {k: v * scale for k, v in self.muscle_peak_torque.items()}
        return new

    def segment_mass(self, segment: str) -> float:
        return SEGMENT_MASS_FRACTION[segment] * self.mass_kg

    def segment_length(self, segment: str) -> float:
        return SEGMENT_LENGTH_FRACTION[segment] * self.height_m

    def to_dict(self) -> Dict:
        d = dict(self.__dict__)
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "BodyParams":
        known = {k: v for k, v in data.items() if k in cls.__annotations__ or k in (
            "name", "height_m", "mass_kg", "age_years", "sex")}
        return cls(**known)
