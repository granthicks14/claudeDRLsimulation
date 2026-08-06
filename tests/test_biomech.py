"""Tests for the biomechanical energy and muscle models."""
import numpy as np
import pytest

from milerunner.biomech.energy import EnergySystem
from milerunner.biomech.muscles import MUSCLE_GROUPS, MuscleSystem
from milerunner.biomech.params import (SEGMENT_MASS_FRACTION, BodyParams)


def test_segment_mass_fractions_sum_to_one():
    # head + trunk + 2*(each paired segment) == whole body
    total = (SEGMENT_MASS_FRACTION["head_neck"] + SEGMENT_MASS_FRACTION["trunk"]
             + 2 * (SEGMENT_MASS_FRACTION["upper_arm"] + SEGMENT_MASS_FRACTION["forearm"]
                    + SEGMENT_MASS_FRACTION["hand"] + SEGMENT_MASS_FRACTION["thigh"]
                    + SEGMENT_MASS_FRACTION["shank"] + SEGMENT_MASS_FRACTION["foot"]))
    assert total == pytest.approx(1.0, abs=1e-6)


def test_body_scaling_preserves_consistency():
    b = BodyParams(mass_kg=90.0).scaled()
    assert b.heat_capacity_j_per_c == pytest.approx(3500 * 90.0)
    # torques scale up with mass
    assert b.muscle_peak_torque["knee"] > BodyParams().muscle_peak_torque["knee"]


def test_exhaustion_above_critical_speed():
    p = BodyParams()
    es = EnergySystem(p)
    # Running well above critical speed must exhaust the W' reserve in finite time.
    t = 0.0
    while not es.exhausted and t < 3000:
        es.step(0.1, speed_mps=p.critical_speed_mps + 1.5,
                cadence_hz=p.tendon_natural_cadence_hz)
        t += 0.1
    assert es.exhausted
    assert 0 < t < 600  # plausible time-to-exhaustion window


def test_faster_pace_exhausts_sooner():
    p = BodyParams()

    def ttf(speed):
        es = EnergySystem(p)
        t = 0.0
        while not es.exhausted and t < 5000:
            es.step(0.1, speed_mps=speed, cadence_hz=p.tendon_natural_cadence_hz)
            t += 0.1
        return t

    fast = ttf(p.critical_speed_mps + 1.5)
    slow = ttf(p.critical_speed_mps + 0.5)
    assert fast < slow  # harder effort -> earlier exhaustion


def test_below_critical_speed_is_sustainable():
    p = BodyParams()
    es = EnergySystem(p)
    t = 0.0
    while not es.exhausted and t < 1800:
        es.step(0.2, speed_mps=p.critical_speed_mps - 0.5,
                cadence_hz=p.tendon_natural_cadence_hz)
        t += 0.2
    assert not es.exhausted  # easy pace should not exhaust W'


def test_heart_rate_rises_with_effort():
    p = BodyParams()
    es = EnergySystem(p)
    hr0 = es.hr
    for _ in range(300):
        es.step(0.1, speed_mps=p.critical_speed_mps, cadence_hz=p.tendon_natural_cadence_hz)
    assert es.hr > hr0 + 20
    assert es.hr <= p.hr_max_bpm + 6


def test_breathing_mismatch_reduces_oxygen():
    p = BodyParams()
    matched = EnergySystem(p)
    mismatched = EnergySystem(p)
    for _ in range(200):
        matched.step(0.1, speed_mps=p.critical_speed_mps, breathing=0.7)
        mismatched.step(0.1, speed_mps=p.critical_speed_mps, breathing=0.0)
    # Poor breathing should deliver less oxygen for the same effort.
    assert matched.vo2 > mismatched.vo2


def test_muscle_fatigue_accumulates_and_recovers():
    p = BodyParams()
    ms = MuscleSystem(p)
    high = {g: 1.0 for g in MUSCLE_GROUPS}
    for _ in range(300):
        ms.step(0.1, high)
    fatigued = ms.fatigue["quad"]
    assert fatigued > 0.3
    assert ms.available_frac("quad") < 1.0
    low = {g: 0.0 for g in MUSCLE_GROUPS}
    for _ in range(600):
        ms.step(0.1, low)
    assert ms.fatigue["quad"] < fatigued  # recovers when rested


def test_available_torque_bounded():
    p = BodyParams()
    ms = MuscleSystem(p)
    for g in MUSCLE_GROUPS:
        assert 0.25 <= ms.available_frac(g) <= 1.0
