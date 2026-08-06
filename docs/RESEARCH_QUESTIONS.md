# Research questions

The platform is built to *answer these empirically from discovered runs*, not to
assert answers. `scripts/analyze.py` turns the best recorded agent's telemetry
and the body's physiological model into a data‑driven report; the sections below
explain the method for each question. All numbers come from simulation, none are
hand‑authored.

## Is there a faster stride pattern than humans currently use?
The agent controls raw joint torques, so its gait — stride length, cadence,
ground‑contact timing, arm swing, lean — is emergent. `analyze.py` reports the
discovered mean cadence and implied stride length; compare against typical human
mile values (~180 steps/min, ~1.4 m stride). The 3D replay
(`scripts/evaluate.py`) lets you inspect the form directly for novel patterns.
The body's tendon model gives an economy optimum near a natural cadence, but the
agent is free to discover that this is worth exploiting — or to trade economy for
speed.

## What is the theoretical fastest mile?
Computed from the **critical‑power model**: with critical speed `CS` and
anaerobic distance reserve `D′`, the fastest time to cover distance `d` is
`t = (d − D′) / CS`. For the default average male this is **≈ 6:26** — a physics
floor that ignores VO₂ kinetics, aerodynamic drag and biomechanical losses, so
real discovered runs are slower. Change the body (`configs/body_types.yaml`) to
see how the floor moves; a lightweight high‑VO₂max build has a much faster floor.

## How much energy should be conserved for the final lap?
The W′‑balance reserve is tracked every step. `analyze.py` reports the reserve at
the start of the final quarter and at the finish. A well‑optimised run empties
the reserve *right at the line* — spending it too early causes a late‑race
collapse (the model exhausts the runner), spending it too late leaves speed on
the table. Because pacing is emergent, the optimal reserve profile is a
*discovered* result, visible in the energy curve.

## What cadence is optimal?
Reported as the discovered mean cadence. The body's elastic‑recoil economy peaks
near a natural cadence, but whether the fastest legal mile uses that cadence, or a
higher one that sacrifices economy for speed, is settled by the search — not
prescribed.

## What breathing pattern is best?
Breathing is an action; oxygen delivery is best when respiratory effort matches
metabolic demand, and over‑breathing wastes energy. The agent must discover to
breathe harder as it works harder (and the rhythm that does so efficiently). The
breathing signal and its relationship to effort are in the telemetry.

## Could an AI discover a completely new running form?
This is the open‑ended goal. Because nothing about the gait is coded, and
population‑based training continually explores new architectures, hyperparameters
and reward trade‑offs, unusual but physically legal forms can emerge. Inspect the
best‑agent 3D replay and the per‑muscle‑group fatigue heat‑map: a form that loads
muscles or times contacts unlike human running, yet obeys every joint, strength
and energy limit, is a candidate novel technique. Longer training over more
generations widens the search.

---

### How to reproduce the analyses

```bash
python scripts/run.py --config default          # train (longer = better)
python scripts/analyze.py --experiment default   # -> experiments/report.md
python scripts/evaluate.py --experiment default  # -> interactive 3D replay
python scripts/benchmark.py --biomech            # validate the physiology model
```

Run different bodies and weather to compare:

```bash
python scripts/run.py --experiment lightweight \
  --set body.mass_kg=60 body.vo2max_ml_kg_min=62 body.critical_speed_mps=4.7
```
