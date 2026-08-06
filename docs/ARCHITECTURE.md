# Architecture

This document describes how MileRunner is put together and the design decisions
behind each layer. The guiding constraint is that **the platform builds the
learning system, not the running strategy** — every technique must be discovered
by reinforcement learning and evolution.

```
                         ┌─────────────────────────────────────────┐
                         │            ContinuousTrainer             │
                         │  auto-start · loop forever · pause/resume│
                         │  status.json · DB logging · checkpoints  │
                         └───────────────┬─────────────────────────┘
                                         │ generations
                         ┌───────────────▼─────────────────────────┐
                         │               Population (PBT)           │
                         │  train → evaluate → select top 10% →     │
                         │  mutate/crossover → breed offspring       │
                         └───────┬───────────────────────┬─────────┘
                                 │ per agent             │ persistence
                 ┌───────────────▼──────────┐   ┌────────▼─────────┐
                 │  Agent (SB3)             │   │  ExperimentDB    │
                 │  PPO/SAC/TD3/DDPG/A2C/   │   │  (SQLite)        │
                 │  RecurrentPPO            │   │  + checkpoints   │
                 │  + arch: MLP/CNN/        │   └──────────────────┘
                 │  Transformer/GRU         │
                 └───────────────┬──────────┘
                                 │ acts / observes
                 ┌───────────────▼──────────────────────────────────┐
                 │                 MileRunEnv (Gymnasium)            │
                 │   ┌──────────┐  ┌──────────┐  ┌───────────────┐   │
                 │   │ Humanoid │  │ Energy   │  │ Muscle system │   │
                 │   │ (MuJoCo) │  │ system   │  │ (fatigue)     │   │
                 │   └──────────┘  └──────────┘  └───────────────┘   │
                 │        Track / atmosphere (drag, wind, temp)      │
                 └───────────────────────────────────────────────────┘
```

## Layers

### 1. Physics (`milerunner/physics`)
- **`body_builder.py`** generates a MuJoCo MJCF humanoid *parametrically* from
  `BodyParams`. Segment masses/lengths come from anthropometric fractions, so a
  60 kg or 90 kg "body type" yields a physically consistent skeleton. Joints are
  named after the muscle groups that drive them and given clinical range‑of‑motion
  limits. Total mass matches the spec exactly (masses set per‑geom).
- **`humanoid.py`** wraps `MjModel`/`MjData` and exposes joint angles/velocities,
  foot ground‑reaction forces (from contact solver), COM velocity, balance, and
  **fatigue‑scaled torque clipping** so the body never exceeds current strength.
- **`track.py`** models the 400 m oval, air density vs temperature/altitude,
  aerodynamic drag with wind, and the small extra cost of running the bends.
  Physics runs on a straight ground plane (fast, stable) with curvature folded
  into an energetic penalty.

### 2. Biomechanics (`milerunner/biomech`) — the realism core
- **`params.py`** — all anthropometric & physiological constants.
- **`energy.py`** — the stateful physiology: VO₂ kinetics, heart rate, the
  **critical‑speed / D′ (W′‑balance)** anaerobic reserve (the power–duration
  law that makes pacing a real decision), lactate, glycogen, breathing‑delivery
  efficiency, and humidity‑limited thermoregulation. Critical speed is guarded
  to remain below VO₂max pace so the model can't be made superhuman by a bad
  config.
- **`muscles.py`** — per‑group activation and local fatigue → the per‑joint
  torque limit fed to the physics layer.

### 3. Environment (`milerunner/envs`)
- **`mile_env.py`** composes physics + biomechanics + track into a Gymnasium
  env. Control at 100 Hz, physics at 1000 Hz (10 substeps). Actions are joint
  torques (clipped by fatigue) plus a breathing channel. It tracks cadence and
  stride length from foot‑strike transitions, updates all physiological state,
  computes reward, and emits rich telemetry.
- **`observations.py`** assembles the 74‑D observation (labelled for the
  dashboard). **`rewards.py`** defines the weighted reward whose weights are
  *evolvable*, so the platform never fixes the speed/economy/stability trade‑off
  by hand.

### 4. Agents (`milerunner/agents`)
- **`networks.py`** — interchangeable feature extractors (MLP, 1D‑CNN,
  Transformer/attention over tokenised observations, GRU memory) for the
  architecture search.
- **`factory.py`** — builds any of PPO/SAC/TD3/DDPG/A2C/RecurrentPPO from a
  hyperparameter+architecture spec, so evolution can swap both the algorithm and
  the network.

### 5. Evolution (`milerunner/evolution`)
- **`genome.py`** — the evolvable spec: algorithm, architecture, activation,
  optimisation & exploration hyperparameters, **reward weights**, and a training
  budget gene. Explicitly *not* included: any stride/cadence/pacing/breathing
  policy. Provides bounded mutation and uniform crossover.
- **`population.py`** — population‑based training: per‑agent continued training
  from checkpoints (experience compounds), fitness evaluation, elitism (top
  10%), tournament selection, offspring warm‑started from elite weights when
  architectures match ("exploit + explore"), and checkpoint garbage‑collection
  to bound disk. Fully serialisable for pause/resume.

### 6. Training pipeline (`milerunner/training`)
- **`env_builder.py`** — single or (sub)process‑vectorised parallel envs.
- **`evaluation.py`** — deterministic mile evaluation → fitness (finish time
  dominates; partial‑distance credit early on) + telemetry, muscle‑fatigue
  timelines, and a sub‑sampled 3D skeleton for replay.
- **`checkpoint.py`** — per‑agent (model + genome) and whole‑search‑state
  persistence (atomic writes).
- **`tournament.py`** — cross‑algorithm ranking + per‑algo/arch aggregation.
- **`trainer.py`** — the continuous orchestrator (auto‑start, loop, signal
  handling for graceful pause, status file, DB logging, resume).

### 7. Database (`milerunner/database`)
- **`experiment_db.py`** — append‑mostly SQLite schema for experiments,
  generations, individuals, evaluations, checkpoints and records, with the
  queries the dashboard and analysis need. WAL mode for concurrent reads while
  training writes.

### 8. Dashboard & visualization (`milerunner/dashboard`)
- **`figures.py`** — Plotly builders for every required chart.
- **`replay.py`** — a **renderer‑agnostic 3D runner**: the skeleton is
  reconstructed from forward kinematics (no OpenGL needed, works headless) and
  animated with Plotly; a photorealistic MuJoCo video path is used when a GL
  backend is present.
- **`app.py`** — the live Dash app (auto‑refresh from status.json + DB).

## Data & control flow per generation

1. For each individual: build its env with *its* evolved reward weights →
   build/continue its SB3 agent (warm‑start from checkpoint) → `learn(budget)` →
   save checkpoint (+genome) → evaluate on the mile → record fitness & telemetry
   → log to DB; update best‑ever record if it's a new fastest mile.
2. Rank the population; log the generation summary; write `status.json`.
3. Evolve: keep elites (they keep training next gen), cull the rest, breed
   mutated/recombined offspring, GC dead checkpoints.
4. Save full search state (atomic) for resume.

## Determinism & reproducibility

Seeds flow from config through `seed_everything` to numpy/torch/env RNGs. Each
run records its fully‑resolved config in the DB. The env is deterministic given
a seed (verified by tests).

## Extending

- **New body type**: add a block to `configs/body_types.yaml` (or set `body.*`).
- **New algorithm/architecture**: register it in `agents/factory.py` /
  `agents/networks.py` and add its name to the genome's option lists.
- **New reward term**: add to `RewardWeights` + `compute_reward`; it becomes
  evolvable automatically.
- **Distributed scale‑out**: run multiple `scripts/run.py` workers against one
  `--set trainer.db_path=...` shared database.
