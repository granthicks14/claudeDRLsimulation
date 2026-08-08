# 🏃 MileRunner — Discovering the Fastest Human Mile with Deep RL + Evolution

## ▶ Watch it run in your browser — FREE, one click (no credit card)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/granthicks14/claudeDRLsimulation?quickstart=1)

**Click the badge above** (while signed in to GitHub). It spins up a free cloud
machine, auto-installs everything, and auto-opens the live dashboard — you watch
the AI runners, the track, the race, and live stats in your browser. First build
takes ~3–5 min, then the runners appear after generation 0 (~2–3 min). No GPU, no
payment. Full walkthrough: [`docs/CODESPACES_SETUP.md`](docs/CODESPACES_SETUP.md).

> Free GitHub accounts include ~60 hours/month of Codespaces with **no credit
> card**; it auto-stops when idle so you won't burn hours. Training **auto-saves
> and resumes**, so stopping/reopening never loses progress.
>
> **Want it to run longer?** No free, no-card service runs 24/7 forever, but see
> [`docs/RUN_LONGER.md`](docs/RUN_LONGER.md): raise the Codespaces idle timeout to
> **4 hours**, use **Kaggle for 12-hour sessions** (30 GB RAM, no card), or
> **Oracle Cloud Always Free** for true 24/7 (needs a card for ID only).

---

A research-grade platform that trains **populations of AI agents** to discover
the fastest possible one‑mile running strategy for an **average human body**,
under realistic biomechanics and physics. The agents are *not* told how to run:
stride pattern, cadence, pacing, arm swing, body lean and breathing rhythm all
**emerge** from reinforcement learning inside a physiologically‑constrained
simulation, while **evolutionary population‑based training** searches over
algorithms, network architectures, hyperparameters and reward trade‑offs.

Training is **continuous and autonomous** — launch it and it keeps improving for
as long as you give it compute; stop it and resume later without losing
progress. The longer it runs, the faster the discovered mile.

> **Design philosophy.** The framework builds the *infrastructure, simulation,
> and learning system only*. It never hard‑codes a running strategy or manually
> optimizes the runners — every technique is discovered by the RL + evolution
> loop. As the brief notes, the realism ceiling is set by the biomechanical
> model, so the hardest engineering here is the **muscle / tendon / fatigue /
> energy‑system simulation**, which is treated as a first‑class component.

---

## Quickstart

```bash
pip install -r requirements.txt          # torch, mujoco, stable-baselines3, …

# 1) Start continuous, autonomous training (auto-starts, runs until Ctrl-C):
python scripts/run.py                     # default config (average male, full mile)
python scripts/run.py --config smoke      # 2-minute demo on a laptop CPU

# 2) Watch live statistics & visualizations in another terminal:
python scripts/dashboard.py               # http://127.0.0.1:8050

# 3) Anytime: evaluate & replay the best agent discovered so far:
python scripts/evaluate.py --experiment default     # writes an interactive 3D replay

# 4) Generate a data-driven research report:
python scripts/analyze.py --experiment default

# 5) Benchmark the simulator & validate the biomechanics:
python scripts/benchmark.py --all
```

Pause anytime with **Ctrl‑C**; re‑run the same command to **resume exactly where
you left off** (generation, cumulative timesteps, population and best models are
all restored).

---

## What it does

- **Trains thousands of agents** (configurable) simultaneously with
  population‑based training and evolutionary selection.
- **Multiple RL algorithms compete** in a tournament: PPO, SAC, TD3, DDPG, A2C
  (the synchronous form of A3C), plus recurrent PPO (LSTM).
- **Architecture search** over MLP / 1D‑CNN / Transformer‑attention / GRU‑memory
  feature extractors, evolved alongside the algorithm.
- **Every generation**: train → evaluate on the mile → keep the best 10% →
  mutate hyperparameters & reward weights → breed offspring (warm‑started from
  elite weights) → continue. Elites keep training, so experience compounds.
- **Full persistence**: an SQLite database records every experiment, generation,
  agent and evaluation; the best mile times and model checkpoints are preserved.
- **Live analytics dashboard**: fastest mile, speed / cadence / heart‑rate /
  oxygen / energy / lactate curves, muscle‑fatigue heat‑map, per‑algorithm and
  per‑architecture comparisons, and an animated **3D replay** of the best runner.

---

## The human body model (the hard part)

`milerunner/biomech` and `milerunner/physics` implement an **average male**
(175 cm, 77 kg) with physiologically grounded systems. None of these encode a
*strategy* — they describe the body the agent must learn to move.

| System | Model | File |
|---|---|---|
| **Skeleton** | 3D bipedal MuJoCo humanoid; segment masses & lengths from Winter/Dempster anthropometry (total mass exact to spec) | `physics/body_builder.py` |
| **Muscle groups** | feet/calves→ankle, quads/hamstrings→knee, glutes→hip, core→trunk, shoulders/arms→shoulder/elbow, neck; each with a peak‑torque budget & **local fatigue** that limits available torque | `biomech/muscles.py` |
| **Joint limits** | clinical range‑of‑motion per joint — impossible poses are physically impossible | `physics/body_builder.py` |
| **Aerobic system** | VO₂max, oxygen‑uptake kinetics (first‑order lag), lactate threshold | `biomech/energy.py` |
| **Anaerobic system** | **critical‑speed / D′ (W′‑balance) model** — the validated power–duration law that makes pacing matter | `biomech/energy.py` |
| **Cardiac** | heart‑rate model with lag + cardiac drift from heat & lactate | `biomech/energy.py` |
| **Tendon elasticity** | baseline economy includes recoil; running *off* the body's natural cadence loses it (an economy penalty, not a hand‑given optimum) | `biomech/energy.py` |
| **Thermoregulation** | metabolic heat vs convective + humidity‑limited evaporative cooling; heat‑stroke cutoff | `biomech/energy.py` |
| **Glycogen** | finite carbohydrate store; bonking ends the run | `biomech/energy.py` |
| **Breathing** | respiratory effort as an action; O₂ delivery is best when breathing matches demand — the optimal pattern must be discovered | `biomech/energy.py` |
| **Environment** | 400 m oval bookkeeping, air density vs temperature/altitude, wind (head/tail) drag, curve cost, ground friction & reaction forces | `physics/track.py`, `physics/humanoid.py` |

The model is **internally consistent and validated**: critical speed is guarded
to stay below VO₂max pace, and the critical‑power relationship reproduces the
correct hyperbolic time‑to‑exhaustion curve (see `python scripts/benchmark.py
--biomech`). For the default body the physics implies a **theoretical fastest
mile of ≈ 6:26** — a realistic floor for an average fit male.

### Observations & actions

Each agent observes (74 values): speed, acceleration, heart rate, oxygen level &
demand, per‑muscle‑group fatigue, all joint angles & velocities, stride length,
cadence, distance remaining/covered, balance (uprightness, lean, angular rate),
energy reserves (W′, glycogen), ground‑reaction forces, lactate, core
temperature, foot contacts, race‑clock and breathing.

Each agent controls (23 values): joint torques for all actuated joints — from
which foot placement, stride frequency/length, arm swing, muscle activation,
body lean, head position and push‑off force all emerge — plus a **breathing**
command. Pacing emerges over the episode from how the policy spends the finite
W′ reserve.

Torques are clipped to the **current** (fatigue‑reduced) human strength of each
muscle group every step, so the body can never exceed human capability.

---

## How learning stays autonomous and open‑ended

- **Auto‑start.** `scripts/run.py` begins training immediately.
- **Continuous.** It loops generations indefinitely; the best mile keeps
  dropping as compute accrues. Elites persist and *continue* training from their
  own checkpoints, so an agent alive for many generations has trained far longer
  than a fresh one — an agent trained for a week beats one trained for an hour,
  by construction.
- **Self‑improving via evolution.** Poor agents are culled and replaced by
  mutated/recombined offspring of the best. The search explores learning rates,
  reward weights, network size/type, exploration and training‑budget genes.
- **Reproducible & resumable.** Full search state is checkpointed every
  generation; a shared SQLite DB preserves every experiment and the best models
  ever found.
- **Nothing hand‑authored.** Claude built the environment and learning system;
  the running technique is discovered entirely by RL + evolution. The reward
  function scores *outcomes* (faster legal mile, stability, economy) and its
  weights are themselves evolved — no stride/pacing/breathing policy is coded.

---

## Hosted live dashboard (Docker)

Run the trainer **and** the live web dashboard together in one container and
deploy it to any host that runs long‑lived containers:

```bash
docker compose up --build        # local:  http://localhost:8050
```

One‑click cloud deploys are included: **Render** (`render.yaml` Blueprint),
**Fly.io** (`fly.toml`), Railway, or any Docker host. The image runs
`scripts/run.py` in the background and serves the dashboard via gunicorn, sharing
a persistent `/app/experiments` volume so training survives restarts. See
[`docs/DEPLOY.md`](docs/DEPLOY.md). (Vercel can only host the static `index.html`
project page — it can't run the training process.)

**Free, no credit card:** the easiest ways to run it for free are
**GitHub Codespaces** (8 GB RAM, auto-starts in your browser — see
[`docs/CODESPACES_SETUP.md`](docs/CODESPACES_SETUP.md)) or **Google Colab** —
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/granthicks14/claudeDRLsimulation/blob/main/notebooks/MileRunner_Colab.ipynb).
Both need no payment method. (Render's free 512 MB tier OOMs once PyTorch loads,
and Hugging Face Spaces may now require a card.) See
[`docs/DEPLOY.md`](docs/DEPLOY.md#free-hosting-options) for all options.

## Scaling from a laptop to a cluster

The same code scales by config (`configs/`):

- `smoke.yaml` — tiny population, short episodes; a couple of minutes on a CPU.
- `default.yaml` — 16‑agent population, full mile.
- `cluster.yaml` — hundreds of agents/node, subprocess vectorized envs, CUDA;
  point many worker processes/nodes at one shared experiment database to reach
  the brief's **10,000+ runners**.

GPU/CUDA is used automatically when available (`hardware.device: auto`) and
falls back to CPU unchanged. Physics runs at **1000+ steps/sec per env** (≈8,000
on one CPU core here) and multiplies with parallel environments.

---

## Repository layout

```
milerunner/
  biomech/     params, energy system (VO2/HR/lactate/W'/thermo/glycogen), muscles
  physics/     MuJoCo humanoid builder, humanoid wrapper, track & atmosphere
  envs/        Gymnasium mile environment, observations, reward shaping
  agents/      network architectures (MLP/CNN/Transformer/GRU) + SB3 agent factory
  evolution/   genome (evolvable spec) + population-based training / selection
  training/    parallel env builder, evaluation, checkpointing, tournament, trainer
  database/    SQLite experiment tracking + best-model registry
  dashboard/   live Dash app, Plotly figures, 3D skeleton replay
  utils/       config, logging, seeding, device
configs/       default, smoke, cluster, body_types, weather (YAML)
scripts/       run (train), dashboard, evaluate, analyze, benchmark
tests/         unit + integration tests (pytest)
docs/          architecture & research notes
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design and
[`docs/RESEARCH_QUESTIONS.md`](docs/RESEARCH_QUESTIONS.md) for how the platform
addresses each research question.

---

## Testing

```bash
python -m pytest -q            # 32 unit + integration tests
python scripts/benchmark.py --physics --biomech    # throughput + physiology validation
```

The tests cover the biomechanical model (segment masses, exhaustion curves,
fatigue), the physics/env (mass, stability, determinism, physics rate), the
evolutionary machinery (bounds, selection, elitism, state round‑trip), and the
infrastructure (config, database, rewards, networks, tournament).

---

## Configuration

Everything is driven by YAML in `configs/`. Override on the command line:

```bash
python scripts/run.py --config default \
  --experiment windy_hot \
  --set weather.temperature_c=32 weather.wind_mps=-4 population.size=64
```

Child configs inherit via `base:` (e.g. `smoke.yaml` extends `default.yaml`).

## License

MIT — see `LICENSE`.
