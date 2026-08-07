"""Translate a loaded YAML :class:`Config` into the concrete config objects the
training pipeline consumes. Keeping this in one place makes experiments fully
reproducible: a single YAML file determines the environment, population and
trainer.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .biomech.params import BodyParams
from .envs.mile_env import EnvConfig
from .evolution.population import PopulationConfig
from .physics.track import MILE_M, Weather
from .training.trainer import TrainerConfig
from .utils.config import Config
from .utils.device import resolve_device


def build_body(cfg: Config) -> BodyParams:
    b = cfg.section("body").to_dict()
    return BodyParams.from_dict(b) if b else BodyParams()


def build_weather(cfg: Config) -> Weather:
    w = cfg.section("weather").to_dict()
    known = {k: w[k] for k in ("temperature_c", "wind_mps", "humidity",
                               "altitude_m", "pressure_kpa") if k in w}
    return Weather(**known)


def build_env_config(cfg: Config) -> EnvConfig:
    e = cfg.section("env")
    return EnvConfig(
        distance_m=float(e.get("distance_m", MILE_M)),
        control_hz=float(e.get("control_hz", 100.0)),
        physics_timestep=float(e.get("physics_timestep", 0.001)),
        max_time_s=float(e.get("max_time_s", 900.0)),
        terminate_on_fall=bool(e.get("terminate_on_fall", True)),
        terminate_on_exhaustion=bool(e.get("terminate_on_exhaustion", False)),
        friction=float(e.get("friction", 1.0)),
        reset_noise=float(e.get("reset_noise", 0.02)),
        randomize_weather=bool(e.get("randomize_weather", False)),
        randomize_body=bool(e.get("randomize_body", False)),
        stall_timeout_s=float(e.get("stall_timeout_s", 0.0)),
        stall_min_progress_m=float(e.get("stall_min_progress_m", 1.0)),
        pace_deadline_s=float(e.get("pace_deadline_s", 0.0)),
        pace_deadline_m=float(e.get("pace_deadline_m", 0.0)),
    )


def build_population_config(cfg: Config) -> PopulationConfig:
    p = cfg.section("population")
    device = resolve_device(cfg.get("hardware.device", "auto"))
    algos = p.get("algos", None)
    if isinstance(algos, str):
        algos = [algos]
    return PopulationConfig(
        size=int(p.get("size", 12)),
        elite_frac=float(p.get("elite_frac", 0.1)),
        timesteps_per_gen=int(p.get("timesteps_per_gen", 4000)),
        n_envs=int(cfg.get("hardware.n_envs", 2)),
        subprocess=bool(cfg.get("hardware.subprocess", False)),
        device=device,
        algos=algos,
        tournament_k=int(p.get("tournament_k", 3)),
        mutation_strength=float(p.get("mutation_strength", 0.15)),
        keep_best_models=int(p.get("keep_best_models", 25)),
        warm_start=bool(p.get("warm_start", True)),
    )


def build_trainer_config(cfg: Config, experiment_name: Optional[str] = None) -> TrainerConfig:
    t = cfg.section("trainer")
    name = experiment_name or t.get("experiment_name", cfg.get("experiment_name", "default"))
    return TrainerConfig(
        experiment_name=name,
        max_generations=int(t.get("max_generations", 0)),
        checkpoint_root=t.get("checkpoint_root", "checkpoints"),
        state_dir=t.get("state_dir", "experiments"),
        db_path=t.get("db_path", "experiments/milerunner.db"),
        status_path=t.get("status_path", "experiments/status.json"),
        save_every=int(t.get("save_every", 1)),
        seed=int(cfg.get("seed", 0)),
    )


def build_all(cfg: Config, experiment_name: Optional[str] = None
              ) -> Tuple[TrainerConfig, PopulationConfig, EnvConfig]:
    return (build_trainer_config(cfg, experiment_name),
            build_population_config(cfg),
            build_env_config(cfg))
