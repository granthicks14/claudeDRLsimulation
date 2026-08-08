"""Continuous, autonomous training orchestrator.

This is what the user launches. It:

* auto-starts population-based training the moment it runs;
* runs generation after generation **indefinitely** (or until a generation
  cap), so the longer it runs the more experience accumulates and the better the
  best mile time gets;
* checkpoints the full search state every generation, so the run can be paused
  (Ctrl-C) and resumed later exactly where it stopped;
* writes a live status file the dashboard reads, so training runs in the
  background while the user watches statistics and visualisations;
* logs every experiment, generation, evaluation and record to the database and
  preserves the best models ever discovered.

The agents learn entirely on their own — this loop only schedules compute,
selection and persistence.
"""
from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from ..database.experiment_db import ExperimentDB
from ..envs.mile_env import EnvConfig
from ..training.checkpoint import load_search_state, save_search_state
from ..training.tournament import aggregate_by, run_tournament
from ..utils.logging import get_logger

log = get_logger("milerunner.trainer")


@dataclass
class TrainerConfig:
    experiment_name: str = "default"
    max_generations: int = 0            # 0 = run forever
    checkpoint_root: str = "checkpoints"
    state_dir: str = "experiments"
    db_path: str = "experiments/milerunner.db"
    status_path: str = "experiments/status.json"
    save_every: int = 1
    seed: int = 0


class ContinuousTrainer:
    def __init__(self, trainer_cfg: TrainerConfig, pop_cfg: "PopulationConfig",
                 env_cfg: EnvConfig, full_config: Optional[Dict[str, Any]] = None,
                 body=None):
        # Imported here (not at module top) to avoid an evolution<->training
        # import cycle: population.py imports training helpers.
        from ..evolution.population import Population
        self.tcfg = trainer_cfg
        self.pop_cfg = pop_cfg
        self.env_cfg = env_cfg
        self.body = body
        self.full_config = full_config or {}
        os.makedirs(self.tcfg.state_dir, exist_ok=True)
        self.db = ExperimentDB(self.tcfg.db_path)
        self.exp_id = self.db.get_or_create_experiment(
            self.tcfg.experiment_name, self.full_config)
        self.population = Population(
            pop_cfg, env_cfg, self.db, self.exp_id, self.tcfg.experiment_name,
            root=self.tcfg.checkpoint_root, seed=self.tcfg.seed, body=body,
        )
        self._stop = False
        self._state_path = os.path.join(self.tcfg.state_dir,
                                        f"{self.tcfg.experiment_name}_state.json")

    # ------------------------------------------------------------------ #
    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):  # pragma: no cover - signal path
            log.warning("Received signal %s -> will pause after this generation.", signum)
            self._stop = True
        try:
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
        except Exception:  # pragma: no cover - not in main thread
            pass

    def resume_or_init(self) -> None:
        state = load_search_state(self._state_path)
        if state is not None:
            self.population.load_state_dict(state)
            log.info("Resumed experiment '%s' at generation %d (%d cumulative steps).",
                     self.tcfg.experiment_name, self.population.generation,
                     self.population.total_timesteps)
        else:
            self.population.initialize()
            log.info("Started fresh experiment '%s'.", self.tcfg.experiment_name)

    # ------------------------------------------------------------------ #
    def _write_status(self, summary: Dict[str, Any]) -> None:
        rows = run_tournament(self.population.individuals)
        status = {
            "experiment": self.tcfg.experiment_name,
            "updated_at": time.time(),
            "generation": self.population.generation,
            "total_timesteps": self.population.total_timesteps,
            "population_size": len(self.population.individuals),
            "best_record": self.population.best_record,
            "last_summary": summary,
            "leaderboard": [asdict(r) for r in rows[:15]],
            "by_algo": aggregate_by(rows, "algo"),
            "by_arch": aggregate_by(rows, "arch"),
        }
        tmp = self.tcfg.status_path + ".tmp"
        os.makedirs(os.path.dirname(self.tcfg.status_path) or ".", exist_ok=True)
        with open(tmp, "w") as fh:
            json.dump(status, fh, indent=2, default=str)
        os.replace(tmp, self.tcfg.status_path)

        # Sidecar: the current best agent's telemetry, so the dashboard can show
        # live speed/HR/energy/fatigue curves and the 3D replay every generation
        # — even before any agent completes a full mile.
        tele = self.population.latest_best_telemetry
        if tele:
            tele_path = os.path.join(os.path.dirname(self.tcfg.status_path) or ".",
                                     "best_telemetry.json")
            tmp2 = tele_path + ".tmp"
            with open(tmp2, "w") as fh:
                json.dump(tele, fh, default=str)
            os.replace(tmp2, tele_path)

        # Sidecar: per-agent race data (one runner per lane) for the race view.
        race = getattr(self.population, "latest_race", [])
        if race:
            race_path = os.path.join(os.path.dirname(self.tcfg.status_path) or ".",
                                     "race.json")
            tmp3 = race_path + ".tmp"
            with open(tmp3, "w") as fh:
                json.dump(race, fh, default=str)
            os.replace(tmp3, race_path)

    def _save(self) -> None:
        save_search_state(self._state_path, self.population.state_dict())

    # ------------------------------------------------------------------ #
    def run(self, max_generations: Optional[int] = None) -> None:
        """Main loop. Blocks; runs until stopped or the generation cap is hit."""
        self._install_signal_handlers()
        self.resume_or_init()
        self.db.set_experiment_status(self.exp_id, "running")
        cap = max_generations if max_generations is not None else self.tcfg.max_generations
        start_gen = self.population.generation
        try:
            while not self._stop:
                if cap and (self.population.generation - start_gen) >= cap:
                    log.info("Reached generation cap (%d).", cap)
                    break
                summary = self.population.train_generation()
                self._write_status(summary)
                if self.population.generation % self.tcfg.save_every == 0:
                    self._save()
                self.population.evolve()
                self._save()
        finally:
            self._save()
            self.db.set_experiment_status(self.exp_id, "paused")
            log.info("Training paused/stopped at generation %d. State saved to %s",
                     self.population.generation, self._state_path)

    def stop(self) -> None:
        self._stop = True
