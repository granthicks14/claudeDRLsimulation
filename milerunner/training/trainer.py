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
        self._save_sidecars()

    def _save_sidecars(self) -> None:
        """Write the telemetry + race sidecars the dashboard reads to show the
        runner. Called every generation AND after each agent (incremental) AND
        after a startup warmup, so the runner appears within seconds."""
        d = os.path.dirname(self.tcfg.status_path) or "."
        os.makedirs(d, exist_ok=True)
        tele = self.population.latest_best_telemetry
        if tele:
            p = os.path.join(d, "best_telemetry.json")
            with open(p + ".tmp", "w") as fh:
                json.dump(tele, fh, default=str)
            os.replace(p + ".tmp", p)
        race = getattr(self.population, "latest_race", [])
        if race:
            p = os.path.join(d, "race.json")
            with open(p + ".tmp", "w") as fh:
                json.dump(race, fh, default=str)
            os.replace(p + ".tmp", p)

    def _save(self) -> None:
        save_search_state(self._state_path, self.population.state_dict())

    # ------------------------------------------------------------------ #
    def run(self, max_generations: Optional[int] = None) -> None:
        """Main loop. Blocks; runs until stopped or the generation cap is hit."""
        self._install_signal_handlers()
        self.resume_or_init()
        self.db.set_experiment_status(self.exp_id, "running")

        # Warm-up: produce a first (untrained) runner rollout immediately so the
        # dashboard shows a runner within seconds instead of waiting for a full
        # generation. Best-effort — never blocks training if it fails.
        if not self.population.latest_best_telemetry:
            try:
                log.info("[8] Warm-up rollout so the runner appears immediately …")
                self.population.quick_warmup()
                self._save_sidecars()
                log.info("[8] Warm-up done — the dashboard can show the runner now.")
            except Exception as e:  # pragma: no cover
                log.warning("warm-up rollout skipped: %s", e)

        cap = max_generations if max_generations is not None else self.tcfg.max_generations
        start_gen = self.population.generation
        try:
            while not self._stop:
                if cap and (self.population.generation - start_gen) >= cap:
                    log.info("Reached generation cap (%d).", cap)
                    break
                # Update the sidecars after every agent so the runner refreshes
                # continuously, not just once per generation.
                summary = self.population.train_generation(on_agent_done=self._save_sidecars)
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
