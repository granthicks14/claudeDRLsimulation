"""Population-based training (PBT) with evolutionary selection.

Thousands of runner-agents (configurable) form a population. Each generation:

1. **Train** every surviving agent for a compute budget (it *continues* from its
   own checkpoint, so experience accumulates over generations — an agent alive
   for many generations has trained far longer than a fresh one).
2. **Evaluate** each on the mile to get a fitness (finish time dominates).
3. **Select** the best ``elite_frac`` (default top 10%). Elites survive and keep
   training.
4. **Breed** offspring from elites via tournament selection + crossover +
   mutation, warm-starting from a parent's weights when architectures match
   (PBT "exploit + explore"). Offspring replace the culled agents.

Because elites persist and accumulate training while the population keeps
exploring new hyperparameters, architectures and reward-weight trade-offs, the
system improves for as long as it runs. No running strategy is coded here.
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..agents.factory import build_agent
from ..database.experiment_db import ExperimentDB
from ..envs.mile_env import EnvConfig
from ..training.env_builder import build_single_env, build_vec_env
from ..training.evaluation import EvalResult, evaluate_policy
from ..utils.logging import get_logger
from .genome import Genome, crossover, mutate, random_genome

log = get_logger("milerunner.population")


@dataclass
class Individual:
    genome: Genome
    model_path: Optional[str] = None       # persistent checkpoint (its own)
    init_from: Optional[str] = None        # warm-start source for first build
    fitness: float = -1e9
    total_timesteps: int = 0
    age: int = 0                           # generations survived
    last_result: Optional[Dict[str, Any]] = None
    telemetry: Dict[str, Any] = field(default_factory=dict, repr=False)  # transient

    def arch_signature(self) -> tuple:
        g = self.genome
        hp = g.hyperparams
        return (g.algo, g.arch, g.activation, hp.get("hidden"), hp.get("depth"),
                hp.get("features_dim"))


@dataclass
class PopulationConfig:
    size: int = 12
    elite_frac: float = 0.1
    timesteps_per_gen: int = 4000
    n_envs: int = 2
    subprocess: bool = False
    device: str = "auto"
    algos: Optional[List[str]] = None
    tournament_k: int = 3
    mutation_strength: float = 0.15
    eval_seed: int = 20240
    keep_best_models: int = 25
    warm_start: bool = True


class Population:
    def __init__(self, cfg: PopulationConfig, env_config: EnvConfig,
                 db: ExperimentDB, exp_id: int, experiment_name: str,
                 root: str = "checkpoints", seed: int = 0,
                 train_fn: Optional[Callable] = None,
                 eval_fn: Optional[Callable] = None):
        self.cfg = cfg
        self.env_config = env_config
        self.db = db
        self.exp_id = exp_id
        self.experiment_name = experiment_name
        self.root = root
        self.rng = np.random.default_rng(seed)
        self.generation = 0
        self.individuals: List[Individual] = []
        self.total_timesteps = 0
        self.best: Optional[Individual] = None
        self.best_record: Optional[Dict[str, Any]] = None
        # Telemetry of the current generation's best agent (finished or not), so
        # the dashboard can always show live curves before any full mile exists.
        self.latest_best_telemetry: Dict[str, Any] = {}
        self._train_fn = train_fn or self._default_train
        self._eval_fn = eval_fn or self._default_eval
        os.makedirs(os.path.join(root, experiment_name), exist_ok=True)

    # ------------------------------------------------------------------ #
    def initialize(self) -> None:
        self.individuals = []
        for _ in range(self.cfg.size):
            g = random_genome(self.rng, algos=self.cfg.algos, generation=0)
            self.individuals.append(Individual(genome=g))
            self.db.log_individual(self.exp_id, g.to_dict())
        log.info("Initialised population of %d agents", len(self.individuals))

    # ------------------------------------------------------------------ #
    def _agent_paths(self, genome_id: str) -> str:
        d = os.path.join(self.root, self.experiment_name, genome_id)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "model.zip")

    def _default_train(self, ind: Individual, timesteps: int) -> int:
        """Build/continue an agent and train it. Returns timesteps trained."""
        g = ind.genome
        env = build_vec_env(
            n_envs=self.cfg.n_envs, reward_weights=g.reward_weights_obj(),
            config=self.env_config, seed=int(self.rng.integers(0, 1_000_000)),
            subprocess=self.cfg.subprocess,
        )
        try:
            model = build_agent(g.algo, env, hyperparams=g.hp_for_agent(), arch=g.arch,
                                device=self.cfg.device,
                                seed=int(self.rng.integers(0, 1_000_000)))
            # Warm-start: continue from own checkpoint, or a parent's weights.
            src = ind.model_path or ind.init_from
            if src and os.path.exists(src):
                try:
                    model.set_parameters(src, exact_match=False)
                except Exception as e:  # architecture mismatch -> fresh start
                    log.debug("warm-start failed for %s: %s", g.genome_id, e)
            reset = ind.model_path is None
            model.learn(total_timesteps=timesteps, reset_num_timesteps=reset,
                        progress_bar=False)
            path = self._agent_paths(g.genome_id)
            model.save(path)
            # Persist the genome next to the weights so the agent can be
            # reloaded and replayed standalone (scripts/evaluate.py).
            import json as _json
            with open(os.path.join(os.path.dirname(path), "genome.json"), "w") as fh:
                _json.dump(g.to_dict(), fh, indent=2)
            ind.model_path = path
        finally:
            env.close()
        return timesteps

    def _default_eval(self, ind: Individual) -> EvalResult:
        g = ind.genome
        env = build_single_env(reward_weights=g.reward_weights_obj(),
                               config=self.env_config, seed=self.cfg.eval_seed)
        from ..agents.factory import build_agent as _b
        # Reload model for evaluation.
        model = _b(g.algo, env, hyperparams=g.hp_for_agent(), arch=g.arch,
                   device=self.cfg.device)
        if ind.model_path and os.path.exists(ind.model_path):
            try:
                model.set_parameters(ind.model_path, exact_match=False)
            except Exception:
                pass
        result = evaluate_policy(model, env=env, record_telemetry=True,
                                 record_skeleton=True)
        env.close()
        return result

    # ------------------------------------------------------------------ #
    def train_generation(self, timesteps_override: Optional[int] = None) -> Dict[str, Any]:
        """Run one full PBT generation and return summary stats."""
        t0 = time.time()
        base = timesteps_override or self.cfg.timesteps_per_gen
        for ind in self.individuals:
            budget = max(256, int(base * ind.genome.train_budget_mult))
            trained = self._train_fn(ind, budget)
            ind.total_timesteps += trained
            self.total_timesteps += trained
            result = self._eval_fn(ind)
            ind.fitness = result.fitness
            ind.last_result = result.to_metrics()
            ind.telemetry = getattr(result, "telemetry", {}) or {}
            self.db.log_evaluation(self.exp_id, ind.genome.genome_id,
                                   self.generation, {
                                       **result.to_metrics(),
                                       "total_timesteps": ind.total_timesteps,
                                   })
            self._maybe_record(ind, result)

        self.individuals.sort(key=lambda i: i.fitness, reverse=True)
        best = self.individuals[0]
        if self.best is None or best.fitness > self.best.fitness:
            self.best = best
        # Publish the best agent's telemetry so the dashboard shows live curves
        # even before any agent completes a full mile.
        if best.telemetry:
            self.latest_best_telemetry = best.telemetry
        fits = [i.fitness for i in self.individuals]
        best_mile = None
        if best.last_result and best.last_result.get("finished"):
            best_mile = best.last_result.get("mile_time")

        self.db.log_generation(
            self.exp_id, self.generation,
            best_fitness=float(best.fitness),
            best_mile_time=best_mile,
            mean_fitness=float(np.mean(fits)),
            population_size=len(self.individuals),
            total_timesteps=self.total_timesteps,
        )
        summary = {
            "generation": self.generation,
            "best_fitness": float(best.fitness),
            "mean_fitness": float(np.mean(fits)),
            "best_mile_time": best_mile,
            "best_genome": best.genome.genome_id,
            "best_algo": best.genome.algo,
            "best_arch": best.genome.arch,
            "total_timesteps": self.total_timesteps,
            "seconds": time.time() - t0,
            "n_finished": sum(1 for i in self.individuals
                              if i.last_result and i.last_result.get("finished")),
        }
        log.info("Gen %d | best_fit=%.1f mean=%.1f mile=%s finished=%d/%d steps=%d (%.1fs)",
                 self.generation, summary["best_fitness"], summary["mean_fitness"],
                 f"{best_mile:.1f}s" if best_mile else "-", summary["n_finished"],
                 len(self.individuals), self.total_timesteps, summary["seconds"])
        return summary

    # ------------------------------------------------------------------ #
    def _maybe_record(self, ind: Individual, result: EvalResult) -> None:
        if result.finished and result.mile_time:
            if self.best_record is None or result.mile_time < self.best_record["mile_time"]:
                path = ind.model_path or ""
                self.db.log_record(self.exp_id, ind.genome.genome_id,
                                   result.mile_time, self.generation, path,
                                   telemetry=result.telemetry)
                self.db.log_checkpoint(self.exp_id, ind.genome.genome_id,
                                      self.generation, path, result.mile_time,
                                      ind.fitness, is_best=True)
                self.best_record = {
                    "mile_time": result.mile_time, "genome_id": ind.genome.genome_id,
                    "generation": self.generation, "path": path,
                }
                log.info("NEW BEST MILE: %.2fs by %s (gen %d)",
                         result.mile_time, ind.genome.genome_id, self.generation)

    # ------------------------------------------------------------------ #
    def evolve(self) -> None:
        """Select elites, cull the rest, breed mutated offspring."""
        self.individuals.sort(key=lambda i: i.fitness, reverse=True)
        n = len(self.individuals)
        n_elite = max(1, int(round(n * self.cfg.elite_frac)))
        elites = self.individuals[:n_elite]
        for e in elites:
            e.age += 1

        next_gen = self.generation + 1
        offspring: List[Individual] = []
        n_children = n - n_elite
        for _ in range(n_children):
            p1 = self._tournament(elites if len(elites) > 1 else self.individuals)
            p2 = self._tournament(elites if len(elites) > 1 else self.individuals)
            if p1.genome.genome_id != p2.genome.genome_id and self.rng.random() < 0.7:
                child_g = crossover(p1.genome, p2.genome, self.rng, generation=next_gen)
                child_g = mutate(child_g, self.rng, strength=self.cfg.mutation_strength,
                                 algos=self.cfg.algos, generation=next_gen)
                parent = p1 if self.rng.random() < 0.5 else p2
            else:
                child_g = mutate(p1.genome, self.rng, strength=self.cfg.mutation_strength,
                                 algos=self.cfg.algos, generation=next_gen)
                parent = p1
            child = Individual(genome=child_g)
            # Warm-start from the parent when architectures match.
            if self.cfg.warm_start and parent.model_path and \
                    self._arch_match(child_g, parent.genome):
                child.init_from = parent.model_path
            self.db.log_individual(self.exp_id, child_g.to_dict())
            offspring.append(child)

        # cull culled agents' large checkpoints to save disk (keep elites + best)
        self._gc_checkpoints(keep=[i.genome.genome_id for i in elites])
        self.individuals = elites + offspring
        self.generation = next_gen

    @staticmethod
    def _arch_match(a: Genome, b: Genome) -> bool:
        keys = ("hidden", "depth", "features_dim")
        return (a.algo == b.algo and a.arch == b.arch and a.activation == b.activation
                and all(a.hyperparams.get(k) == b.hyperparams.get(k) for k in keys))

    def _tournament(self, pool: List[Individual]) -> Individual:
        k = min(self.cfg.tournament_k, len(pool))
        contenders = list(self.rng.choice(pool, size=k, replace=False))
        return max(contenders, key=lambda i: i.fitness)

    def _gc_checkpoints(self, keep: List[str]) -> None:
        keep_set = set(keep)
        if self.best_record:
            keep_set.add(self.best_record["genome_id"])
        base = os.path.join(self.root, self.experiment_name)
        if not os.path.isdir(base):
            return
        for gid in os.listdir(base):
            if gid not in keep_set:
                p = os.path.join(base, gid)
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)

    # ------------------------------------------------------------------ #
    def state_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "total_timesteps": self.total_timesteps,
            "rng_state": self.rng.bit_generator.state,
            "best_record": self.best_record,
            "individuals": [
                {
                    "genome": i.genome.to_dict(),
                    "model_path": i.model_path,
                    "fitness": i.fitness,
                    "total_timesteps": i.total_timesteps,
                    "age": i.age,
                    "last_result": i.last_result,
                }
                for i in self.individuals
            ],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.generation = state["generation"]
        self.total_timesteps = state["total_timesteps"]
        try:
            self.rng.bit_generator.state = state["rng_state"]
        except Exception:  # pragma: no cover
            pass
        self.best_record = state.get("best_record")
        self.individuals = []
        for d in state["individuals"]:
            ind = Individual(
                genome=Genome.from_dict(d["genome"]),
                model_path=d.get("model_path"),
                fitness=d.get("fitness", -1e9),
                total_timesteps=d.get("total_timesteps", 0),
                age=d.get("age", 0),
                last_result=d.get("last_result"),
            )
            self.individuals.append(ind)
        if self.individuals:
            self.best = max(self.individuals, key=lambda i: i.fitness)
