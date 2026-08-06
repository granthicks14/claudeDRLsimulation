"""Tests for genomes and population-based evolutionary training."""
import numpy as np
import pytest

from milerunner.evolution.genome import (ARCHES, CONT_BOUNDS, INT_BOUNDS,
                                        Genome, crossover, mutate,
                                        random_genome)


def test_random_genome_within_bounds():
    rng = np.random.default_rng(0)
    for _ in range(50):
        g = random_genome(rng)
        for name, (lo, hi, _) in CONT_BOUNDS.items():
            assert lo <= g.hyperparams[name] <= hi
        for name, (lo, hi) in INT_BOUNDS.items():
            assert lo <= g.hyperparams[name] <= hi
        assert g.arch in ARCHES


def test_mutation_stays_in_bounds_and_changes_id():
    rng = np.random.default_rng(1)
    g = random_genome(rng)
    for _ in range(100):
        c = mutate(g, rng, strength=0.4)
        assert c.genome_id != g.genome_id
        assert g.genome_id in c.parent_ids
        for name, (lo, hi, _) in CONT_BOUNDS.items():
            assert lo <= c.hyperparams[name] <= hi
        g = c


def test_crossover_mixes_parents():
    rng = np.random.default_rng(2)
    a = random_genome(rng)
    b = random_genome(rng)
    child = crossover(a, b, rng)
    assert child.algo in (a.algo, b.algo)
    assert child.arch in (a.arch, b.arch)
    assert set(child.parent_ids) == {a.genome_id, b.genome_id}


def test_genome_roundtrip():
    rng = np.random.default_rng(3)
    g = random_genome(rng)
    g2 = Genome.from_dict(g.to_dict())
    assert g2.algo == g.algo and g2.hyperparams == g.hyperparams


def _make_stub_population(tmp_path, size=8):
    """Population whose train/eval are stubbed so tests run without RL."""
    from milerunner.database.experiment_db import ExperimentDB
    from milerunner.envs.mile_env import EnvConfig
    from milerunner.evolution.population import Population, PopulationConfig
    from milerunner.training.evaluation import EvalResult

    db = ExperimentDB(str(tmp_path / "t.db"))
    exp = db.create_experiment("t", {})
    cfg = PopulationConfig(size=size, elite_frac=0.25, timesteps_per_gen=10)

    def fake_train(ind, timesteps):
        return timesteps

    # deterministic fitness so we can assert selection behaviour:
    # fitness proportional to learning_rate for reproducibility.
    def fake_eval(ind):
        lr = ind.genome.hyperparams.get("learning_rate", 1e-4)
        fit = float(lr * 1e5) + ind.total_timesteps * 0.0
        return EvalResult(fitness=fit, finished=False, mile_time=None, distance=100.0,
                          mean_speed=3.0, peak_speed=4.0, mean_cadence=2.8, mean_hr=150,
                          mean_stride=1.3, total_reward=1.0, steps=10)

    pop = Population(cfg, EnvConfig(), db, exp, "t", root=str(tmp_path / "ck"),
                    seed=0, train_fn=fake_train, eval_fn=fake_eval)
    return pop, db, exp


def test_population_evolution_preserves_size_and_elites(tmp_path):
    pop, db, exp = _make_stub_population(tmp_path, size=8)
    pop.initialize()
    assert len(pop.individuals) == 8
    pop.train_generation()
    best_before = max(i.fitness for i in pop.individuals)
    top_ids_before = {i.genome.genome_id for i in
                      sorted(pop.individuals, key=lambda x: x.fitness, reverse=True)[:2]}
    pop.evolve()
    assert len(pop.individuals) == 8  # size preserved
    assert pop.generation == 1
    # elites (top 25% -> 2) survive into the next generation
    surviving = {i.genome.genome_id for i in pop.individuals}
    assert top_ids_before & surviving


def test_population_improves_or_holds_best(tmp_path):
    pop, db, exp = _make_stub_population(tmp_path, size=10)
    pop.initialize()
    bests = []
    for _ in range(4):
        s = pop.train_generation()
        bests.append(s["best_fitness"])
        pop.evolve()
    # best fitness should be non-decreasing across generations (elitism).
    assert bests[-1] >= bests[0] - 1e-6


def test_population_state_roundtrip(tmp_path):
    import json
    pop, db, exp = _make_stub_population(tmp_path, size=6)
    pop.initialize()
    pop.train_generation()
    pop.evolve()
    state = json.loads(json.dumps(pop.state_dict(), default=str))
    pop2, _, _ = _make_stub_population(tmp_path, size=6)
    pop2.load_state_dict(state)
    assert pop2.generation == pop.generation
    assert len(pop2.individuals) == len(pop.individuals)
