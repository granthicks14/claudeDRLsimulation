"""Tests for database, config, rewards, networks and tournament."""
import numpy as np
import pytest

from milerunner.database.experiment_db import ExperimentDB
from milerunner.envs.rewards import RewardWeights, compute_reward
from milerunner.utils.config import Config, load_config


def test_config_load_default_and_inheritance():
    cfg = load_config("default")
    assert cfg.get("body.mass_kg") == 77.0
    assert cfg.get("env.physics_timestep") == 0.001
    smoke = load_config("smoke")
    # smoke inherits default then overrides
    assert smoke.get("body.mass_kg") == 77.0
    assert smoke.get("population.size") == 6
    assert smoke.get("trainer.max_generations") == 3


def test_config_overrides_and_dotted_access():
    cfg = load_config("default", overrides={"population": {"size": 99}})
    assert cfg.get("population.size") == 99
    assert cfg.get("nonexistent.key", "d") == "d"


def test_reward_faster_finish_scores_higher():
    w = RewardWeights()
    common = dict(dprogress=0.05, speed=6.0, uprightness=1.0, lateral_speed=0.0,
                  metabolic_power=1200, mass=77, overexertion=0.0, joint_violation=0.0,
                  action_sq=0.1, fell=False, exhausted=False, dt=0.01)
    fast = compute_reward(w, finished=True, finish_time=230, **common)
    slow = compute_reward(w, finished=True, finish_time=300, **common)
    assert fast.total > slow.total  # faster mile -> more reward


def test_reward_penalizes_falling():
    w = RewardWeights()
    common = dict(dprogress=0.0, speed=0.0, uprightness=0.2, lateral_speed=0.0,
                  metabolic_power=300, mass=77, overexertion=0.0, joint_violation=0.0,
                  action_sq=0.1, finished=False, finish_time=0, exhausted=False, dt=0.01)
    fell = compute_reward(w, fell=True, **common)
    ok = compute_reward(w, fell=False, **common)
    assert fell.total < ok.total
    assert fell.terms["fall"] < 0


def test_reward_weights_roundtrip():
    w = RewardWeights(progress=2.0, finish_bonus=99)
    w2 = RewardWeights.from_dict(w.to_dict())
    assert w2.progress == 2.0 and w2.finish_bonus == 99


def test_database_logging_and_queries(tmp_path):
    db = ExperimentDB(str(tmp_path / "db.sqlite"))
    exp = db.create_experiment("exp1", {"a": 1})
    assert db.get_or_create_experiment("exp1", {}) == exp  # idempotent-ish
    genome = {"genome_id": "g1", "generation": 0, "algo": "ppo", "arch": "mlp",
              "parent_ids": []}
    db.log_individual(exp, genome)
    db.log_evaluation(exp, "g1", 0, {"fitness": 10.0, "mile_time": 300.0,
                                     "finished": True, "distance": 1609,
                                     "mean_speed": 5.4, "peak_speed": 6.0,
                                     "mean_cadence": 3.0, "mean_hr": 180,
                                     "total_timesteps": 1000, "metrics": {}})
    db.log_record(exp, "g1", 300.0, 0, "path/model.zip", {"speed": [1, 2]})
    db.log_generation(exp, 0, best_fitness=10.0, best_mile_time=300.0,
                      mean_fitness=5.0, population_size=8, total_timesteps=1000)
    best = db.best_mile_time(exp)
    assert best["mile_time"] == 300.0
    lb = db.leaderboard(exp)
    assert lb and lb[0]["genome_id"] == "g1"
    summary = db.experiment_summary(exp)
    assert summary["generations"] == 1 and summary["individuals"] == 1
    hist = db.generation_history(exp)
    assert len(hist) == 1


def test_networks_forward_shapes():
    torch = pytest.importorskip("torch")
    import gymnasium as gym
    from milerunner.agents.networks import (CNN1DExtractor, GRUMemoryExtractor,
                                           MLPExtractor, TransformerExtractor)
    space = gym.spaces.Box(-1, 1, shape=(74,), dtype=np.float32)
    x = torch.randn(5, 74)
    for cls in (MLPExtractor, CNN1DExtractor, TransformerExtractor, GRUMemoryExtractor):
        ext = cls(space, features_dim=64)
        out = ext(x)
        assert out.shape == (5, 64)
        assert torch.isfinite(out).all()


def test_tournament_ranks_and_aggregates():
    from types import SimpleNamespace
    from milerunner.training.tournament import aggregate_by, run_tournament

    def ind(gid, algo, arch, fit, mile, finished):
        return SimpleNamespace(
            genome=SimpleNamespace(genome_id=gid, algo=algo, arch=arch),
            fitness=fit, total_timesteps=100,
            last_result={"mile_time": mile, "finished": finished, "mean_speed": 5.0})

    inds = [ind("a", "ppo", "mlp", 30, 250, True),
            ind("b", "sac", "gru", 50, 240, True),
            ind("c", "ppo", "cnn", 10, None, False)]
    rows = run_tournament(inds)
    assert rows[0].genome_id == "b"  # highest fitness ranked first
    agg = aggregate_by(rows, "algo")
    assert agg["ppo"]["count"] == 2
    assert agg["sac"]["best_mile_time"] == 240
