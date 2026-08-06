"""SQLite experiment database.

Persists a complete record of the search: every experiment, every generation,
every individual (genome), and every evaluation result — plus a registry of the
best mile times and best-ever model checkpoints. This is what lets training be
paused and resumed indefinitely without losing progress, and what backs the
analytics dashboard.

The schema is intentionally simple and append-mostly so it stays fast even
after millions of episodes.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at REAL NOT NULL,
    config_json TEXT NOT NULL,
    status TEXT DEFAULT 'running'
);
CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    gen_index INTEGER NOT NULL,
    created_at REAL NOT NULL,
    best_fitness REAL,
    best_mile_time REAL,
    mean_fitness REAL,
    population_size INTEGER,
    total_timesteps INTEGER
);
CREATE TABLE IF NOT EXISTS individuals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    genome_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    algo TEXT,
    arch TEXT,
    genome_json TEXT NOT NULL,
    parent_ids TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    genome_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    fitness REAL,
    mile_time REAL,
    finished INTEGER,
    distance REAL,
    mean_speed REAL,
    peak_speed REAL,
    mean_cadence REAL,
    mean_hr REAL,
    total_timesteps INTEGER,
    metrics_json TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    genome_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    path TEXT NOT NULL,
    mile_time REAL,
    fitness REAL,
    is_best INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER,
    genome_id TEXT,
    mile_time REAL,
    generation INTEGER,
    checkpoint_path TEXT,
    telemetry_json TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_exp ON evaluations(experiment_id, generation);
CREATE INDEX IF NOT EXISTS idx_gen_exp ON generations(experiment_id, gen_index);
"""


class ExperimentDB:
    def __init__(self, path: str = "experiments/milerunner.db"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @contextmanager
    def _tx(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        self._conn.close()

    # ---------------- experiments ---------------- #
    def create_experiment(self, name: str, config: Dict[str, Any]) -> int:
        with self._tx() as c:
            cur = c.execute(
                "INSERT INTO experiments(name, created_at, config_json) VALUES (?,?,?)",
                (name, time.time(), json.dumps(config)),
            )
            return int(cur.lastrowid)

    def get_or_create_experiment(self, name: str, config: Dict[str, Any]) -> int:
        row = self._conn.execute(
            "SELECT id FROM experiments WHERE name=? ORDER BY id DESC LIMIT 1", (name,)
        ).fetchone()
        if row is not None:
            return int(row["id"])
        return self.create_experiment(name, config)

    def set_experiment_status(self, exp_id: int, status: str) -> None:
        with self._tx() as c:
            c.execute("UPDATE experiments SET status=? WHERE id=?", (status, exp_id))

    # ---------------- generations ---------------- #
    def log_generation(self, exp_id: int, gen_index: int, *, best_fitness: float,
                       best_mile_time: Optional[float], mean_fitness: float,
                       population_size: int, total_timesteps: int) -> None:
        with self._tx() as c:
            c.execute(
                """INSERT INTO generations(experiment_id, gen_index, created_at,
                   best_fitness, best_mile_time, mean_fitness, population_size, total_timesteps)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (exp_id, gen_index, time.time(), best_fitness, best_mile_time,
                 mean_fitness, population_size, total_timesteps),
            )

    def last_generation_index(self, exp_id: int) -> int:
        row = self._conn.execute(
            "SELECT MAX(gen_index) AS g FROM generations WHERE experiment_id=?", (exp_id,)
        ).fetchone()
        return int(row["g"]) if row and row["g"] is not None else -1

    # ---------------- individuals ---------------- #
    def log_individual(self, exp_id: int, genome: Dict[str, Any]) -> None:
        with self._tx() as c:
            c.execute(
                """INSERT INTO individuals(experiment_id, genome_id, generation, algo, arch,
                   genome_json, parent_ids, created_at) VALUES (?,?,?,?,?,?,?,?)""",
                (exp_id, genome["genome_id"], genome.get("generation", 0),
                 genome.get("algo"), genome.get("arch"), json.dumps(genome),
                 json.dumps(genome.get("parent_ids", [])), time.time()),
            )

    # ---------------- evaluations ---------------- #
    def log_evaluation(self, exp_id: int, genome_id: str, generation: int,
                       result: Dict[str, Any]) -> None:
        with self._tx() as c:
            c.execute(
                """INSERT INTO evaluations(experiment_id, genome_id, generation, fitness,
                   mile_time, finished, distance, mean_speed, peak_speed, mean_cadence,
                   mean_hr, total_timesteps, metrics_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (exp_id, genome_id, generation, result.get("fitness"),
                 result.get("mile_time"), int(bool(result.get("finished"))),
                 result.get("distance"), result.get("mean_speed"), result.get("peak_speed"),
                 result.get("mean_cadence"), result.get("mean_hr"),
                 result.get("total_timesteps"), json.dumps(result.get("metrics", {})),
                 time.time()),
            )

    # ---------------- checkpoints & records ---------------- #
    def log_checkpoint(self, exp_id: int, genome_id: str, generation: int, path: str,
                       mile_time: Optional[float], fitness: float, is_best: bool = False) -> None:
        with self._tx() as c:
            c.execute(
                """INSERT INTO checkpoints(experiment_id, genome_id, generation, path,
                   mile_time, fitness, is_best, created_at) VALUES (?,?,?,?,?,?,?,?)""",
                (exp_id, genome_id, generation, path, mile_time, fitness, int(is_best), time.time()),
            )

    def log_record(self, exp_id: int, genome_id: str, mile_time: float, generation: int,
                   checkpoint_path: str, telemetry: Optional[Dict] = None) -> None:
        with self._tx() as c:
            c.execute(
                """INSERT INTO records(experiment_id, genome_id, mile_time, generation,
                   checkpoint_path, telemetry_json, created_at) VALUES (?,?,?,?,?,?,?)""",
                (exp_id, genome_id, mile_time, generation, checkpoint_path,
                 json.dumps(telemetry or {}), time.time()),
            )

    def best_mile_time(self, exp_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        q = "SELECT * FROM records"
        args: tuple = ()
        if exp_id is not None:
            q += " WHERE experiment_id=?"
            args = (exp_id,)
        q += " ORDER BY mile_time ASC LIMIT 1"
        row = self._conn.execute(q, args).fetchone()
        return dict(row) if row else None

    # ---------------- queries for dashboard ---------------- #
    def generation_history(self, exp_id: int) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM generations WHERE experiment_id=? ORDER BY gen_index", (exp_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def leaderboard(self, exp_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT genome_id, MIN(mile_time) AS mile_time, MAX(fitness) AS fitness,
               MAX(finished) AS finished, MAX(generation) AS generation
               FROM evaluations WHERE experiment_id=? AND mile_time IS NOT NULL
               GROUP BY genome_id ORDER BY mile_time ASC LIMIT ?""",
            (exp_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def best_by_fitness(self, exp_id: int) -> Optional[Dict[str, Any]]:
        """Highest-fitness agent regardless of whether it finished a mile."""
        row = self._conn.execute(
            """SELECT genome_id, MAX(fitness) AS fitness, MAX(generation) AS generation
               FROM evaluations WHERE experiment_id=? GROUP BY genome_id
               ORDER BY fitness DESC LIMIT 1""", (exp_id,)).fetchone()
        return dict(row) if row else None

    def experiment_summary(self, exp_id: int) -> Dict[str, Any]:
        exp = self._conn.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()
        gens = self._conn.execute(
            "SELECT COUNT(*) AS n, MAX(total_timesteps) AS steps FROM generations WHERE experiment_id=?",
            (exp_id,)).fetchone()
        n_ind = self._conn.execute(
            "SELECT COUNT(*) AS n FROM individuals WHERE experiment_id=?", (exp_id,)).fetchone()
        best = self.best_mile_time(exp_id)
        return {
            "experiment": dict(exp) if exp else None,
            "generations": int(gens["n"]) if gens else 0,
            "total_timesteps": int(gens["steps"] or 0) if gens else 0,
            "individuals": int(n_ind["n"]) if n_ind else 0,
            "best_record": best,
        }

    def list_experiments(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM experiments ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]
