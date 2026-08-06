"""Cross-algorithm tournament.

All agents in a population compete on the same mile. This module ranks them and
aggregates results by RL algorithm and by network architecture, so the platform
can report which *families* of methods are discovering the fastest miles — one
of the research questions. It performs no training; it reads evaluation results.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class TournamentRow:
    rank: int
    genome_id: str
    algo: str
    arch: str
    fitness: float
    mile_time: Optional[float]
    finished: bool
    mean_speed: float
    total_timesteps: int


def run_tournament(individuals) -> List[TournamentRow]:
    """Rank a list of Individuals (must have ``last_result``)."""
    scored = [i for i in individuals if i.last_result is not None]
    scored.sort(key=lambda i: i.fitness, reverse=True)
    rows: List[TournamentRow] = []
    for rank, ind in enumerate(scored, 1):
        r = ind.last_result
        rows.append(TournamentRow(
            rank=rank, genome_id=ind.genome.genome_id, algo=ind.genome.algo,
            arch=ind.genome.arch, fitness=ind.fitness,
            mile_time=r.get("mile_time"), finished=bool(r.get("finished")),
            mean_speed=float(r.get("mean_speed", 0.0)),
            total_timesteps=ind.total_timesteps,
        ))
    return rows


def aggregate_by(rows: List[TournamentRow], key: str) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[TournamentRow]] = defaultdict(list)
    for r in rows:
        groups[getattr(r, key)].append(r)
    out: Dict[str, Dict[str, float]] = {}
    for name, group in groups.items():
        fits = [g.fitness for g in group]
        miles = [g.mile_time for g in group if g.mile_time is not None]
        out[name] = {
            "count": len(group),
            "mean_fitness": float(np.mean(fits)),
            "best_fitness": float(np.max(fits)),
            "best_mile_time": float(np.min(miles)) if miles else None,
            "finish_rate": float(np.mean([g.finished for g in group])),
        }
    return out


def format_leaderboard(rows: List[TournamentRow], top: int = 10) -> str:
    lines = [f"{'#':>3} {'genome':>12} {'algo':>8} {'arch':>11} "
             f"{'fitness':>9} {'mile':>8} {'speed':>6} {'steps':>10}"]
    for r in rows[:top]:
        mile = f"{r.mile_time:.1f}s" if r.mile_time else "  -"
        lines.append(f"{r.rank:>3} {r.genome_id:>12} {r.algo:>8} {r.arch:>11} "
                     f"{r.fitness:>9.1f} {mile:>8} {r.mean_speed:>6.2f} {r.total_timesteps:>10}")
    return "\n".join(lines)
