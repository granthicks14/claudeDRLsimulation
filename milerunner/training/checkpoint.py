"""Checkpoint saving/loading for models, genomes and the whole search state.

Two levels of persistence:

* **Per-agent checkpoints** — an SB3 ``.zip`` plus the genome JSON, so any agent
  can be reloaded and replayed.
* **Search state** — a single JSON snapshot of the population's genomes, the
  current generation, RNG state, cumulative timesteps and best records, so the
  whole continuous run can be paused and resumed exactly where it left off.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ..evolution.genome import Genome


def agent_dir(root: str, experiment: str, genome_id: str) -> str:
    d = os.path.join(root, experiment, genome_id)
    os.makedirs(d, exist_ok=True)
    return d


def save_agent(model, genome: Genome, root: str, experiment: str,
               extra: Optional[Dict[str, Any]] = None) -> str:
    """Save an SB3 model + its genome. Returns the model zip path."""
    d = agent_dir(root, experiment, genome.genome_id)
    model_path = os.path.join(d, "model.zip")
    model.save(model_path)
    with open(os.path.join(d, "genome.json"), "w") as fh:
        json.dump(genome.to_dict(), fh, indent=2)
    if extra:
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(extra, fh, indent=2)
    return model_path


def load_genome(root: str, experiment: str, genome_id: str) -> Genome:
    with open(os.path.join(root, experiment, genome_id, "genome.json")) as fh:
        return Genome.from_dict(json.load(fh))


def load_agent(algo_builder, root: str, experiment: str, genome_id: str, env,
               device: str = "auto"):
    """Reload an SB3 model. ``algo_builder`` maps algo name -> SB3 class."""
    model_path = os.path.join(root, experiment, genome_id, "model.zip")
    genome = load_genome(root, experiment, genome_id)
    cls = algo_builder(genome.algo)
    return cls.load(model_path, env=env, device=device), genome


def save_search_state(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, path)   # atomic


def load_search_state(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)
