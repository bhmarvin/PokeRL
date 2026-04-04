"""Checkpoint pool for PFSP-style self-play training.

Manages a directory of historical checkpoints and assigns them to
SubprocVecEnv workers via per-worker files. Workers pick up new
opponents through SelfPlayPlayer's mtime-based refresh.
"""
from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class PoolEntry:
    name: str
    path: str  # path inside pool directory
    wins: int = 0
    losses: int = 0
    games: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games > 0 else 0.5


class CheckpointPool:
    """Manages a pool of historical checkpoints for league-style self-play.

    Workers each watch a unique file (opponent_0.zip, opponent_1.zip, etc.).
    The pool periodically reassigns which checkpoint each worker's file
    points to, using PFSP sampling to prioritize hard opponents.
    """

    def __init__(self, pool_dir: str, max_size: int = 8):
        self.pool_dir = pool_dir
        self.max_size = max_size
        self.entries: list[PoolEntry] = []
        self._generation = 0
        os.makedirs(pool_dir, exist_ok=True)
        self._load_manifest()

    def _manifest_path(self) -> str:
        return os.path.join(self.pool_dir, "manifest.json")

    def _load_manifest(self) -> None:
        path = self._manifest_path()
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self._generation = data.get("generation", 0)
        self.entries = [PoolEntry(**e) for e in data.get("entries", [])]

    def save_manifest(self) -> None:
        data = {
            "generation": self._generation,
            "entries": [asdict(e) for e in self.entries],
        }
        path = self._manifest_path()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def add(self, checkpoint_path: str, name: Optional[str] = None) -> str:
        """Copy checkpoint into pool. Returns pool path. Evicts oldest if full."""
        if name is None:
            name = f"gen_{self._generation}"
        self._generation += 1

        pool_path = os.path.join(self.pool_dir, f"{name}.zip")
        shutil.copy2(checkpoint_path, pool_path)

        self.entries.append(PoolEntry(name=name, path=pool_path))

        # Evict oldest if over capacity (keep newest)
        while len(self.entries) > self.max_size:
            evicted = self.entries.pop(0)
            if os.path.exists(evicted.path):
                os.remove(evicted.path)
            print(f"  [Pool] evicted {evicted.name} (WR={evicted.win_rate:.0%})", flush=True)

        self.save_manifest()
        print(
            f"  [Pool] added {name} (pool size={len(self.entries)})",
            flush=True,
        )
        return pool_path

    def update_result(self, pool_name: str, won: bool) -> None:
        """Record a win/loss against a pool checkpoint."""
        for entry in self.entries:
            if entry.name == pool_name:
                entry.games += 1
                if won:
                    entry.wins += 1
                else:
                    entry.losses += 1
                return

    def sample_pfsp(self, n: int) -> list[PoolEntry]:
        """Sample n entries weighted by difficulty (PFSP).

        Prioritizes opponents the agent struggles against:
        weight ∝ (1 - win_rate)^2
        Unexplored entries (< 5 games) get weight 1.0.
        """
        if not self.entries:
            return []

        weights = []
        for entry in self.entries:
            if entry.games < 5:
                w = 1.0  # explore unknown opponents
            else:
                w = max((1.0 - entry.win_rate) ** 2, 0.01)  # floor to avoid zero
            weights.append(w)

        return random.choices(self.entries, weights=weights, k=n)

    def assign_to_workers(self, worker_paths: list[str]) -> list[str]:
        """Sample from pool and copy to each worker's watched file.

        Returns list of pool entry names assigned to each worker.
        """
        if not self.entries:
            return []

        sampled = self.sample_pfsp(len(worker_paths))
        assignments = []

        for worker_path, entry in zip(worker_paths, sampled):
            shutil.copy2(entry.path, worker_path)
            assignments.append(entry.name)

        return assignments

    def setup_worker_paths(self, n_workers: int) -> list[str]:
        """Create per-worker checkpoint file paths."""
        paths = []
        for i in range(n_workers):
            path = os.path.join(self.pool_dir, f"opponent_{i}.zip")
            paths.append(path)
        return paths

    def seed(self, checkpoint_path: str, name: str = "seed") -> None:
        """Seed the pool with an initial checkpoint if empty."""
        if self.entries:
            return
        self.add(checkpoint_path, name=name)

    def summary(self) -> str:
        """Human-readable pool status."""
        lines = [f"Pool ({len(self.entries)}/{self.max_size}):"]
        for e in self.entries:
            wr = f"{e.win_rate:.0%}" if e.games >= 5 else "?"
            lines.append(f"  {e.name}: WR={wr} ({e.wins}W/{e.losses}L, {e.games}G)")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.entries)
