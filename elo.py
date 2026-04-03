"""Elo rating tracker for PokeRL checkpoints.

Anchors SimpleHeuristicsPlayer at 1000 Elo. Updates ratings after
benchmark runs. Persists to results/elo_ratings.json.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional


DEFAULT_ELO = 1000
K_FACTOR = 32
ELO_FILE = os.path.join("results", "elo_ratings.json")

# Fixed anchor ratings for baseline opponents
ANCHOR_RATINGS = {
    "random": 400,
    "max_base_power": 700,
    "simple_heuristic": 1000,
}


@dataclass
class EloEntry:
    name: str
    rating: float = DEFAULT_ELO
    games: int = 0
    wins: int = 0
    losses: int = 0


class EloTracker:
    def __init__(self, path: str = ELO_FILE):
        self.path = path
        self.ratings: dict[str, EloEntry] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path) as f:
                data = json.load(f)
            for name, entry in data.items():
                self.ratings[name] = EloEntry(**entry)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(
                {name: asdict(entry) for name, entry in self.ratings.items()},
                f, indent=2,
            )

    def get_rating(self, name: str) -> float:
        if name in ANCHOR_RATINGS:
            return ANCHOR_RATINGS[name]
        if name in self.ratings:
            return self.ratings[name].rating
        return DEFAULT_ELO

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update_from_match(
        self,
        checkpoint_name: str,
        opponent_name: str,
        wins: int,
        losses: int,
        draws: int = 0,
    ) -> float:
        """Update Elo for checkpoint after a series of games vs opponent.
        Returns the new rating."""
        if checkpoint_name not in self.ratings:
            self.ratings[checkpoint_name] = EloEntry(name=checkpoint_name)

        entry = self.ratings[checkpoint_name]
        opp_rating = self.get_rating(opponent_name)
        total = wins + losses + draws

        for _ in range(wins):
            expected = self.expected_score(entry.rating, opp_rating)
            entry.rating += K_FACTOR * (1.0 - expected)
        for _ in range(losses):
            expected = self.expected_score(entry.rating, opp_rating)
            entry.rating += K_FACTOR * (0.0 - expected)
        for _ in range(draws):
            expected = self.expected_score(entry.rating, opp_rating)
            entry.rating += K_FACTOR * (0.5 - expected)

        entry.games += total
        entry.wins += wins
        entry.losses += losses
        self.save()
        return entry.rating

    def update_from_benchmark(self, checkpoint_name: str, summary: dict) -> float:
        """Update Elo from a benchmark summary.json dict."""
        for matchup in summary.get("matchups", []):
            self.update_from_match(
                checkpoint_name,
                matchup["opponent"],
                matchup["wins"],
                matchup["losses"],
                matchup.get("draws", 0),
            )
        return self.get_rating(checkpoint_name)

    def leaderboard(self) -> list[EloEntry]:
        """Return all entries sorted by rating descending."""
        all_entries = list(self.ratings.values())
        # Add anchors for display
        for name, rating in ANCHOR_RATINGS.items():
            if name not in self.ratings:
                all_entries.append(EloEntry(name=name, rating=rating))
        return sorted(all_entries, key=lambda e: e.rating, reverse=True)

    def print_leaderboard(self) -> None:
        print(f"{'Name':<30} {'Elo':>6} {'W':>5} {'L':>5} {'Games':>6}")
        print(f"{'-'*30} {'-'*6} {'-'*5} {'-'*5} {'-'*6}")
        for entry in self.leaderboard():
            print(f"{entry.name:<30} {entry.rating:>6.0f} {entry.wins:>5} {entry.losses:>5} {entry.games:>6}")
