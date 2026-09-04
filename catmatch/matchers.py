"""The two matching algorithms.

Both take the same inputs -- one vector per input record, one vector per
taxonomy element -- and return a single path plus its similarity.

    weighed_embedding  collapse each taxonomy path into one weighted vector and
                       compare all of them at once (the original algorithm)
    tree_based         walk down the taxonomy one level at a time, always taking
                       the best sibling (greedy, top-1)

`flexible` decides where a match stops: fixed at `max_level`, or anywhere
shallower when going deeper stops helping.
"""

import math
from dataclasses import dataclass

import numpy as np

from .embedding import l2_normalize


def weight(k: int, p: int, aj: float = 2.5) -> float:
    """Weight of level k out of p levels (k starts at 1); the p weights sum to 1.

    Deeper levels weigh more; `aj` flattens the slope as it grows.
    """
    total = sum(math.log(aj + i) for i in range(p))
    return math.log(k + aj - 1) / total


@dataclass
class MatchResult:
    path: list[str]
    sim: float

    @property
    def level(self) -> int:
        return len(self.path)


def compose(texts: list[str], emb: dict[str, np.ndarray]) -> np.ndarray:
    """Level-weighted average of the level vectors, normalised to unit length."""
    depth = len(texts)
    vector = sum(weight(i + 1, depth) * emb[text] for i, text in enumerate(texts))
    return l2_normalize(vector)


class WeighedEmbeddingMatcher:
    """Every candidate path becomes one vector; pick the closest of them all."""

    def __init__(self, paths: list[tuple[str, ...]], emb: dict[str, np.ndarray],
                 max_level: int, flexible: bool):
        self.candidates = self._build_candidates(paths, max_level, flexible)
        if not self.candidates:
            raise ValueError(
                f"taxonomy has no path reaching level {max_level}; "
                "lower max_level or set flexible: true"
            )
        self.matrix = np.vstack([compose(list(p), emb) for p in self.candidates])

    @staticmethod
    def _build_candidates(paths: list[tuple[str, ...]], max_level: int,
                          flexible: bool) -> list[tuple[str, ...]]:
        """Fixed depth: only full-depth paths. Flexible: every prefix, 1..max_level."""
        depths = range(1, max_level + 1) if flexible else (max_level,)
        candidates: dict[tuple[str, ...], None] = {}
        for depth in depths:
            for path in paths:
                if len(path) >= depth:
                    candidates.setdefault(path[:depth], None)
        return list(candidates)

    def match(self, vector: np.ndarray) -> MatchResult:
        sims = self.matrix @ vector
        best = int(np.argmax(sims))
        return MatchResult(list(self.candidates[best]), float(sims[best]))


class TreeBasedMatcher:
    """Greedy descent: at each level pick the best sibling under what was chosen."""

    def __init__(self, paths: list[tuple[str, ...]], emb: dict[str, np.ndarray],
                 max_level: int, flexible: bool):
        self.emb = emb
        self.max_level = max_level
        self.flexible = flexible
        self.children = self._build_children(paths, max_level)
        if not self.children.get((), []):
            raise ValueError("taxonomy has no level-1 categories")

    @staticmethod
    def _build_children(paths: list[tuple[str, ...]],
                        max_level: int) -> dict[tuple[str, ...], list[str]]:
        """prefix -> the distinct category names living directly under it."""
        children: dict[tuple[str, ...], dict[str, None]] = {}
        for path in paths:
            for depth in range(min(len(path), max_level)):
                parent = path[:depth]
                children.setdefault(parent, {}).setdefault(path[depth], None)
        return {parent: list(names) for parent, names in children.items()}

    def _best_child(self, prefix: tuple[str, ...],
                    vector: np.ndarray) -> tuple[str, float] | None:
        """The closest sibling under `prefix`, or None at a leaf."""
        names = self.children.get(prefix)
        if not names:
            return None
        sims = np.array([float(self.emb[name] @ vector) for name in names])
        best = int(np.argmax(sims))
        return names[best], float(sims[best])

    def match(self, vector: np.ndarray) -> MatchResult:
        path: list[str] = []
        sim = 0.0

        while len(path) < self.max_level:
            step = self._best_child(tuple(path), vector)
            if step is None:
                break  # leaf reached before max_level -- the branch simply ends
            name, child_sim = step
            # flexible: stop as soon as going one level deeper stops helping
            if self.flexible and path and child_sim <= sim:
                break
            path.append(name)
            sim = child_sim

        return MatchResult(path, sim)
