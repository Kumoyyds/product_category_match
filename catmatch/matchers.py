"""The two matching algorithms.

Both take the same inputs -- one vector per input record, one vector per
taxonomy element -- and return the top `k` paths ranked by similarity (`k=1`
by default; `k>1` feeds an LLM final-selection step).

    weighed_embedding  collapse each taxonomy path into one weighted vector and
                       compare all of them at once (the original algorithm)
    tree_based         walk down the taxonomy one level at a time keeping the
                       best `beam_width` candidates per level, then score the
                       surviving paths with the weighed method and take the best

`flexible` decides where a match stops: fixed at `max_level`, or anywhere
shallower when a shallower path scores better (a branch that runs out of
children always stops there, whatever `flexible` says).
"""

import math
from dataclasses import dataclass

import numpy as np

from .config import MatchingConfig
from .embedding import l2_normalize

Path = tuple[str, ...]


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


def best_of(paths: list[Path], emb: dict[str, np.ndarray],
            vector: np.ndarray, k: int = 1) -> list[MatchResult]:
    """Score paths the weighed way -- one composed vector each -- and rank them.

    Returns the top `k` (descending by sim); shorter than `k` when there are
    fewer than `k` paths to choose from.
    """
    sims = np.array([float(compose(list(p), emb) @ vector) for p in paths])
    order = np.argsort(-sims)[:k]
    return [MatchResult(list(paths[i]), float(sims[i])) for i in order]


class WeighedEmbeddingMatcher:
    """Every candidate path becomes one vector; pick the closest of them all."""

    def __init__(self, paths: list[Path], emb: dict[str, np.ndarray],
                 config: MatchingConfig):
        self.candidates = self._build_candidates(paths, config)
        if not self.candidates:
            raise ValueError(
                f"taxonomy has no path reaching level {config.max_level}; "
                "lower max_level or set flexible: true"
            )
        self.matrix = np.vstack([compose(list(p), emb) for p in self.candidates])

    @staticmethod
    def _build_candidates(paths: list[Path], config: MatchingConfig) -> list[Path]:
        """Fixed depth: only full-depth paths. Flexible: every prefix, 1..max_level."""
        depths = (
            range(1, config.max_level + 1) if config.flexible else (config.max_level,)
        )
        candidates: dict[Path, None] = {}
        for depth in depths:
            for path in paths:
                if len(path) >= depth:
                    candidates.setdefault(path[:depth], None)
        return list(candidates)

    def match(self, vector: np.ndarray, k: int = 1) -> list[MatchResult]:
        """Rank all candidates against `vector`; return the top `k`."""
        sims = self.matrix @ vector
        order = np.argsort(-sims)[:k]
        return [MatchResult(list(self.candidates[i]), float(sims[i])) for i in order]


class TreeBasedMatcher:
    """Beam search down the taxonomy, then a weighed pick among the survivors.

    Each level scores only the children of the paths still in the beam, using the
    same whole-record input vector throughout, and keeps the best `beam_width` of
    them (all of them when there are fewer). A path whose node has no children
    drops out of the beam but stays in the candidate pool -- the branch simply
    ended. What lands in that pool at the end depends on `flexible`:

        flexible=False  only the deepest paths reached (plus branches that ended early)
        flexible=True   every path the beam held at every level

    The winner is then chosen the weighed_embedding way, over that pool.
    """

    def __init__(self, paths: list[Path], emb: dict[str, np.ndarray],
                 config: MatchingConfig):
        self.emb = emb
        self.max_level = config.max_level
        self.flexible = config.flexible
        self.beam_width = config.beam_width
        self.children = self._build_children(paths, config.max_level)
        if not self.children.get((), []):
            raise ValueError("taxonomy has no level-1 categories")

    @staticmethod
    def _build_children(paths: list[Path], max_level: int) -> dict[Path, list[str]]:
        """prefix -> the distinct category names living directly under it."""
        children: dict[Path, dict[str, None]] = {}
        for path in paths:
            for depth in range(min(len(path), max_level)):
                parent = path[:depth]
                children.setdefault(parent, {}).setdefault(path[depth], None)
        return {parent: list(names) for parent, names in children.items()}

    def _descend(self, vector: np.ndarray) -> list[Path]:
        """Run the beam search; return the candidate pool it leaves behind."""
        beam: list[Path] = [()]
        pool: dict[Path, None] = {}

        for _ in range(self.max_level):
            scored: list[tuple[float, Path]] = []
            for prefix in beam:
                names = self.children.get(prefix)
                if not names:
                    if prefix:
                        pool.setdefault(prefix)  # branch ended before max_level
                    continue
                for name in names:
                    scored.append((float(self.emb[name] @ vector), prefix + (name,)))

            if not scored:
                break  # nothing left to expand anywhere in the beam
            scored.sort(key=lambda item: -item[0])
            beam = [path for _, path in scored[: self.beam_width]]
            if self.flexible:
                pool.update(dict.fromkeys(beam))

        pool.update(dict.fromkeys(beam))  # the deepest level reached
        pool.pop((), None)
        return list(pool)

    def match(self, vector: np.ndarray, k: int = 1) -> list[MatchResult]:
        return best_of(self._descend(vector), self.emb, vector, k=k)
