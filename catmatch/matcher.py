"""The one public entry point: `Matcher.match(records, taxonomy)`.

It wires everything together -- contract checks, compression, embedding (cached
on the taxonomy side only), then whichever algorithm the config asks for.
"""

import numpy as np
from tqdm import tqdm

from . import io
from .compress import Compressor
from .config import Config
from .embedding import Embedder
from .matchers import MatchResult, TreeBasedMatcher, WeighedEmbeddingMatcher, compose
from .store import EmbeddingStore

ERROR = "error"


class Matcher:
    def __init__(self, config: Config):
        self.config = config
        self.embedder = Embedder(config.embedding)
        self.compressor = Compressor(config.compression)

    # ---------------------------------------------------------------- helpers

    def _taxonomy_paths(self, taxonomy: list[io.Record],
                        cat_cols: list[str]) -> list[tuple[str, ...]]:
        """Each row becomes a path, cut at its first empty level and at max_level."""
        max_level = self.config.matching.max_level
        if max_level > len(cat_cols):
            raise ValueError(
                f"max_level ({max_level}) exceeds the taxonomy depth "
                f"({len(cat_cols)} cat_* columns)"
            )

        paths: dict[tuple[str, ...], None] = {}
        for row in taxonomy:
            path = []
            for col in cat_cols[:max_level]:
                value = io.clean(row.get(col))
                if value is None:
                    break
                path.append(value)
            if path:
                paths.setdefault(tuple(path), None)
        if not paths:
            raise ValueError("taxonomy contains no usable cat_* values")
        return list(paths)

    @staticmethod
    def _record_levels(record: io.Record, level_cols: list[str]) -> list[str]:
        """The record's level values, cleaned; empties are dropped."""
        return [v for v in (io.clean(record.get(c)) for c in level_cols) if v is not None]

    def _taxonomy_embeddings(self, paths: list[tuple[str, ...]]) -> dict[str, np.ndarray]:
        """Look the elements up in the store, embed whatever is missing, save it."""
        elements = list(dict.fromkeys(name for path in paths for name in path))
        with EmbeddingStore(
            self.config.embedding.db_path, self.config.embedding.model_name
        ) as store:
            known = store.get_many(elements)
            missing = [e for e in elements if e not in known]
            if missing:
                print(
                    f"taxonomy: {len(known)}/{len(elements)} elements already in the db, "
                    f"embedding {len(missing)} new one(s) ..."
                )
                fresh = self.embedder.encode_dict(missing)
                store.put_many(fresh)
                known.update(fresh)
            else:
                print(f"taxonomy: all {len(elements)} elements already in the db")
        return known

    def _input_embeddings(self, rows: list[list[str]]) -> dict[str, np.ndarray]:
        """Compress the over-long values, then embed everything (never cached)."""
        texts = list(dict.fromkeys(text for row in rows for text in row))
        compressed = self.compressor.compress_all(texts)
        for row in rows:
            for i, text in enumerate(row):
                if text in compressed:
                    row[i] = compressed[text]

        texts = list(dict.fromkeys(text for row in rows for text in row))
        print(f"input: embedding {len(texts)} unique value(s) ...")
        return self.embedder.encode_dict(texts)

    # ------------------------------------------------------------------- main

    def match(self, records: list[io.Record],
              taxonomy: list[io.Record]) -> list[io.Record]:
        """Match every record against the taxonomy; one result row per input row."""
        level_cols = io.level_columns(records, "level")
        cat_cols = io.level_columns(taxonomy, "cat")

        paths = self._taxonomy_paths(taxonomy, cat_cols)
        emb = self._taxonomy_embeddings(paths)

        # the embedding text can differ from the original (compression), so keep both
        originals = [self._record_levels(r, level_cols) for r in records]
        to_embed = [list(row) for row in originals]
        emb.update(self._input_embeddings(to_embed))

        algo = self.config.matching.algo
        matcher_cls = (
            TreeBasedMatcher if algo == "tree_based" else WeighedEmbeddingMatcher
        )
        matcher = matcher_cls(
            paths, emb, self.config.matching.max_level, self.config.matching.flexible
        )

        print(f"matching {len(records)} record(s) with {algo} ...")
        cache: dict[tuple[str, ...], MatchResult] = {}
        results: list[MatchResult | None] = []
        for row in tqdm(to_embed):
            key = tuple(row)
            if key not in cache:
                cache[key] = matcher.match(compose(row, emb)) if row else None
            results.append(cache[key])

        return self._to_records(records, results)

    @staticmethod
    def _to_records(records: list[io.Record],
                    results: list[MatchResult | None]) -> list[io.Record]:
        """Input columns, then cat_1..cat_k, match_level and sim."""
        depth = max((r.level for r in results if r), default=0)
        out = []
        for record, result in zip(records, results):
            row = dict(record)
            if result is None:
                row.update({f"cat_{i + 1}": ERROR for i in range(depth)})
                row["match_level"] = ERROR
                row["sim"] = ERROR
            else:
                for i in range(depth):
                    row[f"cat_{i + 1}"] = result.path[i] if i < result.level else None
                row["match_level"] = result.level
                row["sim"] = result.sim
            out.append(row)
        return out
