"""SQLite store for taxonomy embeddings.

Only the taxonomy side is cached: a taxonomy is stable and gets re-used run
after run, while input records differ every time. Vectors are keyed by
`(text, model_name)` so switching embedding models never mixes old vectors in.
"""

import sqlite3
from pathlib import Path

import numpy as np

# sqlite caps the number of bound variables per statement (999 on older builds)
_CHUNK = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    text        TEXT NOT NULL,
    model_name  TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    PRIMARY KEY (text, model_name)
);
"""


class EmbeddingStore:
    def __init__(self, db_path: str | Path, model_name: str):
        self.db_path = Path(db_path)
        self.model_name = model_name
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(_SCHEMA)
        self.conn.commit()

    def get_many(self, texts: list[str]) -> dict[str, np.ndarray]:
        """Return the vectors already in the store; missing texts are absent."""
        found: dict[str, np.ndarray] = {}
        unique = list(dict.fromkeys(texts))
        for i in range(0, len(unique), _CHUNK):
            chunk = unique[i : i + _CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT text, dim, vector FROM embeddings "
                f"WHERE model_name = ? AND text IN ({placeholders})",
                [self.model_name, *chunk],
            ).fetchall()
            for text, dim, blob in rows:
                found[text] = np.frombuffer(blob, dtype=np.float32, count=dim)
        return found

    def missing(self, texts: list[str]) -> list[str]:
        """The texts that still need embedding."""
        known = self.get_many(texts)
        return [t for t in dict.fromkeys(texts) if t not in known]

    def put_many(self, vectors: dict[str, np.ndarray]) -> None:
        rows = [
            (text, self.model_name, len(vec), np.asarray(vec, dtype=np.float32).tobytes())
            for text, vec in vectors.items()
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO embeddings (text, model_name, dim, vector) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "EmbeddingStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
