"""Sentence-transformer wrapper.

The model is loaded lazily on first use and its path comes from the config, so
importing catmatch from any directory is safe (the old `tools.match_func` loaded
the model at import time off `os.getcwd()`).

Vectors come back L2-normalised, which makes a dot product a cosine similarity.
"""

import numpy as np

from .config import EmbeddingConfig


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Scale to unit length; an all-zero vector is returned untouched."""
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


class Embedder:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"loading embedding model from {self.config.model_path} ...")
            self._model = SentenceTransformer(self.config.model_path)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1000,
        )

    def encode_dict(self, texts: list[str]) -> dict[str, np.ndarray]:
        unique = list(dict.fromkeys(texts))
        vectors = self.encode(unique)
        return dict(zip(unique, vectors))
