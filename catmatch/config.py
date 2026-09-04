"""Configuration objects.

Two sources merge here: `.env` carries the LLM connection parameters (secrets),
`config.yaml` carries every behavioural parameter. Callers using catmatch as a
module can skip both and construct the dataclasses directly.

Validation happens in `__post_init__` -- an illegal config blows up at
construction time, not halfway through a run.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class LLMConfig:
    """Connection + behaviour of the compression LLM (OpenAI-compatible API)."""

    api_key: str
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-plus"
    temperature: float = 0.0
    concurrency: int = 4
    request_timeout: float = 60.0
    max_retries: int = 3
    max_output_tokens: int = 512

    def __post_init__(self) -> None:
        for name in ("api_key", "base_url", "model"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        for name in ("concurrency", "request_timeout", "max_output_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")

    @classmethod
    def from_env(
        cls,
        *,
        api_key_var: str = "api_key",
        base_url_var: str = "base_url",
        model_var: str = "model",
        **overrides,
    ) -> "LLMConfig":
        """Read the connection parameters from `.env`.

        Only the keys actually present in `.env` override the defaults; `api_key`
        has no default, so a missing one fails in `__post_init__`.
        """
        load_dotenv()
        env = {
            "api_key": os.getenv(api_key_var, ""),
            "base_url": os.getenv(base_url_var) or None,
            "model": os.getenv(model_var) or None,
        }
        kwargs = {k: v for k, v in env.items() if v is not None}
        kwargs.update(overrides)
        return cls(**kwargs)


@dataclass
class CompressionConfig:
    """Shorten over-long input values before embedding them."""

    enabled: bool = True
    threshold_words: int = 200
    llm: LLMConfig | None = None

    def __post_init__(self) -> None:
        if self.threshold_words <= 0:
            raise ValueError("threshold_words must be positive")
        if self.enabled and self.llm is None:
            self.llm = LLMConfig.from_env()


@dataclass
class EmbeddingConfig:
    model_path: str = "model/all-mpnet-base-v2"
    model_name: str = "all-mpnet-base-v2"
    batch_size: int = 12
    db_path: str = "cache/embeddings.sqlite"

    def __post_init__(self) -> None:
        for name in ("model_path", "model_name", "db_path"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not Path(self.model_path).is_dir():
            raise ValueError(
                f"model_path not found: {self.model_path}\n"
                "download the weights once with:\n"
                "  uv run python -c \"from sentence_transformers import SentenceTransformer; "
                "SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"
                ".save('model/all-mpnet-base-v2')\""
            )


@dataclass
class MatchingConfig:
    algo: str = "tree_based"
    max_level: int = 3
    flexible: bool = True

    ALGOS = ("weighed_embedding", "tree_based")

    def __post_init__(self) -> None:
        if self.algo not in self.ALGOS:
            raise ValueError(f"algo must be one of {self.ALGOS}, got {self.algo!r}")
        if self.max_level <= 0:
            raise ValueError("max_level must be positive")


@dataclass
class Config:
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    # only used by the batch entry point; as a module these come in as arguments
    input_path: str | None = None
    taxonomy_path: str | None = None
    output_path: str | None = None

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        """yaml holds the behaviour, `.env` holds the secrets; they merge here."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        comp = dict(raw.get("compression", {}) or {})
        llm_kwargs = comp.pop("llm", {}) or {}
        if comp.get("enabled", True):
            comp["llm"] = LLMConfig.from_env(**llm_kwargs)
        elif llm_kwargs:
            comp["llm"] = None

        return cls(
            matching=MatchingConfig(**(raw.get("matching", {}) or {})),
            embedding=EmbeddingConfig(**(raw.get("embedding", {}) or {})),
            compression=CompressionConfig(**comp),
            input_path=(raw.get("input", {}) or {}).get("path"),
            taxonomy_path=(raw.get("taxonomy", {}) or {}).get("path"),
            output_path=(raw.get("output", {}) or {}).get("path"),
        )
