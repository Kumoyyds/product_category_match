"""catmatch -- match product records against a category taxonomy.

    from catmatch import Matcher, Config, MatchingConfig

    matcher = Matcher(Config(matching=MatchingConfig(algo="tree_based", max_level=5)))
    results = matcher.match(records, taxonomy)

English only: there is no translation step, input must already be English.
"""

from .config import (
    CompressionConfig,
    Config,
    EmbeddingConfig,
    LLMConfig,
    MatchingConfig,
)
from .io import load_input, load_taxonomy, read_records, write_records
from .matcher import Matcher
from .matchers import MatchResult

__all__ = [
    "Matcher",
    "MatchResult",
    "Config",
    "MatchingConfig",
    "EmbeddingConfig",
    "CompressionConfig",
    "LLMConfig",
    "load_input",
    "load_taxonomy",
    "read_records",
    "write_records",
]
