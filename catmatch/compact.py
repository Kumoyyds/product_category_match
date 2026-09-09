"""Shorten the input's last (free-text) level with an LLM before embedding it.

Only the last level is compacted -- upper levels are short by construction and
the last level is where concept-test descriptions carry marketing copy the
matcher shouldn't see. The prompt (prompts/compact_prompt.txt by default) does
the actual "what is this, strip the rest" extraction; this module just drives
one request per over-long value and parses the numbered-list reply it asks for.

Results are not cached: every run compacts afresh.
"""

from pathlib import Path

from .config import CompactConfig
from .llm import LLMTask

SYSTEM_PROMPT = (
    "You extract a neutral, factual product definition from concept-test copy. "
    "Follow the instructions exactly and never invent details."
)

UNCLEAR = "UNCLEAR"


def word_count(text: str) -> int:
    return len(text.split())


def _strip_numbering(line: str) -> str:
    """Drop a leading "1." / "1)" list marker, if the reply included one."""
    stripped = line.strip()
    head, sep, rest = stripped.partition(".")
    if not sep:
        head, sep, rest = stripped.partition(")")
    if sep and head.strip().isdigit():
        return rest.strip()
    return stripped


class Compactor:
    def __init__(self, config: CompactConfig):
        self.config = config
        self._task: LLMTask | None = None
        self._prompt_template: str | None = None

    @property
    def task(self) -> LLMTask:
        if self._task is None:
            task = LLMTask()
            task.llm = self.config.llm
            self._task = task
        return self._task

    @property
    def prompt_template(self) -> str:
        if self._prompt_template is None:
            self._prompt_template = Path(self.config.prompt_path).read_text(
                encoding="utf-8"
            )
        return self._prompt_template

    def needs_compact(self, text: str) -> bool:
        return self.config.enabled and word_count(text) > self.config.threshold_words

    def compact(self, text: str) -> str:
        """Compact one string; falls back to the original on UNCLEAR/failure."""
        user_prompt = self.prompt_template.format(f"1. {text}")
        reply = self.task.complete(SYSTEM_PROMPT, user_prompt)
        if not reply:
            print("compact: LLM call failed, keeping the original text")
            return text

        first_line = next((ln for ln in reply.splitlines() if ln.strip()), "")
        result = _strip_numbering(first_line)
        if not result or result.strip().upper() == UNCLEAR:
            print("compact: model returned UNCLEAR, keeping the original text")
            return text
        return result

    def compact_all(self, texts: list[str]) -> dict[str, str]:
        """Map every over-long string to its compacted form (short ones skipped)."""
        if not self.config.enabled:
            return {}

        targets = [t for t in dict.fromkeys(texts) if self.needs_compact(t)]
        if not targets:
            return {}

        print(f"compacting {len(targets)} over-long value(s) ...")
        compacted = self.task.map(targets, self.compact, desc="compact")
        return dict(zip(targets, compacted))
