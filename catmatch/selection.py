"""Let an LLM make the final pick among the matcher's top-k candidates.

The matcher already ranks candidates by cosine similarity; when the top few
are close, similarity alone often picks the wrong one. This module hands the
product's full level chain and the numbered candidate paths to an LLM and asks
it for one number. A reply that fails to parse into a valid index -- after
retries -- falls back to the matcher's own top-1.
"""

from pathlib import Path

from .config import SelectionConfig
from .llm import LLMTask
from .matchers import MatchResult

SYSTEM_PROMPT = (
    "You classify products into a category taxonomy. Follow the instructions "
    "exactly and answer with a single number, nothing else."
)


def _first_int(text: str) -> int | None:
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else None


class Selector:
    def __init__(self, config: SelectionConfig):
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

    def select(self, product_text: str, candidates: list[MatchResult]) -> int:
        """Index (0-based) of the chosen candidate; falls back to 0 (top-1)."""
        numbered = "\n".join(
            f"{i + 1}. {' > '.join(c.path)}" for i, c in enumerate(candidates)
        )
        user_prompt = self.prompt_template.format(
            product=product_text, candidates=numbered
        )

        for attempt in range(self.config.llm.max_retries + 1):
            reply = self.task.complete(SYSTEM_PROMPT, user_prompt)
            choice = _first_int(reply)
            if choice is not None and 1 <= choice <= len(candidates):
                return choice - 1
            print(f"selection attempt {attempt + 1} gave an unusable reply: {reply!r}")

        print("selection failed, falling back to top-1")
        return 0

    def select_all(
        self, jobs: list[tuple[str, list[MatchResult]]]
    ) -> list[int]:
        """Concurrent `select` over `(product_text, candidates)` jobs."""
        if not jobs:
            return []
        print(f"selecting among top-k for {len(jobs)} unique record(s) ...")
        return self.task.map(jobs, lambda job: self.select(*job), desc="selection")
