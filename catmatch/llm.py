"""Shared machinery for the two LLM steps (`compact`, `selection`).

Both need the same three things: a lazily-built OpenAI-compatible client,
a retry loop around one chat completion, and a concurrent map with a progress
bar. This is that, factored out so `compact.py` and `selection.py` only have
to worry about prompt-building and reply-parsing.
"""

import concurrent.futures
from typing import Callable, TypeVar

from tqdm import tqdm

from .config import LLMConfig

T = TypeVar("T")
R = TypeVar("R")


class LLMTask:
    """Base for a one-request-per-item LLM step. Subclasses supply `llm`."""

    llm: LLMConfig

    def __init__(self) -> None:
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.llm.api_key,
                base_url=self.llm.base_url,
                timeout=self.llm.request_timeout,
                max_retries=0,  # retries are handled here so failures are visible
            )
        return self._client

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """One chat completion, retried; returns "" once retries are exhausted."""
        for attempt in range(self.llm.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.llm.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.llm.temperature,
                    max_tokens=self.llm.max_output_tokens,
                    stream=False,
                )
                result = (completion.choices[0].message.content or "").strip()
                if result:
                    return result
            except Exception as exc:  # noqa: BLE001 - any API failure is retryable
                print(f"{type(self).__name__} attempt {attempt + 1} failed: {exc}")
        return ""

    def map(self, items: list[T], fn: Callable[[T], R], desc: str) -> list[R]:
        """Run `fn` over `items` concurrently, `llm.concurrency` at a time."""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.llm.concurrency
        ) as executor:
            return list(tqdm(executor.map(fn, items), total=len(items), desc=desc))
