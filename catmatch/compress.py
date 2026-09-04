"""Shorten over-long input values with an LLM before embedding them.

Only the input side is compressed -- taxonomy levels are short by construction.
The one hard rule for the model: do not change the essential info (what the
thing *is*), only drop redundant wording.

Results are not cached: every run compresses afresh.
"""

import concurrent.futures

from tqdm import tqdm

from .config import CompressionConfig

SYSTEM_PROMPT = (
    "You compress product descriptions for an online retail catalogue. "
    "Keep the essential information -- what the product actually is -- and drop "
    "redundant wording, marketing language and repetition. Do not add anything, "
    "do not change what the product is, and never invent details."
)

USER_PROMPT = (
    "Compress the following product text to at most {limit} words. "
    "Output only the compressed text.\n\n{text}"
)


def word_count(text: str) -> int:
    return len(text.split())


class Compressor:
    def __init__(self, config: CompressionConfig):
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            llm = self.config.llm
            self._client = OpenAI(
                api_key=llm.api_key,
                base_url=llm.base_url,
                timeout=llm.request_timeout,
                max_retries=0,  # retries are handled here so failures are visible
            )
        return self._client

    def needs_compression(self, text: str) -> bool:
        return self.config.enabled and word_count(text) > self.config.threshold_words

    def compress(self, text: str) -> str:
        """Compress one string; falls back to a hard truncation if the LLM fails."""
        llm = self.config.llm
        limit = self.config.threshold_words

        for attempt in range(llm.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=llm.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT.format(limit=limit, text=text)},
                    ],
                    temperature=llm.temperature,
                    max_tokens=llm.max_output_tokens,
                    stream=False,
                )
                result = (completion.choices[0].message.content or "").strip()
                if result:
                    return result
            except Exception as exc:  # noqa: BLE001 - any API failure is retryable
                print(f"compression attempt {attempt + 1} failed: {exc}")

        print("compression failed, truncating instead")
        return " ".join(text.split()[:limit])

    def compress_all(self, texts: list[str]) -> dict[str, str]:
        """Map every over-long string to its compressed form (short ones are skipped)."""
        if not self.config.enabled:
            return {}

        targets = [t for t in dict.fromkeys(texts) if self.needs_compression(t)]
        if not targets:
            return {}

        print(f"compressing {len(targets)} over-long value(s) ...")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.llm.concurrency
        ) as executor:
            compressed = list(
                tqdm(executor.map(self.compress, targets), total=len(targets))
            )
        return dict(zip(targets, compressed))
