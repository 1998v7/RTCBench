"""OpenAI-compatible API handler (works with OpenAI, vLLM HTTP, OpenRouter, etc.)."""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import BaseHandler


class OpenAIHandler(BaseHandler):
    """Handler for OpenAI-compatible chat completion APIs.

    Parameters
    ----------
    model_name : model identifier (e.g. "gpt-4o-mini", "Qwen/Qwen3-4B").
    base_url : API base URL (default: OpenAI).
    api_key : API key (default: OPENAI_API_KEY env var).
    temperature, max_tokens : generation parameters.
    max_workers : max concurrent API requests for batch_chat.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = 4096,
        max_workers: int = 16,
        **kwargs,
    ):
        super().__init__(model_name, temperature, max_tokens)
        from openai import OpenAI

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
        )
        self._max_workers = max_workers

    def chat(self, messages: list[dict], _retries: int = 3) -> tuple[str, dict]:
        for attempt in range(1, _retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=42,
                )
                if not response.choices:
                    raise ValueError(f"Empty choices in response: {response}")
                choice = response.choices[0]
                content = (choice.message.content or "").strip()
                usage_obj = response.usage
                usage = {
                    "input_token": usage_obj.prompt_tokens if usage_obj else 0,
                    "output_token": usage_obj.completion_tokens if usage_obj else 0,
                }
                return content, usage
            except Exception as e:
                if attempt < _retries:
                    wait = 2 ** attempt
                    print(f"[API] Attempt {attempt} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[API] All {_retries} attempts failed: {e}. Returning empty.")
                    return "", {"input_token": 0, "output_token": 0}

    def batch_chat(self, batch_messages: list[list[dict]]) -> list[tuple[str, dict]]:
        if not batch_messages:
            return []
        results: list[tuple[str, dict] | None] = [None] * len(batch_messages)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            future_to_idx = {
                pool.submit(self.chat, msgs): i
                for i, msgs in enumerate(batch_messages)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return results  # type: ignore[return-value]
