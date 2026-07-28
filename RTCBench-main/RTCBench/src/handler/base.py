"""Abstract model handler for inference backends."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseHandler(ABC):
    """Unified interface for sending chat messages to a model.

    Subclasses wrap specific backends (vLLM local, OpenAI-compatible API, etc.)
    and return plain text (not structured tool_call) so the prompt strategy
    can parse the output format-agnostically.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def chat(self, messages: list[dict]) -> tuple[str, dict]:
        """Send *messages* to the model.

        Returns
        -------
        content : str
            The model's text response.
        usage : dict
            ``{"input_token": int, "output_token": int}``
        """
        ...

    def batch_chat(self, batch_messages: list[list[dict]]) -> list[tuple[str, dict]]:
        """Send a batch of conversations to the model.

        Default implementation calls chat() sequentially.
        Subclasses (e.g. VLLMHandler) should override for true batching.
        """
        return [self.chat(msgs) for msgs in batch_messages]

    def shutdown(self) -> None:
        """Release resources (override if needed)."""
        pass
