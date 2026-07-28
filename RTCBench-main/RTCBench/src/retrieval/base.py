"""Abstract retriever interface for tool search."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """Abstract base for tool retrieval backends."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return the top-k tool definitions most relevant to *query*."""
        ...

    def __call__(self, query: str, top_k: int | None = None) -> list[dict]:
        return self.retrieve(query, top_k)
