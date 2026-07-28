"""Hybrid retriever using Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

from ..config import RETRIEVAL_TOP_K
from .base import BaseRetriever
from .bm25 import BM25Retriever
from .dense import DenseRetriever


class HybridRetriever(BaseRetriever):
    """Combine BM25 and Dense retrieval via Reciprocal Rank Fusion.

    RRF score for each tool = sum over retrievers of 1 / (k + rank_i),
    where k is a smoothing constant (default 60, standard in literature).

    Parameters
    ----------
    bm25 : BM25Retriever instance.
    dense : DenseRetriever instance.
    top_k : final number of results to return.
    rrf_k : RRF smoothing constant.
    retrieval_depth : how many candidates to fetch from each retriever
        before fusing (should be >= top_k).
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        top_k: int | None = None,
        rrf_k: int = 60,
        retrieval_depth: int | None = None,
    ):
        self._bm25 = bm25
        self._dense = dense
        self._top_k = top_k or RETRIEVAL_TOP_K
        self._rrf_k = rrf_k
        self._depth = retrieval_depth or self._top_k * 3

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self._top_k
        depth = max(self._depth, k * 3)

        bm25_results = self._bm25.retrieve(query, top_k=depth)
        dense_results = self._dense.retrieve(query, top_k=depth)

        rrf_scores: dict[str, float] = {}
        tool_by_name: dict[str, dict] = {}

        for rank, tool in enumerate(bm25_results):
            name = tool["name"]
            rrf_scores[name] = rrf_scores.get(name, 0) + 1.0 / (self._rrf_k + rank + 1)
            tool_by_name[name] = tool

        for rank, tool in enumerate(dense_results):
            name = tool["name"]
            rrf_scores[name] = rrf_scores.get(name, 0) + 1.0 / (self._rrf_k + rank + 1)
            tool_by_name[name] = tool

        sorted_names = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
        return [tool_by_name[n] for n in sorted_names]
