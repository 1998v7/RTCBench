"""BM25 (Okapi) keyword retriever with automatic CJK tokenization."""
from __future__ import annotations

import re

from ..config import RETRIEVAL_TOP_K, TOOL_BANK_PATH
from .base import BaseRetriever
from .indexer import load_tool_bank

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _tokenize(text: str, use_jieba: bool) -> list[str]:
    """Tokenize text: jieba for CJK content, whitespace otherwise."""
    lowered = text.lower()
    if use_jieba:
        import jieba
        return [w for w in jieba.cut(lowered) if w.strip()]
    return lowered.split()


class BM25Retriever(BaseRetriever):
    """BM25Okapi keyword retriever over tool doc texts.

    Automatically detects CJK characters in the corpus and uses jieba
    for tokenization when present. Falls back to whitespace splitting
    for pure ASCII/Latin content.

    Parameters
    ----------
    tools, doc_texts : pre-built corpus.  If *None*, loaded from tool bank.
    top_k : default number of results.
    """

    def __init__(
        self,
        tools: list[dict] | None = None,
        doc_texts: list[str] | None = None,
        top_k: int | None = None,
    ):
        from rank_bm25 import BM25Okapi

        if tools is None or doc_texts is None:
            tools, doc_texts = load_tool_bank(TOOL_BANK_PATH)

        self._tools = tools
        self._top_k = top_k or RETRIEVAL_TOP_K
        self._use_jieba = any(_has_cjk(doc) for doc in doc_texts[:50])

        if self._use_jieba:
            import jieba
            jieba.setLogLevel(jieba.logging.WARNING)

        tokenized = [_tokenize(doc, self._use_jieba) for doc in doc_texts]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self._top_k
        q_tokens = _tokenize(query, self._use_jieba)
        scores = self._bm25.get_scores(q_tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [self._tools[i] for i in top_indices]

    def retrieve_with_scores(
        self, query: str, top_k: int | None = None
    ) -> list[tuple[dict, float]]:
        """Return (tool, score) pairs for rank fusion."""
        k = top_k or self._top_k
        q_tokens = _tokenize(query, self._use_jieba)
        scores = self._bm25.get_scores(q_tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [(self._tools[i], float(scores[i])) for i in top_indices]
