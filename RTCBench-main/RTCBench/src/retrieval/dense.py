"""Transformer-based dense retriever with cosine similarity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import RETRIEVAL_TOP_K, TOOL_BANK_PATH, EMBEDDING_CACHE_DIR
from .base import BaseRetriever
from .indexer import load_tool_bank

DEFAULT_MODEL = "ToolBench/ToolBench_IR_bert_based_uncased"


class DenseRetriever(BaseRetriever):
    """Sentence-embedding retriever using cosine similarity.

    Parameters
    ----------
    tools, doc_texts : pre-built corpus.  If *None*, loaded from tool bank.
    top_k : default number of results.
    model_name : HuggingFace sentence-transformer model id.
    cache_dir : directory for caching pre-computed embeddings.
    """

    def __init__(
        self,
        tools: list[dict] | None = None,
        doc_texts: list[str] | None = None,
        top_k: int | None = None,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Path | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        if tools is None or doc_texts is None:
            tools, doc_texts = load_tool_bank(TOOL_BANK_PATH)

        self._tools = tools
        self._model = SentenceTransformer(model_name)
        self._top_k = top_k or RETRIEVAL_TOP_K

        cache_dir = cache_dir or EMBEDDING_CACHE_DIR
        self._corpus_embeddings = self._load_or_compute_embeddings(
            doc_texts, model_name, cache_dir
        )

    def _load_or_compute_embeddings(
        self, doc_texts: list[str], model_name: str, cache_dir: Path
    ) -> np.ndarray:
        cache_dir.mkdir(parents=True, exist_ok=True)
        content_hash = hashlib.md5(
            json.dumps(doc_texts, sort_keys=True).encode()
        ).hexdigest()[:12]
        model_slug = model_name.replace("/", "_")
        cache_path = cache_dir / f"emb_{model_slug}_{content_hash}.npy"

        if cache_path.exists():
            embeddings = np.load(cache_path)
        else:
            embeddings = self._model.encode(
                doc_texts, show_progress_bar=True, batch_size=64
            )
            np.save(cache_path, embeddings)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-9)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self._top_k
        query_emb = self._model.encode([query], show_progress_bar=False)[0]
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        scores = self._corpus_embeddings @ query_emb
        top_indices = scores.argsort()[::-1][:k]
        return [self._tools[i] for i in top_indices]

    def retrieve_with_scores(
        self, query: str, top_k: int | None = None
    ) -> list[tuple[dict, float]]:
        """Return (tool, score) pairs for rank fusion."""
        k = top_k or self._top_k
        query_emb = self._model.encode([query], show_progress_bar=False)[0]
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        scores = self._corpus_embeddings @ query_emb
        top_indices = scores.argsort()[::-1][:k]
        return [(self._tools[i], float(scores[i])) for i in top_indices]
