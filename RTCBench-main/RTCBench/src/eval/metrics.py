"""Retrieval quality metrics: Recall@K, MRR, etc."""
from __future__ import annotations


def _normalize_name(name: str) -> str:
    return name.replace(".", "_")


def recall_at_k(
    gt_tool_names: list[str],
    search_tools_results: list[dict],
) -> float | None:
    """Fraction of GT tool names found in all retrieved results.

    Returns None if there are no search results or no GT names.
    """
    if not search_tools_results or not gt_tool_names:
        return None

    retrieved_names: set[str] = set()
    for item in search_tools_results:
        for t in item.get("retrieved", []):
            n = t.get("name", "")
            if n:
                retrieved_names.add(_normalize_name(n))

    gt_normalized = {_normalize_name(n) for n in gt_tool_names}
    if not gt_normalized:
        return None

    found = gt_normalized & retrieved_names
    return len(found) / len(gt_normalized)


def mrr(
    gt_tool_names: list[str],
    search_tools_results: list[dict],
) -> float | None:
    """Mean Reciprocal Rank of the first GT tool found in retrieved results.

    Considers all search_tool calls; for each GT name, finds the best
    (lowest) rank across all retrievals.
    Returns None if no search results or no GT names.
    """
    if not search_tools_results or not gt_tool_names:
        return None

    gt_normalized = {_normalize_name(n) for n in gt_tool_names}

    best_ranks: dict[str, int] = {}
    for item in search_tools_results:
        retrieved = item.get("retrieved", [])
        for rank, t in enumerate(retrieved, start=1):
            n = _normalize_name(t.get("name", ""))
            if n in gt_normalized:
                if n not in best_ranks or rank < best_ranks[n]:
                    best_ranks[n] = rank

    if not best_ranks:
        return 0.0

    reciprocal_ranks = [1.0 / r for r in best_ranks.values()]
    return sum(reciprocal_ranks) / len(gt_normalized)


def gt_all_retrieved(
    gt_tool_names: list[str],
    search_tools_results: list[dict],
) -> bool | None:
    """Whether ALL GT tools are present in the retrieved results."""
    r = recall_at_k(gt_tool_names, search_tools_results)
    if r is None:
        return None
    return r == 1.0
