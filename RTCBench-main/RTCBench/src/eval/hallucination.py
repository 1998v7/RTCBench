"""Hallucination detection (per RTCBench paper Section 3.5).

Two dimensions, computed per-sample as boolean flags so that the aggregate
rate is the **union** rather than the sum:

  1) Skip-search    -- model called a non-search tool without first invoking
                       search_tool, even though no ground-truth tool was
                       visible in the initial context.  Only applies to the
                       "none" / "distractor" variants; under "with_gt" a
                       direct call is correct behaviour.
  2) Phantom Tool   -- model called a tool whose name appears in NEITHER the
                       initial visible tools NOR any retrieval result.

The aggregate `hallucination` rate is the fraction of samples for which at
least one of the two flags fired.

Helpers in this module operate on the records produced by the orchestrator
(`Inference_retrieval_*.json`) plus the corresponding test entries (which
carry the `function` / `distractor_tools` lists used to build the initial
visible tool set).
"""
from __future__ import annotations

from typing import Iterable

from ..prompt.default import DefaultPromptStrategy

# Reuse the same strict ast-based parser the orchestrator uses, so that
# math/latex fragments like ``4(1)`` or ``f(x)`` in the model's reasoning
# never get mistaken for tool calls.
_PARSER = DefaultPromptStrategy()


def parse_raw_call_names(text: str) -> list[str]:
    """Extract non-search tool names from a raw model output string.

    Uses the prompt strategy's ``parse_tool_calls`` (ast.parse on the
    ``[func(...)]`` block) so non-Python text is rejected as parse errors.
    Returns names in order of first appearance, with ``search_tool`` removed.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    calls = _PARSER.parse_tool_calls(text)
    seen: list[str] = []
    seen_set: set[str] = set()
    for c in calls:
        if not isinstance(c, dict):
            continue
        for name in c.keys():
            if name == "search_tool" or name in seen_set:
                continue
            seen.append(name)
            seen_set.add(name)
    return seen


def initial_tool_names(test_entry: dict, variant: str) -> list[str]:
    """Tools visible in the initial system prompt for a given variant."""
    if variant == "with_gt":
        tools = test_entry.get("function") or []
    elif variant == "distractor":
        tools = test_entry.get("distractor_tools") or []
    else:  # "none" or unknown -> empty
        tools = []
    names: list[str] = []
    seen: set[str] = set()
    for t in tools:
        if not isinstance(t, dict):
            continue
        n = t.get("name")
        if n and n not in seen:
            names.append(n)
            seen.add(n)
    return names


def collect_called_names(
    decoded: list,
    raw_outputs: Iterable[str] | None = None,
) -> list[str]:
    """Union of tool names called by the model.

    Combines both:
      * decoded (parsed) tool calls from `model_responses_decoded`
      * raw regex parse of every assistant turn in `raw_model_outputs`
        (so we still detect direct calls when the parser failed)
    """
    seen: list[str] = []
    seen_set: set[str] = set()

    def _add(name: str) -> None:
        if name and name != "search_tool" and name not in seen_set:
            seen.append(name)
            seen_set.add(name)

    for call in decoded or []:
        if isinstance(call, dict):
            for k in call.keys():
                _add(k)

    for raw in (raw_outputs or []):
        for n in parse_raw_call_names(raw):
            _add(n)

    return seen


def collect_retrieved_names(search_tools_results: list[dict]) -> list[str]:
    """Flat list of every tool name returned by any search_tool call."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for sr in search_tools_results or []:
        for tool in sr.get("retrieved", []) or []:
            if isinstance(tool, dict):
                n = tool.get("name", "")
                if n and n not in seen_set:
                    seen.append(n)
                    seen_set.add(n)
    return seen


def per_sample_flags(
    *,
    variant: str,
    search_count: int,
    initial_names: list[str],
    retrieved_names: list[str],
    called_names: list[str],
) -> dict:
    """Return per-sample hallucination flags following the paper definition.

    Returns a dict with keys:
      - is_skip_search
      - is_phantom_tool
      - phantom_tool_names    (list[str], possibly empty)
      - is_hallucination      (skip_search OR phantom_tool)
    """
    skip_search = (
        variant != "with_gt"
        and search_count == 0
        and len(called_names) > 0
    )

    available = set(initial_names) | set(retrieved_names)
    phantom_names = [n for n in called_names if n not in available]
    phantom = len(phantom_names) > 0

    return {
        "is_skip_search": bool(skip_search),
        "is_phantom_tool": bool(phantom),
        "phantom_tool_names": phantom_names,
        "is_hallucination": bool(skip_search or phantom),
    }


def aggregate(scores: list[dict]) -> dict:
    """Aggregate per-sample hallucination flags over a list of score dicts.

    `scores` entries are expected to carry the per-sample fields written by
    `per_sample_flags` (or a superset thereof). Missing fields default to
    False/empty so this is safe to call on legacy records too.
    """
    total = len(scores)
    skip_search = sum(1 for s in scores if s.get("is_skip_search"))
    phantom = sum(1 for s in scores if s.get("is_phantom_tool"))
    halluc = sum(1 for s in scores if s.get("is_hallucination"))
    phantom_tool_total = sum(len(s.get("phantom_tool_names") or []) for s in scores)

    return {
        "skip_search": skip_search,
        "phantom_tool_entries": phantom,
        "phantom_tool_total": phantom_tool_total,
        "hallucination_entries": halluc,
        "skip_search_rate": skip_search / total if total else 0.0,
        "phantom_tool_rate": phantom / total if total else 0.0,
        "hallucination_rate": halluc / total if total else 0.0,
        "hallucination_permille": (halluc / total * 1000.0) if total else 0.0,
    }
