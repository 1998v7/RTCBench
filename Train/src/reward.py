"""Reward function for VERL reward pipeline.

compute_score signature must match:
  def compute_score(data_source, solution_str, ground_truth, extra_info) -> dict

Uses fine-grained tool call matching with partial credit:
  - Order-independent function name matching (frequency-based Jaccard)
  - Per-parameter value matching with type coercion
  - Partial credit for partially correct tool calls
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter
from typing import Any


# ── Text preprocessing ────────────────────────────────────────────────────────

def _strip_think(content: str) -> str:
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def _strip_tool_response(content: str) -> str:
    """Remove <tool_response>...</tool_response> sections.

    Tool response JSON may contain unmatched brackets inside string values
    (e.g. "range ['from', 'to')") which breaks naive bracket-depth counting.
    """
    return re.sub(r"<tool_response>.*?</tool_response>", "", content, flags=re.DOTALL)


def _has_valid_format(content: str) -> bool:
    if not content:
        return False
    op = content.find("<think>")
    cl = content.find("</think>")
    return 0 <= op < cl


def _extract_bracket_groups(content: str) -> list[str]:
    """Extract all top-level balanced bracket groups from response text."""
    stripped = _strip_think(content)
    stripped = _strip_tool_response(stripped)
    groups: list[str] = []
    i = 0
    while i < len(stripped):
        if stripped[i] == "[":
            depth = 0
            for j in range(i, len(stripped)):
                if stripped[j] == "[":
                    depth += 1
                elif stripped[j] == "]":
                    depth -= 1
                    if depth == 0:
                        groups.append(stripped[i : j + 1])
                        i = j + 1
                        break
            else:
                break
        else:
            i += 1
    return groups


def _count_valid_search_calls(content: str) -> int:
    """Count AST-parseable search_tool calls with non-empty query strings.

    Aligned with benchmark's parse_search_calls: requires valid AST syntax,
    function name == 'search_tool', and a non-empty string 'query' parameter.
    """
    count = 0
    for group in _extract_bracket_groups(content):
        if "search_tool(" not in group:
            continue
        parsed = _parse_tool_calls(group)
        if not parsed:
            continue
        for call in parsed:
            if len(call) != 1 or "search_tool" not in call:
                continue
            q = call["search_tool"].get("query")
            if isinstance(q, str) and q.strip():
                count += 1
    return count


def _extract_retrieved_tool_names(content: str) -> set[str]:
    """Extract retrieved tool names from <tool_response> sections.

    Uses regex instead of JSON parsing because tool responses are often
    truncated by MAX_SEARCH_RESULT_LEN, making the JSON incomplete.
    """
    names: set[str] = set()
    for m in re.finditer(r"<tool_response>(.*?)</tool_response>", content, flags=re.DOTALL):
        body = m.group(1)
        for name_match in re.finditer(r'"name"\s*:\s*"([^"]+)"', body):
            names.add(name_match.group(1))
    return names


def _extract_tool_calls_string(content: str) -> str | None:
    """Extract the last non-search bracket group from the response.

    In multi-turn responses the first bracket group is often [search_tool(...)],
    so we iterate through ALL bracket groups and keep the last one that contains
    a function call but is NOT a search_tool call.

    Tool response sections are stripped first to avoid bracket confusion from
    JSON content inside tool descriptions.
    """
    stripped = _strip_think(content)
    stripped = _strip_tool_response(stripped)
    last_match = None
    i = 0
    while i < len(stripped):
        if stripped[i] == "[":
            depth = 0
            for j in range(i, len(stripped)):
                if stripped[j] == "[":
                    depth += 1
                elif stripped[j] == "]":
                    depth -= 1
                    if depth == 0:
                        raw = stripped[i : j + 1]
                        if "(" in raw and "search_tool(" not in raw:
                            last_match = raw
                        i = j + 1
                        break
            else:
                break
        else:
            i += 1
    return last_match


# ── AST-based tool call parsing ───────────────────────────────────────────────

def _parse_tool_calls(s: str) -> list[dict[str, dict]] | None:
    """Parse '[func(a=1, b=2), func2(c=3)]' into [{'func': {'a': 1, 'b': 2}}, ...].

    Returns None on parse failure.
    """
    s = s.strip()
    if not s:
        return None

    normalized = s.replace("true", "True").replace("false", "False").replace("null", "None")

    try:
        tree = ast.parse(normalized, mode="eval")
    except Exception:
        return None

    expr = tree.body

    if isinstance(expr, ast.List):
        calls = expr.elts
    elif isinstance(expr, ast.Call):
        calls = [expr]
    else:
        return None

    results = []
    for node in calls:
        parsed = _parse_single_call(node)
        if parsed is None:
            return None
        results.append(parsed)
    return results


def _parse_single_call(node: ast.AST) -> dict[str, dict] | None:
    if not isinstance(node, ast.Call):
        return None

    func_name = _get_func_name(node.func)
    if func_name is None:
        return None

    params = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue
        params[kw.arg] = _ast_to_value(kw.value)

    return {func_name: params}


def _get_func_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        parent = _get_func_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
    return None


def _ast_to_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.List):
        return [_ast_to_value(e) for e in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(_ast_to_value(e) for e in node.elts)
    elif isinstance(node, ast.Dict):
        return {
            _ast_to_value(k): _ast_to_value(v)
            for k, v in zip(node.keys, node.values)
        }
    elif isinstance(node, ast.Set):
        return {_ast_to_value(e) for e in node.elts}
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = _ast_to_value(node.operand)
        if isinstance(val, (int, float)):
            return -val
    elif isinstance(node, ast.Name):
        if node.id == "True":
            return True
        elif node.id == "False":
            return False
        elif node.id == "None":
            return None
        return node.id
    return str(ast.dump(node))


# ── Value comparison ──────────────────────────────────────────────────────────

def _normalize_value(v: Any) -> Any:
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, list):
        return [_normalize_value(e) for e in v]
    if isinstance(v, dict):
        return {_normalize_value(k): _normalize_value(val) for k, val in v.items()}
    return v


def _values_match(pred_val: Any, gt_val: Any) -> bool:
    pn = _normalize_value(pred_val)
    gn = _normalize_value(gt_val)

    if pn == gn:
        return True

    try:
        if float(pred_val) == float(gt_val):
            return True
    except (ValueError, TypeError):
        pass

    try:
        ps = str(pred_val).strip().strip("'\"").lower()
        gs = str(gt_val).strip().strip("'\"").lower()
        if ps == gs:
            return True
    except (ValueError, TypeError):
        pass

    return False


# ── Fine-grained scoring (adapted from ToolRL paper) ─────────────────────────

def _match_score(list1: list, list2: list) -> float:
    """Frequency-aware Jaccard similarity, order-independent."""
    if list1 == list2:
        return 1.0
    if not list1 or not list2:
        return 0.0
    count1 = Counter(list1)
    count2 = Counter(list2)
    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    union = len(list1) + len(list2) - intersection
    return intersection / union if union > 0 else 0.0


OUTCOME_MAX = 3.0
OUTCOME_MIN = -3.0


def _compute_tool_call_score(
    pred_calls: list[dict],
    gt_calls: list[dict],
    max_reward: float = OUTCOME_MAX,
    min_reward: float = OUTCOME_MIN,
) -> float:
    """Fine-grained scoring for tool calls with partial credit.

    Scoring breakdown (all accumulated into a raw ``score``):
      1. Name matching (order-independent, frequency-based Jaccard)  → up to 1.0
      2. Per GT tool: find best-matching pred tool by name, then
         a. parameter-key Jaccard                                    → up to 1.0
         b. per-param value correctness                              → up to len(gt_params)

    local_max_possible = 1.0 + Σ (1.0 + len(gt_params_i))
    outcome = (max - min) * raw_score / local_max_possible + min   ∈ [min, max]
    """
    if not pred_calls or not gt_calls:
        return min_reward

    pred_names = [list(c.keys())[0].lower() for c in pred_calls]
    gt_names = [list(c.keys())[0].lower() for c in gt_calls]
    score = _match_score(pred_names, gt_names)

    local_max_possible = 1.0
    used_pred_indices: set[int] = set()

    for gt_call in gt_calls:
        gt_name = list(gt_call.keys())[0]
        gt_params = gt_call[gt_name]
        local_max_possible += 1.0 + len(gt_params)

        best_match_score = 0.0
        best_match_index = -1

        for i, pred_call in enumerate(pred_calls):
            if i in used_pred_indices:
                continue
            pred_name = list(pred_call.keys())[0]
            if pred_name.lower() != gt_name.lower():
                continue

            pred_params = pred_call[pred_name]
            param_keys_score = _match_score(
                sorted(gt_params.keys()), sorted(pred_params.keys())
            )
            correctness_score = sum(
                1.0
                for k, v in gt_params.items()
                if k in pred_params and _values_match(pred_params[k], v)
            )
            total = param_keys_score + correctness_score

            if total > best_match_score:
                best_match_score = total
                best_match_index = i

        if best_match_index >= 0:
            used_pred_indices.add(best_match_index)
            score += best_match_score

    return (max_reward - min_reward) * score / local_max_possible + min_reward


# ── Main reward function ──────────────────────────────────────────────────────

FORMAT_REWARD = 1.0
SEARCH_BASE = 1.0
RECALL_REWARD = 2.0
WITHGT_REWARD = 3.0
COUNT_PENALTY = 0.1
HONEST_REJECT_REWARD = 0.5


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """3-component reward: format + outcome + search_behavior.

    Ranges:
      format:   0 or FORMAT_REWARD (1.0)
      outcome:  [OUTCOME_MIN, OUTCOME_MAX]  ([-3, 3])
      search:
        with_gt:  not searched → +3.0,  searched → -2.0
        none:     not searched → -2.0
                  searched → 1.0 + recall*2.0 - 0.1*exceeded(count+1)
      honest reject (none, searched, recall=0, no func call) → 0.5
      total:    [-3, 7.0]
    """
    extra_info = extra_info or {}
    ik = extra_info.get("interaction_kwargs", {})
    variant = ik.get("variant", "none")
    trajectory_search_count = ik.get("trajectory_search_count")
    if trajectory_search_count is None:
        trajectory_search_count = extra_info.get("trajectory_search_count")

    search_tool_count = _count_valid_search_calls(solution_str)
    is_none = float(variant != "with_gt")
    search_truncated = float(bool(re.search(
        r"<tool_response>.*?\.\.\.\(truncated\).*?</tool_response>", solution_str, flags=re.DOTALL
    )))

    # ── 1. Format ────────────────────────────────────────────────────────
    if not _has_valid_format(solution_str):
        return {
            "score": OUTCOME_MIN,
            "outcome": OUTCOME_MIN,
            "format_ok": 0.0,
            "search_ok": 0.0,
            "retrieval_recall": 0.0,
            "search_tool_count": float(search_tool_count),
            "variant_is_none": is_none,
            "search_truncated": search_truncated,
        }

    # ── 2. Search behavior (aligned with interaction.py) ─────────────────
    searched = search_tool_count > 0
    recall = 0.0
    if variant == "with_gt":
        search_reward = WITHGT_REWARD if not searched else -2.0
    else:
        if not searched:
            search_reward = -2.0
        else:
            retrieved = _extract_retrieved_tool_names(solution_str)
            if ground_truth and retrieved:
                gt_calls = _parse_tool_calls(ground_truth)
                if gt_calls:
                    gt_names = {list(c.keys())[0].lower() for c in gt_calls}
                    retrieved_lower = {n.lower() for n in retrieved}
                    recall = len(gt_names & retrieved_lower) / len(gt_names)
            search_reward = SEARCH_BASE + RECALL_REWARD * recall
            if trajectory_search_count is not None and search_tool_count > trajectory_search_count + 1:
                search_reward -= COUNT_PENALTY

    if search_reward < 0:
        return {
            "score": search_reward,
            "outcome": 0.0,
            "format_ok": 1.0,
            "search_ok": search_reward,
            "retrieval_recall": recall,
            "search_tool_count": float(search_tool_count),
            "variant_is_none": is_none,
            "search_truncated": search_truncated,
        }

    # ── 3. Outcome (fine-grained matching with partial credit) ───────────
    pred_str = _extract_tool_calls_string(solution_str)

    if not pred_str:
        if searched and recall == 0.0:
            return {
                "score": HONEST_REJECT_REWARD,
                "outcome": 0.0,
                "format_ok": 1.0,
                "search_ok": search_reward,
                "retrieval_recall": 0.0,
                "search_tool_count": float(search_tool_count),
                "variant_is_none": is_none,
                "search_truncated": search_truncated,
            }

    outcome = OUTCOME_MIN
    if pred_str and ground_truth:
        pred_calls = _parse_tool_calls(pred_str)
        gt_calls = _parse_tool_calls(ground_truth)
        if pred_calls is not None and gt_calls is not None:
            outcome = _compute_tool_call_score(pred_calls, gt_calls)

    score = FORMAT_REWARD + outcome + search_reward
    return {
        "score": score,
        "outcome": outcome,
        "search_ok": search_reward,
        "format_ok": 1.0,
        "retrieval_recall": recall,
        "search_tool_count": float(search_tool_count),
        "variant_is_none": is_none,
        "search_truncated": search_truncated,
    }
