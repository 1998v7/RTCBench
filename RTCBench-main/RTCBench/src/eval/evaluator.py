"""Evaluator: BFCL ast_checker + retrieval metrics on inference results."""
from __future__ import annotations

import ast as _ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

from ..config import BFCL_ROOT, POSSIBLE_ANSWER_DIR, TEST_DATA_DIR
from .metrics import recall_at_k, mrr, gt_all_retrieved


def _ensure_bfcl_on_path() -> None:
    if BFCL_ROOT.exists() and str(BFCL_ROOT) not in sys.path:
        sys.path.insert(0, str(BFCL_ROOT))


def _patch_convert_func_name() -> None:
    """Bypass BFCL's convert_func_name which requires model_name in MODEL_CONFIG_MAPPING.

    The conversion only matters for API models (OpenAI/Google) that replace '.' with '_'
    in function names. For our local models this is never needed, and unknown model names
    would cause a KeyError.
    """
    import bfcl_eval.eval_checker.ast_eval.ast_checker as _mod
    _mod.convert_func_name = lambda func_name, model_name: func_name


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


class Evaluator:
    """Evaluate inference results for a BFCL-retrieval category.

    Parameters
    ----------
    test_category : e.g. "simple_python", "multiple", etc.
    data_dir : override for test data directory.
    """

    def __init__(
        self,
        test_category: str,
        data_dir: Path | None = None,
    ):
        self.test_category = test_category
        self.data_dir = data_dir or TEST_DATA_DIR
        _ensure_bfcl_on_path()

    def _load_test_entries(self) -> list[dict]:
        path = self.data_dir / f"BFCL_v4_retrieval_{self.test_category}.json"
        return load_jsonl(path)

    def _load_ground_truth(self) -> list[dict]:
        path = (
            self.data_dir / "possible_answer"
            / f"BFCL_v4_retrieval_{self.test_category}.json"
        )
        if not path.exists():
            path = POSSIBLE_ANSWER_DIR / f"BFCL_v4_retrieval_{self.test_category}.json"
        if not path.exists():
            return []
        return load_jsonl(path)

    @staticmethod
    def _parse_non_search_calls(text: str) -> list[str]:
        """Detect non-search_tool function call names in raw model output.

        Uses regex to catch calls even when ast.parse fails (e.g. unquoted args).
        Returns a list of function names that are NOT search_tool.
        """
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        names = re.findall(r"(\w[\w.]*)(?=\s*\()", text)
        return [n for n in names if n != "search_tool"]

    def evaluate(self, results: list[dict]) -> list[dict]:
        """Run evaluation on inference results.

        Returns a list of per-entry score dicts.
        """
        from bfcl_eval.constants.enums import Language
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
        _patch_convert_func_name()

        test_entries = self._load_test_entries()
        gt_list = self._load_ground_truth()
        if not gt_list:
            raise ValueError(
                f"No ground truth for {self.test_category}"
            )

        entry_by_id = {e["id"]: e for e in test_entries}
        gt_by_id = {g["id"]: g for g in gt_list}

        scores: list[dict] = []
        for r in results:
            entry_id = r["id"]
            test_entry = entry_by_id.get(entry_id)
            gt_entry = gt_by_id.get(entry_id)

            if not test_entry or not gt_entry:
                scores.append({
                    "id": entry_id,
                    "valid": False,
                    "error": "Missing test entry or ground truth",
                })
                continue

            # retrieval metrics
            gt_names = test_entry.get("gt_tool_names", [])
            if not gt_names and gt_entry:
                for call in gt_entry.get("ground_truth", []):
                    if isinstance(call, dict):
                        gt_names.extend(call.keys())
                gt_names = list(dict.fromkeys(gt_names))
            search_results = r.get("search_tools_results", [])
            search_count = len(search_results)

            entry_recall = recall_at_k(gt_names, search_results)
            entry_mrr = mrr(gt_names, search_results)
            entry_gt_retrieved = gt_all_retrieved(gt_names, search_results)

            retrieved_tool_names = []
            for sr in search_results:
                for tool in sr.get("retrieved", []):
                    name = tool.get("name", "") if isinstance(tool, dict) else ""
                    if name and name not in retrieved_tool_names:
                        retrieved_tool_names.append(name)

            # detect direct tool-call attempts in first-turn raw output
            raw_outputs = r.get("metadata", {}).get("raw_model_outputs", [])
            direct_tool_attempt = False
            if search_count == 0 and raw_outputs:
                first_raw = raw_outputs[0] if raw_outputs else ""
                if isinstance(first_raw, str) and first_raw.strip():
                    first_turn_calls = self._parse_non_search_calls(first_raw)
                    if first_turn_calls:
                        direct_tool_attempt = True

            # function calling evaluation
            decoded = r.get("model_responses_decoded", [])
            gt_answer = gt_entry["ground_truth"]

            common_fields = {
                "search_count": search_count,
                "recall_at_k": entry_recall,
                "mrr": entry_mrr,
                "gt_all_retrieved": entry_gt_retrieved,
                "retrieved_tool_names": retrieved_tool_names,
                "direct_tool_attempt": direct_tool_attempt,
            }

            if not decoded:
                scores.append({
                    "id": entry_id,
                    "valid": False,
                    "error_type": "no_tool_call",
                    "model_output": decoded,
                    "ground_truth": gt_answer,
                    **common_fields,
                })
                continue

            try:
                result = ast_checker(
                    func_description=test_entry["function"],
                    model_output=decoded,
                    possible_answer=gt_answer,
                    language=Language.PYTHON,
                    test_category=self.test_category,
                    model_name="default",
                )
                result["id"] = entry_id
                result["model_output"] = decoded
                result["ground_truth"] = gt_answer
                result.update(common_fields)
                scores.append(result)
            except Exception as e:
                scores.append({
                    "id": entry_id,
                    "valid": False,
                    "error": str(e),
                    "error_type": "eval_error",
                    "model_output": decoded,
                    "ground_truth": gt_answer,
                    **common_fields,
                })

        return scores

    @staticmethod
    def _get_called_tool_names(score_entry: dict) -> set[str]:
        """Extract tool names the model actually called."""
        names: set[str] = set()
        for call in score_entry.get("model_output", []):
            if isinstance(call, dict):
                names.update(call.keys())
        return names

    @staticmethod
    def compute_stats(scores: list[dict]) -> dict:
        """Compute summary statistics from per-entry scores."""
        total = len(scores)
        valid = sum(1 for s in scores if s.get("valid"))

        recalls = [s["recall_at_k"] for s in scores if s.get("recall_at_k") is not None]
        mrrs = [s["mrr"] for s in scores if s.get("mrr") is not None]
        gt_retrieved_count = sum(
            1 for s in scores if s.get("gt_all_retrieved") is True
        )
        gt_evaluated_count = sum(
            1 for s in scores if s.get("gt_all_retrieved") is not None
        )

        search_counts = [s.get("search_count", 0) for s in scores]
        no_tool_call = sum(
            1 for s in scores if s.get("error_type") == "no_tool_call"
        )

        # --- hallucination metrics ---
        has_call = lambda s: s.get("error_type") != "no_tool_call" and s.get("model_output")

        # Type 1: skipped search — either decoded calls exist or raw output
        # shows the model attempted to call non-search tools directly
        no_search_call = sum(
            1 for s in scores
            if s.get("search_count", 0) == 0
            and (has_call(s) or s.get("direct_tool_attempt"))
        )
        # Type 2: searched but GT tools not fully retrieved, model still called
        incomplete_retr_call = sum(
            1 for s in scores
            if s.get("search_count", 0) > 0
            and s.get("gt_all_retrieved") is False
            and has_call(s)
        )
        # Type 3: model called tool names not present in search results
        phantom_tool_entries = 0
        phantom_tool_total = 0
        for s in scores:
            if not has_call(s):
                continue
            called = Evaluator._get_called_tool_names(s)
            retrieved = set(s.get("retrieved_tool_names", []))
            phantom = called - retrieved if retrieved else set()
            if phantom:
                phantom_tool_entries += 1
                phantom_tool_total += len(phantom)

        n_with_calls = sum(1 for s in scores if has_call(s))

        return {
            "total": total,
            "accuracy": valid / total if total else 0,
            "valid_count": valid,
            "no_tool_call_count": no_tool_call,
            "retrieval": {
                "searched_count": gt_evaluated_count,
                "mean_recall_at_k": sum(recalls) / len(recalls) if recalls else None,
                "mean_mrr": sum(mrrs) / len(mrrs) if mrrs else None,
                "gt_all_retrieved": gt_retrieved_count,
                "gt_all_retrieved_ratio": (
                    gt_retrieved_count / gt_evaluated_count
                    if gt_evaluated_count else None
                ),
            },
            "hallucination": {
                "no_search_call": no_search_call,
                "incomplete_retr_call": incomplete_retr_call,
                "phantom_tool_entries": phantom_tool_entries,
                "phantom_tool_total": phantom_tool_total,
                "n_with_calls": n_with_calls,
            },
            "search_calls": {
                "total": sum(search_counts),
                "avg": sum(search_counts) / total if total else 0,
                "max": max(search_counts) if search_counts else 0,
                "distribution": dict(Counter(search_counts)),
            },
        }

    @staticmethod
    def _fmt(val, fmt=".2%", fallback="N/A"):
        return f"{val:{fmt}}" if val is not None else fallback

    @staticmethod
    def format_report(scores: list[dict], test_category: str) -> str:
        """Return a compact evaluation report."""
        s = Evaluator.compute_stats(scores)
        f = Evaluator._fmt
        r = s["retrieval"]
        h = s["hallucination"]
        sc = s["search_calls"]

        searched = r["searched_count"]
        total = s["total"]
        h_total = h["no_search_call"] + h["phantom_tool_entries"]
        h_pct = h_total / total * 100 if total else 0

        lines = [
            f"==== {test_category} ({total} samples) ====",
            f"  Accuracy: {s['valid_count']}/{total} = {s['accuracy']:.2%}",
            f"  Retrieval: searched {searched}/{total}"
            f"  |  Recall@K: {f(r['mean_recall_at_k'])}"
            f"  |  GT all retrieved: {r['gt_all_retrieved']}/{searched} = {f(r['gt_all_retrieved_ratio'])}"
            f"  |  Avg search calls: {sc['avg']:.2f}",
            f"  Hallucination ({h_pct:.2f}%, {h_total}/{total})"
            f"  | {{skip_search: {h['no_search_call']}"
            f", phantom_tool: {h['phantom_tool_entries']}}}",
        ]
        return "\n".join(lines)
