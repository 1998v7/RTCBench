"""Evaluator for BFCL categories: strict ast_checker + retrieval metrics."""
from __future__ import annotations

import ast as _ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

from ..config import BFCL_ROOT, POSSIBLE_ANSWER_DIR, TEST_DATA_DIR
from .metrics import recall_at_k, mrr, gt_all_retrieved
from . import hallucination as _h


def _ensure_bfcl_on_path() -> None:
    if BFCL_ROOT.exists() and str(BFCL_ROOT) not in sys.path:
        sys.path.insert(0, str(BFCL_ROOT))


def _patch_convert_func_name() -> None:
    """Bypass BFCL's convert_func_name which requires model_name in MODEL_CONFIG_MAPPING."""
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
    """Evaluate inference results for a BFCL-retrieval category (strict matching).

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
        """Detect non-search_tool function call names in raw model output."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        names = re.findall(r"(\w[\w.]*)(?=\s*\()", text)
        return [n for n in names if n != "search_tool"]

    def evaluate(self, results: list[dict]) -> list[dict]:
        """Run evaluation on inference results. Returns per-entry score dicts."""
        from bfcl_eval.constants.enums import Language
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
        _patch_convert_func_name()

        test_entries = self._load_test_entries()
        gt_list = self._load_ground_truth()
        if not gt_list:
            raise ValueError(f"No ground truth for {self.test_category}")

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

            retrieved_tool_names = _h.collect_retrieved_names(search_results)

            raw_outputs = r.get("metadata", {}).get("raw_model_outputs", [])
            variant = r.get("metadata", {}).get("variant", "none")
            initial_names = _h.initial_tool_names(test_entry, variant)

            decoded = r.get("model_responses_decoded", [])
            gt_answer = gt_entry["ground_truth"]

            called_names = _h.collect_called_names(decoded, raw_outputs)
            direct_tool_attempt = bool(
                search_count == 0
                and not decoded
                and called_names
            )

            halluc_flags = _h.per_sample_flags(
                variant=variant,
                search_count=search_count,
                initial_names=initial_names,
                retrieved_names=retrieved_tool_names,
                called_names=called_names,
            )

            common_fields = {
                "variant": variant,
                "search_count": search_count,
                "recall_at_k": entry_recall,
                "mrr": entry_mrr,
                "gt_all_retrieved": entry_gt_retrieved,
                "initial_tool_names": initial_names,
                "retrieved_tool_names": retrieved_tool_names,
                "called_tool_names": called_names,
                "direct_tool_attempt": direct_tool_attempt,
                **halluc_flags,
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
        names: set[str] = set()
        for call in score_entry.get("model_output", []):
            if isinstance(call, dict):
                names.update(call.keys())
        return names

    @staticmethod
    def compute_stats(scores: list[dict]) -> dict:
        total = len(scores)
        valid = sum(1 for s in scores if s.get("valid"))

        recalls = [s["recall_at_k"] for s in scores if s.get("recall_at_k") is not None]
        mrrs = [s["mrr"] for s in scores if s.get("mrr") is not None]
        gt_retrieved_count = sum(1 for s in scores if s.get("gt_all_retrieved") is True)
        gt_evaluated_count = sum(1 for s in scores if s.get("gt_all_retrieved") is not None)

        search_counts = [s.get("search_count", 0) for s in scores]
        no_tool_call = sum(1 for s in scores if s.get("error_type") == "no_tool_call")

        has_call = lambda s: s.get("error_type") != "no_tool_call" and s.get("model_output")

        # Forced calls under incomplete retrieval -- not a hallucination per
        # the paper definition, kept as an auxiliary diagnostic.
        incomplete_retr_call = sum(
            1 for s in scores
            if s.get("search_count", 0) > 0
            and s.get("gt_all_retrieved") is False
            and has_call(s)
        )

        halluc_agg = _h.aggregate(scores)
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
                # Paper Section 3.5: per-sample union of {skip-search, phantom}.
                "skip_search": halluc_agg["skip_search"],
                "phantom_tool_entries": halluc_agg["phantom_tool_entries"],
                "phantom_tool_total": halluc_agg["phantom_tool_total"],
                "hallucination_entries": halluc_agg["hallucination_entries"],
                "hallucination_rate": halluc_agg["hallucination_rate"],
                "hallucination_permille": halluc_agg["hallucination_permille"],
                "skip_search_rate": halluc_agg["skip_search_rate"],
                "phantom_tool_rate": halluc_agg["phantom_tool_rate"],
                # Auxiliary, not in paper's hallucination definition:
                "incomplete_retr_call": incomplete_retr_call,
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
        s = Evaluator.compute_stats(scores)
        f = Evaluator._fmt
        r = s["retrieval"]
        h = s["hallucination"]
        sc = s["search_calls"]

        searched = r["searched_count"]
        total = s["total"]
        h_total = h["hallucination_entries"]
        h_permille = h["hallucination_permille"]

        lines = [
            f"==== {test_category} ({total} samples) ====",
            f"  Accuracy: {s['valid_count']}/{total} = {s['accuracy']:.2%}",
            f"  Retrieval: searched {searched}/{total}"
            f"  |  Recall@K: {f(r['mean_recall_at_k'])}"
            f"  |  GT all retrieved: {r['gt_all_retrieved']}/{searched} = {f(r['gt_all_retrieved_ratio'])}"
            f"  |  Avg search calls: {sc['avg']:.2f}",
            f"  Hallucination ({h_permille:.2f}\u2030, {h_total}/{total})"
            f"  | {{skip_search: {h['skip_search']}"
            f", phantom_tool: {h['phantom_tool_entries']}}}",
        ]
        return "\n".join(lines)
