"""Evaluator for HammerBench categories: ROUGE-L fuzzy matching + retrieval metrics.

Uses ROUGE-L (F-score >= 0.7 threshold) for parameter value matching,
following HammerBench's original evaluation methodology.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from rouge import Rouge

from ..config import HAMMERBENCH_ZH_DIR
from .metrics import recall_at_k, mrr, gt_all_retrieved
from . import hallucination as _h

_rouge = Rouge()

ROUGEL_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# ROUGE-L helpers (ported from HammerBench official eval)
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r"([\u4e00-\u9fff])")


def _rouge_l_f_en(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F-score between two English strings."""
    ref = reference.strip(".?! ").lower()
    hyp = hypothesis.strip(".?! ").lower()
    if not hyp and not ref:
        return 1.0
    if not hyp or not ref:
        return 0.0
    try:
        scores = _rouge.get_scores(hyp, ref)
        return scores[0]["rouge-l"]["f"]
    except Exception:
        return 0.0


def _rouge_l_f_zh(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F-score between two Chinese strings.

    Inserts spaces around each CJK character so rouge treats them as
    individual tokens, matching HammerBench's official implementation.
    """
    ref = _CJK_RE.sub(r" \1 ", reference.strip("。！ "))
    hyp = _CJK_RE.sub(r" \1 ", hypothesis.strip("。！ "))
    if not hyp.strip() and not ref.strip():
        return 1.0
    if not hyp.strip() or not ref.strip():
        return 0.0
    try:
        scores = _rouge.get_scores(hyp, ref)
        return scores[0]["rouge-l"]["f"]
    except Exception:
        return 0.0


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _rouge_l_f(reference: str, hypothesis: str) -> float:
    """Auto-detect language and use the appropriate ROUGE-L function."""
    if _has_cjk(reference) or _has_cjk(hypothesis):
        return _rouge_l_f_zh(reference, hypothesis)
    return _rouge_l_f_en(reference, hypothesis)


def _params_match_rougel(
    gt_params: dict,
    model_params: dict,
    threshold: float = ROUGEL_THRESHOLD,
) -> bool:
    """Check if model params match GT params using ROUGE-L.

    GT params are in BFCL format: {key: [val1, val2, ...]} where any
    value in the list is acceptable. Model params are {key: val}.

    Steps (following HammerBench logic):
    1. If model has keys not in GT -> False
    2. Exact match -> True
    3. For each GT key, compute ROUGE-L between model value and GT value;
       all must be >= threshold.
    """
    gt_flat = {}
    for k, v in gt_params.items():
        sv = str(v[0]) if isinstance(v, list) and len(v) > 0 else str(v)
        if sv.strip():
            gt_flat[k] = sv

    model_flat = {}
    for k, v in model_params.items():
        sv = str(v)
        if sv.strip():
            model_flat[k] = sv

    for k in model_flat:
        if k not in gt_flat:
            return False

    if gt_flat == model_flat:
        return True

    scores = []
    for k, gt_val in gt_flat.items():
        if k not in model_flat:
            scores.append(0.0)
        else:
            scores.append(_rouge_l_f(gt_val, model_flat[k]))

    return len(gt_flat) > 0 and min(scores) >= threshold


def _func_name_match(gt_name: str, model_name: str) -> bool:
    """Check function name match (case-insensitive, dot-normalized)."""
    return gt_name.replace("_", ".").lower() == model_name.replace("_", ".").lower()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------
# HammerBench Evaluator
# ---------------------------------------------------------------------------

class HammerBenchEvaluator:
    """Evaluate inference results for HammerBench categories using ROUGE-L.

    Parameters
    ----------
    test_category : e.g. "hammerbench".
    data_dir : override for test data directory.
    rougel_threshold : minimum ROUGE-L F-score for parameter value match.
    """

    def __init__(
        self,
        test_category: str,
        data_dir: Path | None = None,
        rougel_threshold: float = ROUGEL_THRESHOLD,
    ):
        self.test_category = test_category
        self.data_dir = data_dir or HAMMERBENCH_ZH_DIR
        self.rougel_threshold = rougel_threshold

    def _load_test_entries(self) -> list[dict]:
        path = self.data_dir / f"{self.test_category}.json"
        return load_jsonl(path)

    def _load_ground_truth(self) -> list[dict]:
        path = (
            self.data_dir / "possible_answer"
            / f"{self.test_category}.json"
        )
        if not path.exists():
            return []
        return load_jsonl(path)

    @staticmethod
    def _parse_non_search_calls(text: str) -> list[str]:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        names = re.findall(r"(\w[\w.]*)(?=\s*\()", text)
        return [n for n in names if n != "search_tool"]

    def _check_single_call(
        self,
        model_call: dict,
        gt_call: dict,
    ) -> dict:
        """Check a single model call against a single GT call.

        Returns {valid, error_type, error, details}.
        """
        gt_name = list(gt_call.keys())[0]
        gt_params = gt_call[gt_name]

        model_name = list(model_call.keys())[0]
        model_params = model_call[model_name]

        if not _func_name_match(gt_name, model_name):
            return {
                "valid": False,
                "error_type": "wrong_func_name",
                "error": f"Expected '{gt_name}', got '{model_name}'",
            }

        if _params_match_rougel(gt_params, model_params, self.rougel_threshold):
            return {"valid": True, "error_type": "", "error": ""}

        mismatches = []
        gt_flat = {k: (str(v[0]) if isinstance(v, list) and v else str(v)) for k, v in gt_params.items()}
        model_flat = {k: str(v) for k, v in model_params.items()}

        for k in model_flat:
            if k not in gt_flat:
                mismatches.append(f"Extra param '{k}'='{model_flat[k]}'")

        for k, gv in gt_flat.items():
            if not gv.strip():
                continue
            if k not in model_flat:
                mismatches.append(f"Missing param '{k}' (expected '{gv}')")
            else:
                mv = model_flat[k]
                score = _rouge_l_f(gv, mv)
                if score < self.rougel_threshold:
                    mismatches.append(
                        f"Param '{k}': model='{mv}' vs gt='{gv}' (ROUGE-L={score:.3f})"
                    )

        return {
            "valid": False,
            "error_type": "value_error:rougel",
            "error": "; ".join(mismatches) if mismatches else "Parameter mismatch",
        }

    def evaluate(self, results: list[dict]) -> list[dict]:
        """Run evaluation on inference results. Returns per-entry score dicts."""
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

            # retrieval metrics
            gt_names = []
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

            if len(decoded) != len(gt_answer):
                # wrong number of calls
                errors = [
                    f"Expected {len(gt_answer)} call(s), got {len(decoded)}"
                ]
                gt_name_set = {list(c.keys())[0] for c in gt_answer}
                model_name_set = {list(c.keys())[0] for c in decoded}
                if gt_name_set != model_name_set:
                    errors.append(
                        f"GT names: {gt_name_set}, Model names: {model_name_set}"
                    )
                scores.append({
                    "id": entry_id,
                    "valid": False,
                    "error_type": "wrong_count",
                    "error": errors,
                    "model_output": decoded,
                    "ground_truth": gt_answer,
                    **common_fields,
                })
                continue

            all_valid = True
            all_errors = []
            all_error_types = []

            for model_call, gt_call in zip(decoded, gt_answer):
                check = self._check_single_call(model_call, gt_call)
                if not check["valid"]:
                    all_valid = False
                    all_errors.append(check["error"])
                    all_error_types.append(check["error_type"])

            if all_valid:
                scores.append({
                    "id": entry_id,
                    "valid": True,
                    "error": [],
                    "error_type": "",
                    "model_output": decoded,
                    "ground_truth": gt_answer,
                    **common_fields,
                })
            else:
                scores.append({
                    "id": entry_id,
                    "valid": False,
                    "error": all_errors,
                    "error_type": "|".join(all_error_types),
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

        # error type breakdown
        error_types: dict[str, int] = {}
        for s in scores:
            if s.get("valid"):
                continue
            et = s.get("error_type", "unknown")
            for sub_et in et.split("|"):
                sub_et = sub_et.strip()
                if sub_et:
                    error_types[sub_et] = error_types.get(sub_et, 0) + 1

        return {
            "total": total,
            "accuracy": valid / total if total else 0,
            "valid_count": valid,
            "no_tool_call_count": no_tool_call,
            "error_type_breakdown": error_types,
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
        s = HammerBenchEvaluator.compute_stats(scores)
        f = HammerBenchEvaluator._fmt
        r = s["retrieval"]
        h = s["hallucination"]
        sc = s["search_calls"]
        et = s["error_type_breakdown"]

        searched = r["searched_count"]
        total = s["total"]
        h_total = h["hallucination_entries"]
        h_permille = h["hallucination_permille"]

        lines = [
            f"==== {test_category} ({total} samples) ====",
            f"  Accuracy (ROUGE-L≥{ROUGEL_THRESHOLD}): {s['valid_count']}/{total} = {s['accuracy']:.2%}",
            f"  Retrieval: searched {searched}/{total}"
            f"  |  Recall@K: {f(r['mean_recall_at_k'])}"
            f"  |  GT all retrieved: {r['gt_all_retrieved']}/{searched} = {f(r['gt_all_retrieved_ratio'])}"
            f"  |  Avg search calls: {sc['avg']:.2f}",
            f"  Hallucination ({h_permille:.2f}‰, {h_total}/{total})"
            f"  | {{skip_search: {h['skip_search']}"
            f", phantom_tool: {h['phantom_tool_entries']}}}",
        ]

        if et:
            et_parts = [f"{k}: {v}" for k, v in sorted(et.items(), key=lambda x: -x[1])]
            lines.append(f"  Errors: {{{', '.join(et_parts)}}}")

        return "\n".join(lines)
