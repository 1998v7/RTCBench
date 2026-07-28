#!/usr/bin/env python3
"""Post-evaluation LLM judge for BFCL-Retrieval.

Finds entries where the model called different-but-possibly-equivalent tools
and uses an LLM to judge whether the model's response is functionally
equivalent to the ground truth.

Usage:
  python run_judge.py --model-dir results/openrouter_gpt-oss-120b \
      --categories simple_python multiple parallel parallel_multiple \
      --judge-model openai/gpt-4.1-mini \
      --base-url https://openrouter.ai/api/v1 \
      --api-key $OPENROUTER_API_KEY
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    category_scope, get_scope_config,
)
from src.eval import hallucination as _h

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are an expert tool-calling evaluator.

A LLM was asked to call tool(s) to answer a user question. It called different tool(s) than the expected ground-truth (GT). 

Determine whether the tools called by the LLM is **functionally equivalent** to the GT and able to solve the user request.

## User Question
{question}

## Tool Definitions
{tool_defs}

## Ground-Truth Calls
{gt_calls}

## Model's Calls
{model_calls}

## You SHOULD consider the following criteria:
- Do the model's calls serve the same purpose as the Ground-Truth calls for this question?
- Are the arguments semantically equivalent (even if parameter names differ)?
- Would executing the model's calls give the user an answer equivalent to the Ground-Truth?

Answer with a JSON object (no markdown fences):
{{"equivalent": true/false, "reason": "brief explanation"}}
"""


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


def _format_tool_def(tool: dict) -> str:
    """Format a single tool definition compactly."""
    name = tool.get("name", "?")
    desc = tool.get("description", "")
    params = tool.get("parameters", {}).get("properties", {})
    required = set(tool.get("parameters", {}).get("required", []))
    parts = [f"- {name}: {desc}"]
    if params:
        for pname, pmeta in params.items():
            ptype = pmeta.get("type", "any")
            pdesc = pmeta.get("description", "")
            opt = "" if pname in required else " (optional)"
            parts.append(f"    - {pname}: {ptype}{opt} — {pdesc}")
    return "\n".join(parts)


def _format_call(call: dict) -> str:
    """Format a tool call as func_name(k=v, ...)."""
    name = list(call.keys())[0]
    args = call[name]
    if isinstance(args, dict):
        arg_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
    else:
        arg_str = repr(args)
    return f"{name}({arg_str})"


def _resolve_tool_def(
    name: str,
    tool_bank: dict[str, dict],
    inference_entry: dict,
) -> dict | None:
    """Look up tool definition from tool bank or retrieval results."""
    if name in tool_bank:
        return tool_bank[name]
    for sr in inference_entry.get("search_tools_results", []):
        for t in sr.get("retrieved", []):
            if t.get("name") == name:
                return t
    return None


def _has_name_mismatch(eval_entry: dict) -> bool:
    """Check whether the error involves tool-name mismatches (not just param issues)."""
    model_output = eval_entry.get("model_output", [])
    gt_answer = eval_entry.get("ground_truth", [])
    if not model_output or not gt_answer:
        return False
    gt_names = {list(c.keys())[0] for c in gt_answer}
    model_names = {list(c.keys())[0] for c in model_output}
    return bool(gt_names - model_names)


def _is_judgeable(eval_entry: dict) -> bool:
    """Return True if this eval entry might benefit from LLM judging."""
    error_type = eval_entry.get("error_type", "")
    if eval_entry.get("valid", False):
        return False
    if not eval_entry.get("model_output"):
        return False
    if "wrong_func_name" in error_type:
        return True
    if "cannot_find_match" in error_type or "wrong_count" in error_type:
        return _has_name_mismatch(eval_entry)
    return False


def build_judge_input(
    eval_entry: dict,
    test_entry: dict,
    inference_entry: dict,
    tool_bank: dict[str, dict],
) -> dict | None:
    """Build judge input: question, GT calls, model calls, and all tool defs."""
    if not _is_judgeable(eval_entry):
        return None

    model_output = eval_entry.get("model_output", [])
    gt_answer = eval_entry.get("ground_truth", [])
    if not model_output or not gt_answer:
        return None

    question = test_entry["question"][0][0]["content"]

    gt_tool_defs = {f["name"]: f for f in test_entry.get("function", [])}
    model_tool_names = {list(c.keys())[0] for c in model_output}

    all_tool_defs = dict(gt_tool_defs)
    for name in model_tool_names:
        if name not in all_tool_defs:
            tdef = _resolve_tool_def(name, tool_bank, inference_entry)
            if tdef:
                all_tool_defs[name] = tdef

    return {
        "id": eval_entry["id"],
        "question": question,
        "gt_calls": gt_answer,
        "model_calls": model_output,
        "tool_defs": all_tool_defs,
    }


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

REASONING_MODELS = {"openai/o3-mini", "openai/o4-mini", "openai/o3", "openai/o1"}


def _extract_json(text: str) -> dict | None:
    """Try hard to extract a JSON object from potentially messy model output."""
    if not text:
        return None
    import re
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*\"equivalent\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    if '"equivalent"' in text:
        equiv = "true" in text.lower().split('"equivalent"')[1][:20]
        reason_m = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
        return {"equivalent": equiv, "reason": reason_m.group(1) if reason_m else ""}
    return None


def call_judge(client, judge_model: str, judge_input: dict, retries: int = 10) -> dict:
    tool_defs_str = "\n".join(
        _format_tool_def(d) for d in judge_input["tool_defs"].values()
    )
    gt_calls_str = "\n".join(
        f"- {_format_call(c)}" for c in judge_input["gt_calls"]
    )
    model_calls_str = "\n".join(
        f"- {_format_call(c)}" for c in judge_input["model_calls"]
    )

    prompt = JUDGE_PROMPT.format(
        question=judge_input["question"],
        tool_defs=tool_defs_str,
        gt_calls=gt_calls_str,
        model_calls=model_calls_str,
    )

    is_reasoning = judge_model in REASONING_MODELS
    api_kwargs: dict = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if is_reasoning:
        api_kwargs["temperature"] = 1
        api_kwargs["max_tokens"] = 4096
    else:
        api_kwargs["temperature"] = 0.0
        api_kwargs["max_tokens"] = 512

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(**api_kwargs)
            msg = response.choices[0].message
            content = (msg.content or "").strip()
            if not content:
                for attr in ("reasoning_content", "reasoning"):
                    content = getattr(msg, attr, None) or ""
                    if content:
                        break
                if not content and hasattr(msg, "model_extra") and isinstance(msg.model_extra, dict):
                    content = msg.model_extra.get("reasoning_content", "") or msg.model_extra.get("reasoning", "") or ""
                if not content:
                    raw_resp = response.model_dump() if hasattr(response, "model_dump") else str(response)
                    raise ValueError(f"Empty response from API. Raw: {str(raw_resp)[:300]}")
                content = content.strip()

            result = _extract_json(content)
            if result is None:
                raise ValueError(f"Could not parse JSON from: {content[:200]}")
            return {
                "id": judge_input["id"],
                "equivalent": bool(result.get("equivalent", False)),
                "reason": str(result.get("reason", "")),
                "raw_response": content[:500],
            }
        except Exception as e:
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
            else:
                return {
                    "id": judge_input["id"],
                    "equivalent": False,
                    "reason": f"Judge error: {type(e).__name__}: {e}",
                    "raw_response": "",
                }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Post-evaluation LLM judge")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--judge-model", default="openai/gpt-4.1-mini")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--detail", default=None, help="Path to write detailed judge report text")
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key or os.getenv("OPENROUTER_API_KEY", ""),
    )

    model_dir = Path(args.model_dir)

    all_judge_results = []
    all_original_scores = []

    for cat in args.categories:
        eval_path = model_dir / f"Eval_retrieval_{cat}.json"
        inference_path = model_dir / f"Inference_retrieval_{cat}.json"

        is_hammerbench = cat.startswith("hammerbench")
        scope = category_scope(cat)
        scope_cfg = get_scope_config(scope)
        test_data_dir = scope_cfg["test_data_dir"]
        test_fname = f"{cat}.json" if is_hammerbench else f"BFCL_v4_retrieval_{cat}.json"
        test_path = test_data_dir / test_fname

        if not eval_path.exists():
            print(f"[SKIP] {eval_path} not found")
            continue

        eval_scores = load_jsonl(eval_path)
        inference_results = {e["id"]: e for e in load_jsonl(inference_path)}
        test_entries = {e["id"]: e for e in load_jsonl(test_path)}

        tb_file = scope_cfg["tool_bank"]
        with open(tb_file) as f:
            tool_bank = {t["name"]: t for t in json.load(f)["tools"]}

        judgeable = [s for s in eval_scores if _is_judgeable(s)]
        print(f"\n[{cat}] Total={len(eval_scores)}  judgeable={len(judgeable)}")

        judge_inputs = []
        for entry in judgeable:
            ji = build_judge_input(
                entry,
                test_entries[entry["id"]],
                inference_results[entry["id"]],
                tool_bank,
            )
            if ji:
                judge_inputs.append(ji)

        if not judge_inputs:
            all_original_scores.extend(eval_scores)
            continue

        print(f"  Judging {len(judge_inputs)} entries with {args.judge_model}...")
        judge_results = {}
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(call_judge, client, args.judge_model, ji): ji["id"]
                for ji in judge_inputs
            }
            for future in as_completed(futures):
                entry_id = futures[future]
                result = future.result()
                judge_results[entry_id] = result

        equiv_count = sum(1 for r in judge_results.values() if r["equivalent"])
        print(f"  Equivalent: {equiv_count}/{len(judge_results)}")

        for r in sorted(judge_results.values(), key=lambda x: x["id"]):
            tag = "EQUIV" if r["equivalent"] else "DIFF "
            print(f"    [{tag}] {r['id']}: {r['reason'][:120]}")

        adjusted_scores = []
        for s in eval_scores:
            s_copy = dict(s)
            if s["id"] in judge_results and judge_results[s["id"]]["equivalent"]:
                s_copy["valid"] = True
                s_copy["judge_overridden"] = True
                s_copy["judge_reason"] = judge_results[s["id"]]["reason"]
            adjusted_scores.append(s_copy)

        all_judge_results.extend(judge_results.values())
        all_original_scores.extend(adjusted_scores)

        adjusted_path = model_dir / f"Eval_retrieval_{cat}_judged.json"
        with open(adjusted_path, "w") as f:
            for s in adjusted_scores:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    if not all_original_scores:
        print("No scores to report.")
        return

    total = len(all_original_scores)
    original_valid = sum(1 for s in all_original_scores if s.get("valid") and not s.get("judge_overridden"))
    judged_equiv = sum(1 for s in all_original_scores if s.get("judge_overridden"))
    adjusted_valid = original_valid + judged_equiv

    lines: list[str] = []

    def out(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    # Per-category adjusted report
    for cat in args.categories:
        adjusted_path = model_dir / f"Eval_retrieval_{cat}_judged.json"
        if not adjusted_path.exists():
            continue
        cat_scores = load_jsonl(adjusted_path)
        cat_total = len(cat_scores)
        cat_valid = sum(1 for s in cat_scores if s.get("valid"))
        cat_overrides = sum(1 for s in cat_scores if s.get("judge_overridden"))
        cat_searched = sum(1 for s in cat_scores if s.get("search_count", 0) > 0)
        cat_retr_ok = sum(1 for s in cat_scores if s.get("gt_all_retrieved"))

        recall_sum = sum(s.get("recall_at_k", 0) for s in cat_scores if s.get("search_count", 0) > 0)
        avg_search = sum(s.get("search_count", 0) for s in cat_scores) / max(cat_total, 1)
        recall_at_k = recall_sum / max(cat_searched, 1) * 100

        # hallucination metrics (paper §3.5: per-sample union of skip_search & phantom).
        # The Eval_retrieval_*.json files written by the new evaluator already
        # carry per-sample is_skip_search / is_phantom_tool / is_hallucination
        # / phantom_tool_names, so we just aggregate.
        cat_halluc = _h.aggregate(cat_scores)
        # Auxiliary diagnostic, not in paper definition:
        h_incomplete = sum(
            1 for s in cat_scores
            if s.get("search_count", 0) > 0
            and s.get("gt_all_retrieved") is False
            and s.get("error_type") != "no_tool_call"
            and s.get("model_output")
        )

        cat_h_total = cat_halluc["hallucination_entries"]
        cat_h_permille = cat_halluc["hallucination_permille"]

        out(f"==== {cat} ({cat_total} samples, judged +{cat_overrides}) ====")
        out(f"  Accuracy: {cat_valid}/{cat_total} = {cat_valid/cat_total*100:.2f}%")
        retr_parts = [f"searched {cat_searched}/{cat_total}"]
        retr_parts.append(f"Recall@K: {recall_at_k:.2f}%")
        retr_parts.append(f"GT all retrieved: {cat_retr_ok}/{cat_searched} = {cat_retr_ok/max(cat_searched,1)*100:.2f}%")
        retr_parts.append(f"Avg search calls: {avg_search:.2f}")
        out(f"  Retrieval: {'  |  '.join(retr_parts)}")
        out(
            f"  Hallucination ({cat_h_permille:.2f}\u2030, {cat_h_total}/{cat_total})"
            f"  | {{skip_search: {cat_halluc['skip_search']}"
            f", phantom_tool: {cat_halluc['phantom_tool_entries']}"
            f", incomplete_retrieval: {h_incomplete}}}"
        )

    # Overall hallucination across all categories (paper §3.5 definition,
    # per-sample union of skip_search and phantom).
    overall_halluc = _h.aggregate(all_original_scores)
    all_incomplete = sum(
        1 for s in all_original_scores
        if s.get("search_count", 0) > 0
        and s.get("gt_all_retrieved") is False
        and s.get("error_type") != "no_tool_call"
        and s.get("model_output")
    )
    all_with_calls = sum(
        1 for s in all_original_scores
        if s.get("error_type") != "no_tool_call" and s.get("model_output")
    )

    total_h = overall_halluc["hallucination_entries"]
    total_h_permille = overall_halluc["hallucination_permille"]

    out(f"==== ALL_CATEGORIES ({total} samples, judged +{judged_equiv}) ====")
    out(f"  Original accuracy:  {original_valid}/{total} = {original_valid/total*100:.2f}%")
    out(f"  Judge overrides:    +{judged_equiv}")
    out(f"  Adjusted accuracy:  {adjusted_valid}/{total} = {adjusted_valid/total*100:.2f}%")
    out(
        f"  Hallucination ({total_h_permille:.2f}\u2030, {total_h}/{total})"
        f"  | {{skip_search: {overall_halluc['skip_search']}"
        f", phantom_tool: {overall_halluc['phantom_tool_entries']}"
        f", incomplete_retrieval: {all_incomplete}}}"
    )

    judge_report_path = model_dir / "judge_report.json"
    with open(judge_report_path, "w") as f:
        json.dump({
            "total": total,
            "original_valid": original_valid,
            "original_accuracy": original_valid / total,
            "judge_overrides": judged_equiv,
            "adjusted_valid": adjusted_valid,
            "adjusted_accuracy": adjusted_valid / total,
            "hallucination": {
                "skip_search": overall_halluc["skip_search"],
                "phantom_tool_entries": overall_halluc["phantom_tool_entries"],
                "phantom_tool_total": overall_halluc["phantom_tool_total"],
                "hallucination_entries": overall_halluc["hallucination_entries"],
                "hallucination_rate": overall_halluc["hallucination_rate"],
                "hallucination_permille": overall_halluc["hallucination_permille"],
                "skip_search_rate": overall_halluc["skip_search_rate"],
                "phantom_tool_rate": overall_halluc["phantom_tool_rate"],
                "incomplete_retr_call": all_incomplete,
                "n_with_calls": all_with_calls,
            },
            "details": all_judge_results,
        }, f, indent=2, ensure_ascii=False)
    out(f"  Report -> {judge_report_path}")

    # Save detail text file
    if args.detail:
        detail_path = Path(args.detail)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with open(detail_path, "w") as f:
            f.write("\n".join(lines))
        print(f"  Detail -> {detail_path}")


if __name__ == "__main__":
    main()
