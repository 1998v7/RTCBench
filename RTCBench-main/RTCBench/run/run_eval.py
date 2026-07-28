#!/usr/bin/env python3
"""BFCL-Retrieval evaluation entry point.

Usage:
  python run_eval.py --model-dir results/Qwen3-4B --categories simple_python multiple parallel parallel_multiple
  python run_eval.py --model-dir results/gpt-4o-mini --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    TARGET_CATEGORIES,
    category_scope, get_scope_config,
)


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_jsonl(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="BFCL-Retrieval evaluation")
    parser.add_argument("--model-dir", required=True, help="Directory containing Inference_retrieval_*.json files")
    parser.add_argument("--all", action="store_true", help="Evaluate all categories")
    parser.add_argument("--categories", nargs="+", default=None, help="Categories to evaluate")
    parser.add_argument("--detail", default=None, help="Path to write detailed report text")
    args = parser.parse_args()

    if args.all:
        categories = TARGET_CATEGORIES
    elif args.categories:
        categories = args.categories
    else:
        parser.error("Specify --all or --categories")

    model_dir = Path(args.model_dir)

    from src.eval.evaluator_BFCL import Evaluator
    from src.eval.evaluator_hummer import HammerBenchEvaluator

    all_reports = []
    bfcl_scores = []
    hb_scores = []

    for cat in categories:
        inference_path = model_dir / f"Inference_retrieval_{cat}.json"
        if not inference_path.exists():
            print(f"[WARN] {inference_path} not found, skipping {cat}")
            continue

        results = load_jsonl(inference_path)
        is_hammerbench = cat.startswith("hammerbench")

        if is_hammerbench:
            hb_scope = category_scope(cat)
            hb_data_dir = get_scope_config(hb_scope)["test_data_dir"]
            evaluator = HammerBenchEvaluator(
                test_category=cat, data_dir=hb_data_dir,
            )
            scores = evaluator.evaluate(results)
            eval_path = model_dir / f"Eval_retrieval_{cat}.json"
            save_jsonl(scores, eval_path)

            report = HammerBenchEvaluator.format_report(scores, cat)
            print(report)
            all_reports.append(report)
            hb_scores.extend(scores)

        else:
            evaluator = Evaluator(test_category=cat)
            scores = evaluator.evaluate(results)
            eval_path = model_dir / f"Eval_retrieval_{cat}.json"
            save_jsonl(scores, eval_path)

            report = Evaluator.format_report(scores, cat)
            print(report)
            all_reports.append(report)
            bfcl_scores.extend(scores)

    # Per-benchmark JSON + unified overview (avoid mixing BFCL vs HammerBench in one flat file)
    bfcl_stats = None
    hb_stats = None

    if bfcl_scores:
        print("\n" + "=" * 60)
        print("  BFCL SINGLE-TURN OVERALL SUMMARY")
        overall_report = Evaluator.format_report(bfcl_scores, "ALL_BFCL_SINGLE_TURN")
        print(overall_report)
        all_reports.append(overall_report)

        bfcl_stats = Evaluator.compute_stats(bfcl_scores)
        bfcl_path = model_dir / "eval_summary_bfcl.json"
        with open(bfcl_path, "w") as f:
            json.dump(bfcl_stats, f, indent=2)
        print(f"  BFCL summary stats → {bfcl_path}")

    if hb_scores:
        print("\n" + "=" * 60)
        print("  HAMMERBENCH OVERALL SUMMARY")
        hb_report = HammerBenchEvaluator.format_report(hb_scores, "ALL_HAMMERBENCH")
        print(hb_report)
        all_reports.append(hb_report)

        hb_stats = HammerBenchEvaluator.compute_stats(hb_scores)
        hb_path = model_dir / "eval_summary_hammerbench.json"
        with open(hb_path, "w") as f:
            json.dump(hb_stats, f, indent=2)
        print(f"  HammerBench summary stats → {hb_path}")

    if bfcl_stats is not None or hb_stats is not None:
        overview = {
            "bfcl": bfcl_stats,
            "hammerbench": hb_stats,
        }
        overview_path = model_dir / "eval_summary.json"
        with open(overview_path, "w") as f:
            json.dump(overview, f, indent=2)
        print(f"  Combined overview (bfcl + hammerbench) → {overview_path}")

    # save detailed report
    if args.detail:
        detail_path = Path(args.detail)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with open(detail_path, "w") as f:
            f.write("\n".join(all_reports))
        print(f"  Detail report → {detail_path}")


if __name__ == "__main__":
    main()
