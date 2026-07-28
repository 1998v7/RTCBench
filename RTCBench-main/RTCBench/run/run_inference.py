#!/usr/bin/env python3
"""BFCL-Retrieval inference entry point.

Usage examples:
  # Local model via vLLM
  python run_inference.py --model /path/to/model --use-vllm --all

  # OpenAI-compatible API
  python run_inference.py --model gpt-4o-mini --base-url https://api.openai.com/v1 --all

  # Single category with specific retriever
  python run_inference.py --model /path/to/model --use-vllm \
      --test-category simple_python --retriever hybrid --top-k 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    NON_LIVE_CATEGORIES, LIVE_CATEGORIES,
    HAMMERBENCH_CATEGORIES,
    group_by_scope, get_scope_config, category_variant,
)
from src.retrieval.indexer import load_tool_bank


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


def build_retriever(args, tools, doc_texts):
    """Construct retriever based on CLI args."""
    if args.retriever == "bm25":
        from src.retrieval import BM25Retriever
        return BM25Retriever(tools=tools, doc_texts=doc_texts, top_k=args.top_k)

    elif args.retriever == "dense":
        from src.retrieval import DenseRetriever
        return DenseRetriever(
            tools=tools, doc_texts=doc_texts,
            top_k=args.top_k, model_name=args.retriever_model,
        )

    elif args.retriever == "hybrid":
        from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
        bm25 = BM25Retriever(tools=tools, doc_texts=doc_texts, top_k=args.top_k)
        dense = DenseRetriever(
            tools=tools, doc_texts=doc_texts,
            top_k=args.top_k, model_name=args.retriever_model,
        )
        return HybridRetriever(bm25=bm25, dense=dense, top_k=args.top_k)

    else:
        raise ValueError(f"Unknown retriever: {args.retriever}")


def build_handler(args):
    """Construct model handler based on CLI args."""
    if args.use_vllm:
        from src.handler import VLLMHandler
        return VLLMHandler(
            model_name=args.model,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel,
            gpu_memory_utilization=args.gpu_memory_utilization,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    else:
        from src.handler import OpenAIHandler
        return OpenAIHandler(
            model_name=args.model,
            base_url=args.base_url or None,
            api_key=args.api_key or None,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )


def main():
    parser = argparse.ArgumentParser(description="BFCL-Retrieval inference")

    # model
    parser.add_argument("--model", required=True, help="Model path or name")
    parser.add_argument("--model-info", default=None, help="Model info string for output directory (default: model basename)")
    parser.add_argument("--use-vllm", action="store_true", help="Use vLLM local backend")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="", help="API key")

    # vLLM options
    parser.add_argument("--max-model-len", type=int, default=20480)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)

    # generation
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)

    # retrieval
    parser.add_argument("--retriever", default="bm25", choices=["bm25", "dense", "hybrid"])
    parser.add_argument("--retriever-model", default="ToolBench/ToolBench_IR_bert_based_uncased",  help="Sentence-transformer model for dense retriever")
    parser.add_argument("--top-k", type=int, default=10)


    # variant
    parser.add_argument("--variant", default="none", choices=["none", "distractor", "with_gt"], help="'none': no initial tools; 'distractor': random non-GT tools; 'with_gt': GT tools provided")

    # prompt
    parser.add_argument("--prompt", default=None, help="Prompt template name (base/think/...) or auto-resolve from model name")

    # categories
    parser.add_argument("--all", action="store_true", help="Run all categories")
    parser.add_argument("--test-category", default=None, help="Single category to run")

    # output
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--limit", type=int, default=None, help="Max entries per category (for testing)")
    parser.add_argument("--batch-size", type=int, default=256, help="Max entries per batch to avoid OOM (vLLM mode)")
    parser.add_argument("--concurrency", type=int, default=16, help="Max concurrent API requests (API mode)")

    args = parser.parse_args()

    # resolve categories
    if args.all:
        categories = NON_LIVE_CATEGORIES + LIVE_CATEGORIES + HAMMERBENCH_CATEGORIES
    elif args.test_category:
        categories = args.test_category.split(",")
    else:
        parser.error("Specify --all or --test-category")

    model_info = args.model_info or Path(args.model).name
    output_base = Path(args.output_dir) / model_info
    output_base.mkdir(parents=True, exist_ok=True)

    # group categories by scope (each scope has its own tool bank)
    scope_groups = group_by_scope(categories)

    print(f"Building handler for {args.model}...")
    handler = build_handler(args)

    from src.prompt import DefaultPromptStrategy
    from src.orchestrator import InferenceOrchestrator

    # collect all tasks across scopes
    all_tasks: list[tuple[str, list[dict]]] = []

    for scope, scope_cats in scope_groups.items():
        scope_cfg = get_scope_config(scope)
        tool_bank_path = scope_cfg["tool_bank"]
        test_data_dir = scope_cfg["test_data_dir"]

        print(f"\n[{scope}] Loading tool bank from {tool_bank_path}...")
        tools, doc_texts = load_tool_bank(tool_bank_path)
        print(f"  {len(tools)} tools loaded")

        print(f"[{scope}] Building {args.retriever} retriever (top_k={args.top_k})...")
        retriever = build_retriever(args, tools, doc_texts)

        prompt_name = args.prompt or model_info
        prompt_strategy = DefaultPromptStrategy(prompt_name=prompt_name)
        tool_role = "tool" if args.use_vllm else "user"

        orch_cache: dict[str, InferenceOrchestrator] = {}

        for cat in scope_cats:
            fname = f"{cat}.json" if cat.startswith("hammerbench") else f"BFCL_v4_retrieval_{cat}.json"
            data_path = test_data_dir / fname
            if not data_path.exists():
                print(f"[WARN] {data_path} not found, skipping {cat}")
                continue

            cat_variant = category_variant(cat) or args.variant
            if cat_variant not in orch_cache:
                orch_cache[cat_variant] = InferenceOrchestrator(
                    handler=handler,
                    prompt_strategy=prompt_strategy,
                    retriever=retriever,
                    variant=cat_variant,
                    tool_role=tool_role,
                )
            orchestrator = orch_cache[cat_variant]

            entries = load_jsonl(data_path)
            if args.limit:
                entries = entries[:args.limit]
            all_tasks.append((cat, entries, orchestrator))

    total_entries = sum(len(entries) for _, entries, _ in all_tasks)
    print(f"\nVariant: {args.variant}")
    print(f"Total: {total_entries} entries across {len(all_tasks)} categories\n")

    pbar = tqdm(total=total_entries, desc="Inference", unit="entry")

    use_batch = args.use_vllm  # vLLM: batch mode; API: concurrent mode

    if not use_batch:
        from concurrent.futures import ThreadPoolExecutor, as_completed

    for cat, entries, orch in all_tasks:
        out_path = output_base / f"Inference_retrieval_{cat}.json"

        # Resume: load existing partial results and skip already-done entries
        done_ids: set[str] = set()
        existing_results: list[dict] = []
        if out_path.exists():
            existing_results = load_jsonl(out_path)
            done_ids = {r["id"] for r in existing_results}
            if len(done_ids) >= len(entries):
                pbar.update(len(entries))
                pbar.set_postfix(category=cat, status="skipped")
                continue
            pbar.update(len(done_ids))

        remaining = [e for e in entries if e["id"] not in done_ids]
        pbar.set_postfix(category=cat)

        if use_batch:
            bs = args.batch_size
            for start in range(0, len(remaining), bs):
                chunk = remaining[start : start + bs]
                batch_results = orch.run_batch(chunk)
                existing_results.extend(batch_results)
                save_jsonl(existing_results, out_path)
                pbar.update(len(chunk))
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                future_to_idx = {
                    pool.submit(orch.run_single_turn, entry): i
                    for i, entry in enumerate(remaining)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    result = future.result()
                    existing_results.append(result)
                    save_jsonl(existing_results, out_path)
                    pbar.update(1)

        n_search = sum(len(r["search_tools_results"]) for r in existing_results)
        n_calls = sum(len(r["model_responses_decoded"]) for r in existing_results)
        pbar.set_postfix(category=cat, search=n_search, calls=n_calls)

    pbar.close()
    handler.shutdown()
    print("\nInference complete.")


if __name__ == "__main__":
    main()
