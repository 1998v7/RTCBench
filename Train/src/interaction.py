"""VERL BaseInteraction for retrieval-augmented tool calling.

Flow per trajectory:
  Turn 1: model outputs [search_tool(query="...")] or [func_call(...)]
    - If search call (and haven't searched yet):
        execute dense retrieval → return results as user message → continue
    - Otherwise (final call, or second attempt):
        compute 3-component reward → terminate

Reward components:
  format:         <think>...</think> missing → OUTCOME_MIN (-3)
  search_reward:
      with_gt (tools given):   not searched → +1, searched → -1
      none    (need search):   not searched → -1,
          searched → 1.0 * (0.5 + 0.5 * recall)  [0.5 ~ 1.0]
  outcome:        fine-grained tool call matching → [-3, 3]
  honest reject:  searched + recall=0 + no func call → 0.0 (neutral)

Environment variables:
  NON_LIVE_TOOL_BANK_PATH
  LIVE_TOOL_BANK_PATH
  RETRIEVAL_MODEL_PATH
  RETRIEVAL_TOP_K          (default: 10)
  EMBEDDING_CACHE_DIR      (default: /tmp/bfcl_emb_cache)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np

from transformers import AutoTokenizer

from verl.interactions.base import BaseInteraction
from reward import (
    _parse_tool_calls, _compute_tool_call_score,
    OUTCOME_MAX, OUTCOME_MIN, FORMAT_REWARD,
)

# ---------------------------------------------------------------------------
# Lazy-loaded tokenizer for token-based truncation
# ---------------------------------------------------------------------------

_tokenizer_cache: dict[str, AutoTokenizer] = {}
_tokenizer_lock = threading.Lock()


def _get_tokenizer() -> AutoTokenizer:
    model_path = os.environ.get("MODEL_PATH")
    with _tokenizer_lock:
        if model_path not in _tokenizer_cache:
            _tokenizer_cache[model_path] = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True,
            )
        return _tokenizer_cache[model_path]


def _truncate_by_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """Truncate text to at most max_tokens. Returns (text, was_truncated)."""
    tokenizer = _get_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text, False
    truncated_text = tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=False)
    return truncated_text + "\n...(truncated)", True


# ---------------------------------------------------------------------------
# Dense retrieval index
# ---------------------------------------------------------------------------

def _tool_to_doc(tool: dict) -> str:
    parts = [tool.get("name", ""), tool.get("description", "")]
    for pname, pmeta in tool.get("parameters", {}).get("properties", {}).items():
        parts.extend([pname, pmeta.get("description", "")])
    return " ".join(str(p) for p in parts if p)


def _load_tool_bank(path: str) -> tuple[list[str], list[str]]:
    with open(path) as f:
        data = json.load(f)
    tools = data["tools"]
    return [t["name"] for t in tools], [_tool_to_doc(t) for t in tools]


class _DenseIndex:
    def __init__(self, tool_names, doc_texts, model_name, cache_dir):
        from sentence_transformers import SentenceTransformer
        self._tool_names = tool_names
        self._model = SentenceTransformer(model_name, device="cpu")
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        h = hashlib.md5(json.dumps(doc_texts, sort_keys=True).encode()).hexdigest()[:12]
        cache_path = cache_dir / f"emb_{model_name.replace('/', '_')}_{h}.npy"
        if cache_path.exists():
            embs = np.load(cache_path)
        else:
            embs = self._model.encode(doc_texts, show_progress_bar=True, batch_size=64)
            np.save(cache_path, embs)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        self._corpus_embs = embs / (norms + 1e-9)

    def retrieve(self, query: str, top_k: int) -> list[str]:
        q = self._model.encode([query], show_progress_bar=False)[0]
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = self._corpus_embs @ q
        return [self._tool_names[i] for i in scores.argsort()[::-1][:top_k]]


_indices: dict[str, _DenseIndex] = {}
_indices_lock = threading.Lock()

_TOOL_BANK_DIR = Path(__file__).resolve().parent / "tool_bank"
_DEFAULT_NON_LIVE = str(_TOOL_BANK_DIR / "non_live.json")
_DEFAULT_LIVE = str(_TOOL_BANK_DIR / "live.json")
_DEFAULT_MODEL = "ToolBench/ToolBench_IR_bert_based_uncased"


def _get_index(scope: str) -> _DenseIndex:
    if scope in _indices:
        return _indices[scope]
    with _indices_lock:
        if scope in _indices:
            return _indices[scope]
        model = os.environ.get("RETRIEVAL_MODEL_PATH", _DEFAULT_MODEL)
        cache = os.environ.get("EMBEDDING_CACHE_DIR", "/tmp/bfcl_emb_cache")
        if scope == "non_live":
            bank = os.environ.get("NON_LIVE_TOOL_BANK_PATH", _DEFAULT_NON_LIVE)
        elif scope == "live":
            bank = os.environ.get("LIVE_TOOL_BANK_PATH", _DEFAULT_LIVE)
        else:
            raise ValueError(f"Unknown scope: {scope!r}")
        names, docs = _load_tool_bank(bank)
        _indices[scope] = _DenseIndex(names, docs, model, cache)
        return _indices[scope]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_SEARCH_RE = re.compile(
    r'search_tool\(\s*query\s*=\s*["\'](.+?)["\'](?:\s*,\s*top_k\s*=\s*\d+)?\s*\)',
    flags=re.DOTALL,
)


def _strip_think(content: str) -> str:
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def _strip_tool_response(content: str) -> str:
    return re.sub(r"<tool_response>.*?</tool_response>", "", content, flags=re.DOTALL)


def _has_valid_format(content: str) -> bool:
    if not content:
        return False
    op = content.find("<think>")
    cl = content.find("</think>")
    return 0 <= op < cl


def _parse_search_queries(content: str) -> list[str]:
    return [m.group(1).strip() for m in _SEARCH_RE.finditer(_strip_think(content))]


def _extract_tool_calls_string(content: str) -> str | None:
    """Extract the last non-search bracket group from the response.

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


# ---------------------------------------------------------------------------
# Format retrieved tools into the message content injected back to model
# ---------------------------------------------------------------------------

_tool_bank_cache: dict[str, dict[str, dict]] = {}
_tool_bank_cache_lock = threading.Lock()


def _get_tool_bank(scope: str) -> dict[str, dict]:
    """Load and cache tool bank keyed by tool name."""
    if scope in _tool_bank_cache:
        return _tool_bank_cache[scope]
    with _tool_bank_cache_lock:
        if scope in _tool_bank_cache:
            return _tool_bank_cache[scope]
        path = os.environ.get(
            "NON_LIVE_TOOL_BANK_PATH" if scope == "non_live" else "LIVE_TOOL_BANK_PATH",
            _DEFAULT_NON_LIVE if scope == "non_live" else _DEFAULT_LIVE,
        )
        with open(path) as f:
            bank = {t["name"]: t for t in json.load(f)["tools"]}
        _tool_bank_cache[scope] = bank
        return bank


def _format_retrieved_tools(
    queries: list[str], results_per_query: list[list[str]], scope: str,
) -> str:
    """Build the tool result message shown to the model after search."""
    bank = _get_tool_bank(scope)
    entries = []
    for query, tool_names in zip(queries, results_per_query):
        retrieved = [bank[n] for n in tool_names if n in bank]
        entries.append({"query": query, "retrieved_tools": retrieved})
    payload = json.dumps(entries, ensure_ascii=False, indent=2)
    return f"[search_tool results]\n{payload}"


# ---------------------------------------------------------------------------
# Interaction
# ---------------------------------------------------------------------------

SEARCH_BASE = 1.0
RECALL_REWARD = 2.0
WITHGT_REWARD = 3.0
COUNT_PENALTY = 0.1
HONEST_REJECT_REWARD = 0.5


class SearchToolInteraction(BaseInteraction):
    """RL interaction for retrieval-augmented tool calling.

    Registered under name "search_tool" (via interaction_config.yaml).
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._state: dict[str, dict] = {}

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: str = "",
        variant: str = "none",
        gt_tools: list = None,
        scope: str = "non_live",
        trajectory_search_count: int | None = None,
        **kwargs,
    ) -> str:

        if instance_id is None:
            instance_id = str(uuid4())

        self._state[instance_id] = {
            "ground_truth": ground_truth,
            "variant": variant,
            "gt_tools": gt_tools or [],
            "scope": scope,
            "search_count": 0,
            "searched": False,
            "trajectory_search_count": trajectory_search_count,
            "actual_search_calls": 0,
        }

        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> tuple[bool, str, float, dict]:
    
        state = self._state[instance_id]

        last_asst = next((m["content"] for m in reversed(messages) if m.get("role") == "assistant"), "",)

        queries = _parse_search_queries(last_asst)
        top_k = int(os.environ.get("RETRIEVAL_TOP_K", "10"))

        # ── Search turn ───────────────────────────────────────────────────
        if queries and state["search_count"] < 1:
            state["search_count"] += 1
            state["searched"] = True
            state["actual_search_calls"] = len(queries)
            scope = state["scope"]
            index = _get_index(scope)
            results_per_query = [index.retrieve(q, top_k) for q in queries]
            retrieved_names: set[str] = set()
            for names in results_per_query:
                retrieved_names.update(names)
            result_msg = _format_retrieved_tools(queries, results_per_query, scope)
            max_tokens = int(os.environ.get("MAX_SEARCH_RESULT_LEN", "4096"))
            result_msg, truncated = _truncate_by_tokens(result_msg, max_tokens)
            state["search_truncated"] = truncated
            if truncated:
                visible_names = set(re.findall(r'"name"\s*:\s*"([^"]+)"', result_msg))
                state["retrieved_names"] = retrieved_names & visible_names
            else:
                state["retrieved_names"] = retrieved_names
            return False, result_msg, 0.0, {"role": "tool"}

        # ── Final call turn ───────────────────────────────────────────────
        return True, "", self._compute_reward(last_asst, state), {}

    def _recall(self, state: dict) -> float:
        """Compute recall of GT tools among retrieved tools."""
        ground_truth = state["ground_truth"]
        retrieved = state.get("retrieved_names", set())
        if not ground_truth or not retrieved:
            return 0.0
        gt_calls = _parse_tool_calls(ground_truth)
        if not gt_calls:
            return 0.0
        gt_names = set(list(c.keys())[0].lower() for c in gt_calls)
        retrieved_lower = set(n.lower() for n in retrieved)
        return len(gt_names & retrieved_lower) / len(gt_names)

    def _compute_reward(self, content: str, state: dict) -> float:
        """Reward structure:
          with_gt:  not searched → +3.0,  searched → -2.0
          none:     not searched → -2.0
                    searched → 1.0 (base)
                              + recall * 2.0        [0, 2.0]
                              - 0.1 if actual_calls > trajectory_search_count + 1
          honest reject (none, searched, recall=0, no func call) → 0.5
          total = format(1.0) + outcome([-3,3]) + search_reward
        """
        if not _has_valid_format(content):
            return OUTCOME_MIN

        variant = state["variant"]
        searched = state["searched"]
        ground_truth = state["ground_truth"]

        # ── search_reward ─────────────────────────────────────────────────
        recall = 0.0
        if variant == "with_gt":
            search_reward = WITHGT_REWARD if not searched else -2.0
        else:
            if not searched:
                search_reward = -2.0
            else:
                recall = self._recall(state)
                search_reward = SEARCH_BASE + RECALL_REWARD * recall
                expected = state.get("trajectory_search_count")
                actual = state.get("actual_search_calls", 0)
                if expected is not None and actual > expected + 1:
                    search_reward -= COUNT_PENALTY

        if search_reward < 0:
            return search_reward

        # ── outcome ───────────────────────────────────────────────────────
        pred_str = _extract_tool_calls_string(content)

        if not pred_str:
            if searched and recall == 0.0:
                return HONEST_REJECT_REWARD
            return FORMAT_REWARD + OUTCOME_MIN + search_reward

        outcome = OUTCOME_MIN
        if ground_truth:
            pred_calls = _parse_tool_calls(pred_str)
            gt_calls = _parse_tool_calls(ground_truth)
            if pred_calls is not None and gt_calls is not None:
                outcome = _compute_tool_call_score(pred_calls, gt_calls)

        return FORMAT_REWARD + outcome + search_reward

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        self._state.pop(instance_id, None)
