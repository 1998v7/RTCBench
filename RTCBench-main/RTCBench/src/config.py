"""Configuration for BFCL-Retrieval benchmark."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]        # benchmark/
REPO_ROOT = PROJECT_ROOT.parent                           # BFCL_retrieval2/

# ---------------------------------------------------------------------------
# BFCL source (gorilla repo, needed by evaluator for checker functions)
# ---------------------------------------------------------------------------
BFCL_ROOT = Path(
    os.getenv("BFCL_ROOT", str(REPO_ROOT.parent / "gorilla" / "berkeley-function-call-leaderboard"))
)

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------
NON_LIVE_CATEGORIES = [
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
]

LIVE_CATEGORIES = [
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
]

HAMMERBENCH_CATEGORIES = [
    "hammerbench",
]

BFCL_ALL_CATEGORIES = NON_LIVE_CATEGORIES + LIVE_CATEGORIES

TARGET_CATEGORIES = NON_LIVE_CATEGORIES  # backward compat default

# ---------------------------------------------------------------------------
# Data paths (resolve from BFCL_retrieval2/)
# ---------------------------------------------------------------------------
# Data root: REPO_ROOT/test_data/  (BFCL, HammerBench, …)
_data_root = REPO_ROOT / "test_data"

# BFCL: non_live.json, live.json, merged.json under test_data/BFCL/tool_bank/
_tb_env = os.getenv("TOOL_BANK_DIR")
_default_bfcl_tool_bank = _data_root / "BFCL" / "tool_bank"
TOOL_BANK_DIR = (Path(_tb_env) if _tb_env else _default_bfcl_tool_bank).resolve()

_td_env = os.getenv("TEST_DATA_DIR")
TEST_DATA_DIR = (Path(_td_env) if _td_env else _data_root / "BFCL").resolve()

_pa_env = os.getenv("POSSIBLE_ANSWER_DIR")
POSSIBLE_ANSWER_DIR = (Path(_pa_env) if _pa_env else TEST_DATA_DIR / "possible_answer").resolve()

# HammerBench data (zh / en)
HAMMERBENCH_ZH_DIR = (_data_root / "HAMMERBENCH_ZH_500").resolve()
HAMMERBENCH_EN_DIR = (_data_root / "HAMMERBENCH_EN_500").resolve()

SCOPE_CONFIG = {
    "non_live": {
        "categories": NON_LIVE_CATEGORIES,
        "tool_bank": TOOL_BANK_DIR / "non_live.json",
        "test_data_dir": TEST_DATA_DIR,
    },
    "live": {
        "categories": LIVE_CATEGORIES,
        "tool_bank": TOOL_BANK_DIR / "live.json",
        "test_data_dir": TEST_DATA_DIR,
    },
    # Merged non-live + live corpus (same directory as split banks)
    "bfcl_merged": {
        "categories": BFCL_ALL_CATEGORIES,
        "tool_bank": TOOL_BANK_DIR / "merged.json",
        "test_data_dir": TEST_DATA_DIR,
    },
    "hammerbench": {
        "categories": HAMMERBENCH_CATEGORIES,
        "tool_bank": HAMMERBENCH_ZH_DIR / "tool_bank" / "merged.json",
        "test_data_dir": HAMMERBENCH_ZH_DIR,
    },
    "hammerbench_en": {
        "categories": HAMMERBENCH_CATEGORIES,
        "tool_bank": HAMMERBENCH_EN_DIR / "tool_bank" / "merged.json",
        "test_data_dir": HAMMERBENCH_EN_DIR,
    },
}

TOOL_BANK_PATH = SCOPE_CONFIG["non_live"]["tool_bank"]


def get_scope_config(scope: str) -> dict:
    if scope not in SCOPE_CONFIG:
        raise ValueError(f"Unknown scope: {scope!r}. Must be one of {list(SCOPE_CONFIG)}")
    return SCOPE_CONFIG[scope]


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def category_scope(cat: str) -> str:
    """Return the scope for a single category name."""
    if cat.startswith("hammerbench"):
        return os.getenv("HAMMERBENCH_SCOPE", "hammerbench")
    if _truthy_env("BFCL_MERGED_TOOL_BANK") and (
        cat in NON_LIVE_CATEGORIES or cat in LIVE_CATEGORIES
    ):
        return "bfcl_merged"
    if cat.startswith("live_"):
        return "live"
    return "non_live"


def category_variant(cat: str) -> str | None:
    """Return fixed variant for a category, or None to use global default."""
    return None


def group_by_scope(categories: list[str]) -> dict[str, list[str]]:
    """Group categories by scope. Returns {scope: [categories]}."""
    groups: dict[str, list[str]] = {}
    for cat in categories:
        s = category_scope(cat)
        groups.setdefault(s, []).append(cat)
    return groups

# ---------------------------------------------------------------------------
# Retrieval defaults
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K = 10
EMBEDDING_CACHE_DIR = PROJECT_ROOT / "cache" / "embeddings"

# ---------------------------------------------------------------------------
# Type permissiveness ranking (higher = more permissive)
# ---------------------------------------------------------------------------
TYPE_PERMISSIVENESS = {
    "integer": 1,
    "int": 1,
    "float": 2,
    "double": 2,
    "number": 3,
    "string": 0,
    "boolean": 0,
    "array": 0,
    "object": 0,
    "dict": 0,
}
