"""Tool bank indexing: convert tool definitions to searchable text."""
from __future__ import annotations

import json
from pathlib import Path


def tool_to_doc(tool: dict) -> str:
    """Convert a tool definition to searchable text for retrieval."""
    parts = [
        tool.get("name", ""),
        tool.get("description", ""),
    ]
    params = tool.get("parameters", {}).get("properties", {})
    for name, meta in params.items():
        parts.append(name)
        parts.append(meta.get("description", ""))
    return " ".join(str(p) for p in parts if p)


def load_tool_bank(path: Path) -> tuple[list[dict], list[str]]:
    """Load tool bank and generate doc texts.

    Returns (tools, doc_texts) where doc_texts[i] is the searchable text
    for tools[i].
    """
    with open(path) as f:
        data = json.load(f)
    tools = data["tools"]
    doc_texts = [tool_to_doc(t) for t in tools]
    return tools, doc_texts
