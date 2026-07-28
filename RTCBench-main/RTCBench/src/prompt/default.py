"""Default prompt strategy: two-phase retrieval flow with [func(param=val)] format."""
from __future__ import annotations

import json

from .base import BasePromptStrategy
from . import prompt_List
from .prompt_List import SEARCH_TOOL_SCHEMA



class DefaultPromptStrategy(BasePromptStrategy):
    """Default two-phase retrieval prompt with ``[func(param=val)]`` call format.

    Parameters
    ----------
    prompt_name : Name of a variable in ``prompt_List.py`` to use as template.
        E.g. "base_prompt".
    """

    def __init__(self, prompt_name: str = "base_prompt"):
        template = getattr(prompt_List, prompt_name, None)
        if template is None:
            available = [k for k in dir(prompt_List) if not k.startswith("_")]
            raise ValueError(f"Unknown prompt '{prompt_name}'. Available: {available}")
        self.template = template
        self.prompt_name = prompt_name

    def build_system_prompt(self, initial_tools: list[dict] | None = None) -> str:
        tools_list = list(initial_tools) if initial_tools else []
        tools_text = json.dumps(tools_list, indent=2, ensure_ascii=False) if tools_list else "None"
        search_tool_schema = json.dumps(SEARCH_TOOL_SCHEMA, indent=2, ensure_ascii=False)
        return self.template.format(
            search_tool_schema=search_tool_schema,
            tools_text=tools_text,
        )

    def get_search_tool_schema(self) -> dict:
        return SEARCH_TOOL_SCHEMA

    def format_tool_results_message(self, retrieval_records: list[dict]) -> str:
        payload = [
            {"query": rec["query"], "retrieved_tools": rec["tools"]}
            for rec in retrieval_records
        ]
        return "[search_tool results]\n" + json.dumps(payload, indent=2, ensure_ascii=False)
