"""Base prompt strategy: prompt construction and output parsing."""
from __future__ import annotations

import ast
import json
import re
from abc import ABC, abstractmethod


class BasePromptStrategy(ABC):
    """Abstract base for prompt construction and model-output parsing.

    The prompt strategy defines:
      - The system prompt (search_tool schema + instruction)
      - How retrieved tools are formatted in the conversation
      - How the model's text output is parsed into structured tool calls
    """

    # -- Prompt construction --------------------------------------------------

    @abstractmethod
    def build_system_prompt(self, initial_tools: list[dict] | None = None) -> str:
        ...

    @abstractmethod
    def get_search_tool_schema(self) -> dict:
        ...

    @abstractmethod
    def format_tool_results_message(self, retrieval_records: list[dict]) -> str:
        """Format search_tool results into a message string.

        Parameters
        ----------
        retrieval_records : list of {"query": str, "tools": list[dict]}
        """
        ...

    # -- Output parsing -------------------------------------------------------

    @staticmethod
    def _sanitize_value(val):
        """Make a parsed value JSON-serializable (handle Ellipsis, etc.)."""
        if val is ...:
            return "..."
        if isinstance(val, list):
            return [BasePromptStrategy._sanitize_value(v) for v in val]
        if isinstance(val, dict):
            return {k: BasePromptStrategy._sanitize_value(v) for k, v in val.items()}
        return val

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove ``<think>...</think>`` blocks (e.g. Qwen3 thinking mode)."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def parse_tool_calls(self, text: str) -> list[dict]:
        """Parse ``[func_name(param=val, ...)]`` from model output.

        Returns a list of ``{func_name: {param: value}}`` dicts.
        """
        text = self._strip_thinking(text)
        text = text.strip().strip("`\n ")
        # Find the first '[' that starts a tool call block, skipping
        # any preamble text the model may have emitted after </think>
        bracket_idx = text.find("[")
        if bracket_idx > 0:
            text = text[bracket_idx:]
        elif bracket_idx < 0:
            text = "[" + text
        if not text.endswith("]"):
            text = text + "]"
        try:
            parsed = ast.parse(text, mode="eval")
            result: list[dict] = []
            if isinstance(parsed.body, ast.Call):
                calls = [parsed.body]
            elif hasattr(parsed.body, "elts"):
                calls = parsed.body.elts
            elif isinstance(parsed.body, ast.Subscript):
                candidates = []
                if isinstance(parsed.body.value, ast.Call):
                    candidates.append(parsed.body.value)
                sl = parsed.body.slice
                if isinstance(sl, ast.Call):
                    candidates.append(sl)
                elif hasattr(sl, "elts"):
                    candidates.extend(sl.elts)
                calls = candidates if candidates else []
            else:
                calls = []
            for node in calls:
                if not isinstance(node, ast.Call):
                    continue
                parts: list[str] = []
                n = node.func
                while isinstance(n, ast.Attribute):
                    parts.append(n.attr)
                    n = n.value
                if isinstance(n, ast.Name):
                    parts.append(n.id)
                name = ".".join(reversed(parts))
                args: dict = {}
                for kw in node.keywords:
                    if isinstance(kw.value, ast.Constant):
                        val = kw.value.value
                        args[kw.arg] = "..." if val is ... else val
                    else:
                        try:
                            val = ast.literal_eval(kw.value)
                            args[kw.arg] = self._sanitize_value(val)
                        except (ValueError, TypeError):
                            args[kw.arg] = ast.unparse(kw.value)
                result.append({name: args})
            return result
        except SyntaxError:
            return []

    def parse_search_calls(self, text: str) -> list[str] | None:
        """Extract ``search_tool(query='...')`` queries from model output.

        Returns a list of query strings, or *None* if no search calls found.
        """
        try:
            parsed_calls = self.parse_tool_calls(text)
            queries: list[str] = []
            for call in parsed_calls:
                if "search_tool" in call:
                    q = call["search_tool"].get("query")
                    if q is not None and str(q).strip():
                        queries.append(str(q).strip())
            return queries if queries else None
        except Exception:
            return None

    # -- Decode helpers for evaluation ----------------------------------------

    def decode_for_eval(self, model_responses: list[list[dict]]) -> list[dict]:
        """Extract final tool calls (excl. search_tool) for AST evaluation."""
        for step_responses in reversed(model_responses):
            non_search = [c for c in step_responses if "search_tool" not in c]
            if not non_search:
                continue
            result: list[dict] = []
            for call in non_search:
                parsed: dict = {}
                for name, args in call.items():
                    try:
                        args = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        args = {}
                    parsed[name] = args
                result.append(parsed)
            return result
        return []
