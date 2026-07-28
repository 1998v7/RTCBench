"""Inference orchestrator: retrieval-augmented tool-call flow.

Flow (single-turn, strict two-step):
  1. System prompt (search_tool schema + instructions) + user query
  2. Model outputs [search_tool(query="...")] → retriever finds tools
     - If model does NOT call search_tool → record failure, stop.
  3. Retrieved tool definitions appended to conversation
  4. Model outputs [func_name(param=value, ...)] → done

Supports both sequential (run_single_turn) and batched (run_batch) execution.
"""
from __future__ import annotations

from .retrieval.base import BaseRetriever
from .handler.base import BaseHandler
from .prompt.base import BasePromptStrategy


class InferenceOrchestrator:
    """Handler-agnostic retrieval-augmented inference.

    Parameters
    ----------
    handler : model backend (OpenAI API, vLLM, ...).
    prompt_strategy : prompt construction and output parsing.
    retriever : tool retrieval backend (BM25, Dense, Hybrid).
    variant : "none" (no initial tools), "distractor" (random non-GT tools),
               or "with_gt" (GT tools provided — model may call directly).
    """

    def __init__(
        self,
        handler: BaseHandler,
        prompt_strategy: BasePromptStrategy,
        retriever: BaseRetriever,
        variant: str = "none",
        tool_role: str = "user",
    ):
        self.handler = handler
        self.prompt_strategy = prompt_strategy
        self.retriever = retriever
        self.variant = variant
        self.tool_role = tool_role

    def _get_initial_tools(self, test_entry: dict) -> list[dict] | None:
        if self.variant == "with_gt":
            return test_entry.get("function") or None
        if self.variant == "distractor":
            return test_entry.get("distractor_tools") or None
        return None

    # ── Batched two-step inference ────────────────────────────────────────

    def run_batch(self, entries: list[dict]) -> list[dict]:
        """Run batched two-step inference for a list of entries.

        Step 1: batch all first-turn messages → model outputs (search_tool calls)
        Retrieval: run retriever for each search query (CPU, parallel-friendly)
        Step 2: batch all second-turn messages → model outputs (actual tool calls)
        """
        n = len(entries)

        # -- Build Step 1 messages --
        step1_messages = []
        for entry in entries:
            initial_tools = self._get_initial_tools(entry)
            user_content = entry["question"][0][0]["content"]
            msgs = [
                {"role": "system", "content": self.prompt_strategy.build_system_prompt(initial_tools=initial_tools)},
                {"role": "user", "content": user_content},
            ]
            step1_messages.append(msgs)

        # -- Batch Step 1 --
        step1_outputs = self.handler.batch_chat(step1_messages)

        # -- Process Step 1 results & run retrieval --
        needs_step2: list[int] = []
        step2_messages: list[list[dict]] = []

        per_entry_state: list[dict] = []
        for i in range(n):
            content, usage = step1_outputs[i]
            state = {
                "raw_contents": [content],
                "metadata": {
                    "input_token": usage.get("input_token", 0),
                    "output_token": usage.get("output_token", 0),
                },
                "all_responses": [],
                "search_tools_results": [],
            }

            queries = self.prompt_strategy.parse_search_calls(content)

            if not queries:
                if self.variant in ("with_gt", "distractor"):
                    tool_calls = self.prompt_strategy.parse_tool_calls(content)
                    tool_calls = [c for c in tool_calls if "search_tool" not in c]
                    state["all_responses"].append(tool_calls if tool_calls else [])
                else:
                    state["all_responses"].append([])
            else:
                retrieval_records: list[dict] = []
                for q in queries:
                    state["all_responses"].append([{"search_tool": {"query": q}}])
                    retrieved = self.retriever.retrieve(q)
                    state["search_tools_results"].append({
                        "query": q,
                        "retrieved": retrieved,
                    })
                    retrieval_records.append({"query": q, "tools": retrieved})

                tool_msg = self.prompt_strategy.format_tool_results_message(retrieval_records)
                msgs = list(step1_messages[i])
                msgs.append({"role": "assistant", "content": content})
                msgs.append({"role": self.tool_role, "content": tool_msg})

                needs_step2.append(i)
                step2_messages.append(msgs)

            per_entry_state.append(state)

        # -- Batch Step 2 --
        if step2_messages:
            step2_outputs = self.handler.batch_chat(step2_messages)

            for idx_in_batch, i in enumerate(needs_step2):
                content2, usage2 = step2_outputs[idx_in_batch]
                state = per_entry_state[i]
                state["raw_contents"].append(content2)
                state["metadata"]["input_token"] += usage2.get("input_token", 0)
                state["metadata"]["output_token"] += usage2.get("output_token", 0)

                tool_calls = self.prompt_strategy.parse_tool_calls(content2)
                tool_calls = [c for c in tool_calls if "search_tool" not in c]
                state["all_responses"].append(tool_calls if tool_calls else [])

        # -- Assemble results --
        results = []
        for i, entry in enumerate(entries):
            state = per_entry_state[i]
            decoded = self.prompt_strategy.decode_for_eval(state["all_responses"])
            state["metadata"]["raw_model_outputs"] = state["raw_contents"]
            state["metadata"]["variant"] = self.variant
            results.append({
                "id": entry["id"],
                "model_responses": state["all_responses"],
                "model_responses_decoded": decoded,
                "search_tools_results": state["search_tools_results"],
                "metadata": state["metadata"],
            })

        return results

    # ── Sequential (for API / concurrent mode) ─────────────────────────

    def run_single_turn(self, test_entry: dict) -> dict:
        """Run single-turn inference for one entry (step1 → retrieve → step2)."""
        initial_tools = self._get_initial_tools(test_entry)
        user_content = test_entry["question"][0][0]["content"]
        messages = [
            {"role": "system", "content": self.prompt_strategy.build_system_prompt(initial_tools=initial_tools)},
            {"role": "user", "content": user_content},
        ]

        all_responses: list = []
        search_tools_results: list = []
        raw_contents: list[str] = []
        metadata: dict = {"input_token": 0, "output_token": 0}

        # Step 1: expect search_tool call
        content, usage = self.handler.chat(messages)
        raw_contents.append(content)
        metadata["input_token"] += usage.get("input_token", 0)
        metadata["output_token"] += usage.get("output_token", 0)

        queries = self.prompt_strategy.parse_search_calls(content)

        if not queries:
            if self.variant in ("with_gt", "distractor"):
                tool_calls = self.prompt_strategy.parse_tool_calls(content)
                tool_calls = [c for c in tool_calls if "search_tool" not in c]
                all_responses.append(tool_calls if tool_calls else [])
            else:
                all_responses.append([])
        else:
            retrieval_records: list[dict] = []
            for q in queries:
                all_responses.append([{"search_tool": {"query": q}}])
                retrieved = self.retriever.retrieve(q)
                search_tools_results.append({"query": q, "retrieved": retrieved})
                retrieval_records.append({"query": q, "tools": retrieved})

            tool_msg = self.prompt_strategy.format_tool_results_message(retrieval_records)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": self.tool_role, "content": tool_msg})

            # Step 2: expect actual function call
            content2, usage2 = self.handler.chat(messages)
            raw_contents.append(content2)
            metadata["input_token"] += usage2.get("input_token", 0)
            metadata["output_token"] += usage2.get("output_token", 0)

            tool_calls = self.prompt_strategy.parse_tool_calls(content2)
            tool_calls = [c for c in tool_calls if "search_tool" not in c]
            all_responses.append(tool_calls if tool_calls else [])

        decoded = self.prompt_strategy.decode_for_eval(all_responses)
        metadata["raw_model_outputs"] = raw_contents
        metadata["variant"] = self.variant
        return {
            "id": test_entry["id"],
            "model_responses": all_responses,
            "model_responses_decoded": decoded,
            "search_tools_results": search_tools_results,
            "metadata": metadata,
        }
