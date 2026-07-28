"""vLLM local model handler using the offline LLM.chat() API."""
from __future__ import annotations

from .base import BaseHandler


class VLLMHandler(BaseHandler):
    """Handler for local models via vLLM offline inference.

    Parameters
    ----------
    model_name : HuggingFace model path or id.
    max_model_len : maximum context length.
    tensor_parallel_size : number of GPUs for tensor parallelism.
    gpu_memory_utilization : fraction of GPU memory to use.
    temperature, max_tokens : generation parameters.
    """

    def __init__(
        self,
        model_name: str,
        max_model_len: int = 20480,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.6,
        temperature: float = 0.0,
        max_tokens: int | None = 4096,
        **kwargs,
    ):
        super().__init__(model_name, temperature, max_tokens)
        from vllm import LLM, SamplingParams

        self._llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            disable_log_stats=True,
        )
        self._sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=42,
        )

    def chat(self, messages: list[dict]) -> tuple[str, dict]:
        results = self.batch_chat([messages])
        return results[0]

    def batch_chat(self, batch_messages: list[list[dict]]) -> list[tuple[str, dict]]:
        if not batch_messages:
            return []
        outputs = self._llm.chat(
            messages=batch_messages,
            sampling_params=self._sampling,
            use_tqdm=False,
        )
        results = []
        for req_out in outputs:
            out = req_out.outputs[0]
            content = out.text.strip()
            usage = {
                "input_token": len(req_out.prompt_token_ids),
                "output_token": len(out.token_ids),
            }
            results.append((content, usage))
        return results

    def shutdown(self) -> None:
        del self._llm
