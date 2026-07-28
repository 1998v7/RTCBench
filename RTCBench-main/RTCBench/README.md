# Benchmark (Inference & Evaluation)

Inference and evaluation scripts for the **retrieval-augmented tool calling** benchmark. Test data and tool banks live in the parent directory under `test_data/`.

## Requirements

- Python 3.10+
- CUDA GPU for local vLLM inference (API-only or eval-only runs do not need one)
- `bash`

### Python packages

```text
torch
numpy
tqdm
openai
sentence-transformers
rank-bm25
jieba
rouge-score
```

Add `vllm` for local inference. For BFCL categories, also install the official `bfcl_eval` package.

### BFCL setup

BFCL AST checks rely on the official `bfcl_eval`. Default layout:

```text
<workspace>/
├── RTCBench/                               # this repo
└── gorilla/
    └── berkeley-function-call-leaderboard/ # contains bfcl_eval
```

Clone [berkeley-function-call-leaderboard](https://github.com/ShishirPatil/gorilla) and run `pip install -e .` inside it. Verify:

```bash
python -c "import bfcl_eval; print('ok')"
```

Set `BFCL_ROOT` if your BFCL repo lives elsewhere.

## Data

Paths are resolved relative to the repo root. Defaults:


| Content           | Default location (env var override)      |
| ----------------- | ---------------------------------------- |
| level-1           | `test_data/BFCL/` (`TEST_DATA_DIR`)      |
| level-1 tool bank | `test_data/BFCL/tool_bank/`              |
| level-2           | `test_data/HAMMERBENCH_ZH_500/`          |
| level-2 tool bank | `test_data/HAMMERBENCH_ZH_500/tool_bank` |


Other override vars (e.g. `POSSIBLE_ANSWER_DIR`) are listed in `src/config.py`.

## Model configuration (`run.sh`)

`run.sh` defines two model maps:

1. `local_models` — alias → local checkpoint path (vLLM backend).
2. `api_models` — alias → OpenRouter (OpenAI-compatible) model ID.

Edit them for your environment. **Never commit API keys**; pass them via `OPENROUTER_API_KEY`. The backend (vLLM vs HTTP) is selected automatically from `MODEL_INFO`.

## Quick start

From `benchmark/`:

```bash
bash run_task.sh <task> <model> <top_k> <variant> [EXTRA_ENV=value ...]
```


| Arg       | Values                                                          |
| --------- | --------------------------------------------------------------- |
| `task`    | `level_1` | `level_2`                                           |
| `model`   | An alias defined in `run.sh` (e.g. `openrouter_qwen3-32B`)    |
| `top_k`   | Number of retrieved tools                                       |
| `variant` | `none` (no initial tools) or `distractor` (random non-GT tools) |


Example:

```bash
SKIP_JUDGE=1 RESULTS_DIR=test_output bash run_task.sh level_1 openrouter_qwen3-32B 10 none
```

### Pipeline stages (`run.sh`)

1. **Inference** → `RESULTS_DIR/<variant>/<MODEL_INFO>/Inference_retrieval_<category>.json`. Skipped if outputs already exist.
2. **Rule eval** → `Eval_retrieval_*.json`, `eval_summary_*.json`, `eval_detail.txt`.
3. **LLM judge** (optional, BFCL only) — re-checks tool-name mismatches. Disable with `SKIP_JUDGE=1`.

