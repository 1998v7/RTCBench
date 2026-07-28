#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ── Model Configuration ─────────────────────────────────────────────────────

MODEL_INFO="${MODEL_INFO:-openrouter_qwen3.5-flash}"
PROMPT="${PROMPT:-thinking_prompt_zh}"

# Local: vLLM backend, value = path to HF model or checkpoint dir
declare -A local_models=(
  ["Qwen3-4B-Instruct-2507"]="your_local_path/Qwen3-4B-Instruct-2507/"

)
# Remote: OpenRouter API, value = API model ID (e.g. openai/gpt-5.4)
declare -A api_models=(
  ["openrouter_qwen3.5-flash"]="qwen/qwen3.5-flash-02-23"
  ["openrouter_o4-mini"]="openai/o4-mini"
  ["openrouter_qwen3-32B"]="qwen/qwen3-32b"
  ["openrouter_qwen3.5-122b-a10b"]="qwen/qwen3.5-122b-a10b"
  ["openrouter_qwen3.5-397b-a17b"]="qwen/qwen3.5-397b-a17b"
  ["openrouter_gpt-oss-120b"]="openai/gpt-oss-120b"
  ["openrouter_GPT5.4"]="openai/gpt-5.4"
  ["openrouter_KM_k2.5"]="moonshotai/kimi-k2.5"
  ["openrouter_glm-5"]="z-ai/glm-5"
  ["openrouter_gemini_3.1_Pro"]="google/gemini-3.1-pro-preview"
  ["openrouter_gemini_3_Flash"]="google/gemini-3-flash-preview"
)


openrouter_api_key="${OPENROUTER_API_KEY:-Your API Key}"


# Resolve MODEL and backend from MODEL_INFO
if [[ -n "${local_models[${MODEL_INFO}]:-}" ]]; then
  MODEL="${local_models[${MODEL_INFO}]}"
  USE_VLLM=1
  BASE_URL=""
  API_KEY=""
elif [[ -n "${api_models[${MODEL_INFO}]:-}" ]]; then
  MODEL="${api_models[${MODEL_INFO}]}"
  USE_VLLM=0
  BASE_URL="${BASE_URL:-https://openrouter.ai/api/v1}"
  API_KEY="${API_KEY:-${OPENROUTER_API_KEY:-${openrouter_api_key}}}"
else
  echo "Error: MODEL_INFO='${MODEL_INFO}' not in local_models or api_models." >&2
  echo "  local: ${!local_models[*]}" >&2
  echo "  api:   ${!api_models[*]}" >&2
  exit 1
fi

# ── Backend Selection ────────────────────────────────────────────────────────
# Set USE_VLLM=1 for local vLLM, or BASE_URL for OpenAI-compatible API

USE_VLLM="${USE_VLLM}"
BASE_URL="${BASE_URL:-}"
API_KEY="${API_KEY:-}"

# ── vLLM Options ─────────────────────────────────────────────────────────────

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20480}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"

# ── Retrieval ────────────────────────────────────────────────────────────────
# RETRIEVER: bm25, dense, hybrid
# RETRIEVER_MODEL: sentence-transformer model for dense/hybrid
export HF_HOME="/aidata/qiwei/cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

RETRIEVER="${RETRIEVER:-dense}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-ToolBench/ToolBench_IR_bert_based_uncased}"
TOP_K="${TOP_K:-10}"

# ── Scope & Categories ───────────────────────────────────────────────────────────────
SCOPE="${SCOPE:-all}"
TEST_CATEGORY="${TEST_CATEGORY:-all}"
LIMIT="${LIMIT:-}"

# ── Variant ──────────────────────────────────────────────────────────────────
# VARIANT: none (no initial tools), distractor (random non-GT tools)

VARIANT="${VARIANT:-none}"

# ── Output ───────────────────────────────────────────────────────────────────

RESULTS_DIR="${RESULTS_DIR:-results}"
MODEL_DIR="${RESULTS_DIR}/${VARIANT}/${MODEL_INFO}"
mkdir -p "${MODEL_DIR}"

# ── Categories ────────────────────────────────────────────────────────────────

if [[ "${SCOPE}" == "all" ]]; then
  CATEGORIES="simple_python multiple parallel parallel_multiple live_simple live_multiple live_parallel live_parallel_multiple"
elif [[ "${SCOPE}" == "live" ]]; then
  CATEGORIES="live_simple live_multiple live_parallel live_parallel_multiple"
elif [[ "${SCOPE}" == "hammerbench" || "${SCOPE}" == "hammerbench_en" ]]; then
  CATEGORIES="hammerbench"
else
  CATEGORIES="simple_python multiple parallel parallel_multiple"
fi
[[ "${TEST_CATEGORY}" != "all" ]] && CATEGORIES="${TEST_CATEGORY//,/ }"

# BFCL non-live + live: default to single merged tool bank (test_data/BFCL/tool_bank/merged.json).
# Set BFCL_MERGED_TOOL_BANK=0 to load test_data/BFCL/tool_bank/non_live.json and live.json separately.
if [[ "${SCOPE}" == "all" ]]; then
  export BFCL_MERGED_TOOL_BANK="${BFCL_MERGED_TOOL_BANK:-1}"
else
  export BFCL_MERGED_TOOL_BANK="${BFCL_MERGED_TOOL_BANK:-0}"
fi

# ── Check existing results & decide which stages to run ──────────────────────

HAS_INFERENCE=true
HAS_EVAL=true
for cat in ${CATEGORIES}; do
  [[ ! -f "${MODEL_DIR}/Inference_retrieval_${cat}.json" ]] && HAS_INFERENCE=false
  [[ ! -f "${MODEL_DIR}/Eval_retrieval_${cat}.json" ]] && HAS_EVAL=false
done

if ${HAS_INFERENCE} && ${HAS_EVAL}; then
  echo "[SKIP] Inference & Eval results already exist for ${MODEL_INFO}, jumping to LLM Judge."
else

  # ── Run Inference ─────────────────────────────────────────────────────────────

  echo "============================================================"
  echo "  Inference: ${MODEL_INFO} | ${RETRIEVER} (${RETRIEVER_MODEL##*/}) top_k=${TOP_K}"
  echo "============================================================"

  BACKEND_ARGS=""
  if [[ "${USE_VLLM}" == "1" && -z "${BASE_URL}" ]]; then
    BACKEND_ARGS="--use-vllm --max-model-len ${MAX_MODEL_LEN} --tensor-parallel ${TENSOR_PARALLEL} --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}"
  else
    [[ -n "${BASE_URL}" ]] && BACKEND_ARGS="--base-url ${BASE_URL}"
    [[ -n "${API_KEY}" ]] && BACKEND_ARGS="${BACKEND_ARGS} --api-key ${API_KEY}"
  fi

  python run/run_inference.py \
    --model "${MODEL}" --model-info "${MODEL_INFO}" \
    --retriever "${RETRIEVER}" --retriever-model "${RETRIEVER_MODEL}" --top-k "${TOP_K}" \
    --variant "${VARIANT}" --temperature "${TEMPERATURE}" --max-tokens "${MAX_TOKENS}" \
    --output-dir "${RESULTS_DIR}/${VARIANT}" --test-category "${CATEGORIES// /,}" --prompt "${PROMPT}" \
    ${LIMIT:+--limit "${LIMIT}"} ${BACKEND_ARGS}

fi


# ── Run Evaluation ────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Evaluation: ${MODEL_INFO}"
echo "============================================================"

python run/run_eval.py \
  --model-dir "${MODEL_DIR}" \
  --categories ${CATEGORIES} \
  --detail "${MODEL_DIR}/eval_detail.txt"


# ── LLM Judge (tool-name mismatch re-evaluation) ────────────────────────────

SKIP_JUDGE="${SKIP_JUDGE:-0}"

IS_HAMMERBENCH_ONLY=true
for cat in ${CATEGORIES}; do
  [[ "${cat}" != hammerbench* ]] && IS_HAMMERBENCH_ONLY=false
done

if [[ "${SKIP_JUDGE}" == "1" ]]; then
  echo ""
  echo "[SKIP] LLM Judge skipped (SKIP_JUDGE=1)."
elif ${IS_HAMMERBENCH_ONLY}; then
  echo ""
  echo "[SKIP] LLM Judge skipped for hammerbench categories (tools are unique)."
else
  JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-4o-mini}"
  JUDGE_URL="${JUDGE_URL:-https://openrouter.ai/api/v1}"
  JUDGE_KEY="${JUDGE_KEY:-${OPENROUTER_API_KEY:-${openrouter_api_key}}}"
  echo ""
  echo "============================================================"
  echo "  LLM Judge: ${JUDGE_MODEL}"
  echo "============================================================"

  python run/run_judge.py \
    --model-dir "${MODEL_DIR}" \
    --categories ${CATEGORIES} \
    --judge-model "${JUDGE_MODEL}" \
    --base-url "${JUDGE_URL}" \
    --api-key "${JUDGE_KEY}" \
    --max-workers 4 \
    --detail "${MODEL_DIR}/eval_llm_judge.txt"
fi
