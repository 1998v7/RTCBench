set -euo pipefail

TASK="${1:?Usage: bash run_task.sh <task> <model> <top_k> <variant> [extra_env...]}"
MODEL="${2:?Usage: bash run_task.sh <task> <model> <top_k> <variant> [extra_env...]}"
TOP_K="${3:?Usage: bash run_task.sh <task> <model> <top_k> <variant> [extra_env...]}"
VARIANT="${4:?Usage: bash run_task.sh <task> <model> <top_k> <variant> [extra_env...]}"
shift 4

case "${TASK}" in
  bfcl_non_live)
    export SCOPE=non_live
    export TEST_CATEGORY=all
    export RESULTS_DIR="${RESULTS_DIR:-results/bfcl_non_live}"
    export RETRIEVER="${RETRIEVER:-dense}"
    export RETRIEVER_MODEL="${RETRIEVER_MODEL:-ToolBench/ToolBench_IR_bert_based_uncased}"
    export PROMPT="${PROMPT:-thinking_prompt}"
    ;;
  bfcl_live)
    export SCOPE=live
    export TEST_CATEGORY=all
    export RESULTS_DIR="${RESULTS_DIR:-results/bfcl_live}"
    export RETRIEVER="${RETRIEVER:-dense}"
    export RETRIEVER_MODEL="${RETRIEVER_MODEL:-ToolBench/ToolBench_IR_bert_based_uncased}"
    export PROMPT="${PROMPT:-thinking_prompt}"
    ;;
  level_1)
    export SCOPE=all
    export TEST_CATEGORY=all
    export RESULTS_DIR="${RESULTS_DIR:-results/bfcl_all}"
    export RETRIEVER="${RETRIEVER:-dense}"
    export RETRIEVER_MODEL="${RETRIEVER_MODEL:-ToolBench/ToolBench_IR_bert_based_uncased}"
    export PROMPT="${PROMPT:-thinking_prompt}"
    ;;
  level_2)
    export SCOPE=hammerbench
    export HAMMERBENCH_SCOPE=hammerbench
    export TEST_CATEGORY="hammerbench"
    export RESULTS_DIR="${RESULTS_DIR:-results/hammerbench_zh}"
    export RETRIEVER="${RETRIEVER:-dense}"
    export RETRIEVER_MODEL="${RETRIEVER_MODEL:-BAAI/bge-base-zh-v1.5}"
    export PROMPT="${PROMPT:-thinking_prompt_zh}"
    ;;
  *)
    echo "Unknown task: ${TASK}" >&2
    echo "Available: bfcl_non_live, bfcl_live, level_1, level_2" >&2
    exit 1
    ;;
esac

export MODEL_INFO="${MODEL}"
export TOP_K="${TOP_K}"
export VARIANT="${VARIANT}"

# Apply any extra env overrides (e.g. LIMIT=50)
for arg in "$@"; do
  export "${arg}"
done

echo "Task:      ${TASK}"
echo "Model:     ${MODEL_INFO}"
echo "Variant:   ${VARIANT}"
echo "Retriever: ${RETRIEVER} (${RETRIEVER_MODEL##*/}) top_k=${TOP_K}"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Output:    ${RESULTS_DIR}"
echo ""

exec bash "$(dirname "$0")/run.sh"
