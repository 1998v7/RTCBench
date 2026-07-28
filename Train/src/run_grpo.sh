#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "${SCRIPT_DIR}")}"
VERL_ROOT="${VERL_ROOT:-${PROJECT_ROOT}/verl}"
TRAIN_ROOT="${TRAIN_ROOT:-${SCRIPT_DIR}}"
export PROJECT_ROOT VERL_ROOT

# --- Data ---
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/data/rl_train.parquet}"
VAL_DATASET_PATH="${VAL_DATASET_PATH:-${PROJECT_ROOT}/data/rl_val.parquet}"

# --- Model (SFT checkpoint) ---
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/sft_checkpoint}"

# --- Reward ---
REWARD_PATH="${REWARD_PATH:-${TRAIN_ROOT}/reward.py}"

# --- Interaction config ---
INTERACTION_CONFIG="${INTERACTION_CONFIG:-${TRAIN_ROOT}/interaction_config.yaml}"

# --- Tool banks for dense retrieval inside interaction ---
export NON_LIVE_TOOL_BANK_PATH="${NON_LIVE_TOOL_BANK_PATH:-${TRAIN_ROOT}/tool_bank/non_live.json}"
export LIVE_TOOL_BANK_PATH="${LIVE_TOOL_BANK_PATH:-${TRAIN_ROOT}/tool_bank/live.json}"
export RETRIEVAL_MODEL_PATH="${RETRIEVAL_MODEL_PATH:-ToolBench/ToolBench_IR_bert_based_uncased}"
export RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-10}"
export EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-/tmp/bfcl_emb_cache}"

# --- HuggingFace cache ---
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

# The training source directory must be on PYTHONPATH so interaction.py is importable
export PYTHONPATH="${TRAIN_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"

# --- Training hyperparams ---
TRAIN_BATCH="${TRAIN_BATCH:-256}"
LR="${LR:-1e-6}"

export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-4096}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-16384}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-20480}"
export MAX_SEARCH_RESULT_LEN="${MAX_SEARCH_RESULT_LEN:-8192}"

EPOCHS="${EPOCHS:-10}"
N_GPUS="${N_GPUS:-8}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-10}"

ROLLOUT_N="${ROLLOUT_N:-5}"
TP_SIZE="${TP_SIZE:-4}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-2}"

FILTER_WORKERS="${FILTER_WORKERS:-8}"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/saves/retrieval_grpo_Qwen3_13k_v4}"

echo "=============================================="
echo "  VERL GRPO: Retrieval-Augmented Tool Calling"
echo "=============================================="
echo "[config] Dataset:     ${DATASET_PATH}"
echo "[config] Val:         ${VAL_DATASET_PATH}"
echo "[config] Model:       ${MODEL_PATH}"
echo "[config] Reward:      ${REWARD_PATH}"
echo "[config] Interaction: ${INTERACTION_CONFIG}"
echo "[config] Output:      ${OUTPUT_DIR}"
echo "[config] max_prompt=${MAX_PROMPT_LEN}  max_response=${MAX_RESPONSE_LEN}  max_model_len=${MAX_MODEL_LEN}"
echo ""

mkdir -p "${OUTPUT_DIR}"

cd "${TRAIN_ROOT}"
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
python3 main.py \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files="${DATASET_PATH}" \
    data.val_files="${VAL_DATASET_PATH}" \
    data.train_batch_size="${TRAIN_BATCH}" \
    data.val_batch_size="${TRAIN_BATCH}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers="${FILTER_WORKERS}" \
    data.truncation=right \
    data.shuffle=True \
    data.prompt_key=prompt \
    data.reward_fn_key=data_source \
    reward.custom_reward_function.path="${REWARD_PATH}" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_model.enable=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_ASSISTANT_TURNS}" \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="${INTERACTION_CONFIG}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.project_name=bfcl_retrieval_grpo_13k \
    trainer.experiment_name=fine_grained_reward_Qwen3_13k \
    trainer.logger='["console","wandb"]' \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.total_epochs="${EPOCHS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    "$@"
