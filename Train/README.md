# Retrieval-Augmented Tool-Calling RL Training

This directory contains the code and tool-bank metadata needed for the GRPO
training pipeline.

## Contents

- `verl/`: the VERL training framework
- `src/`: retrieval interaction, reward, trainer, configuration,
  launch script, and the live/non-live tool banks used by the retrieval
  environment
- `data/`: RL training and validation datasets

Model checkpoints, embedding model weights, experiment logs, and generated
benchmark outputs are intentionally not included.

## Configuration

The launch script resolves paths relative to this directory. Override these
environment variables when your files are stored elsewhere:

- `MODEL_PATH`: SFT checkpoint used to initialize the policy
- `DATASET_PATH`: training parquet file
- `VAL_DATASET_PATH`: validation parquet file
- `RETRIEVAL_MODEL_PATH`: local path or Hugging Face model ID for the retriever
- `OUTPUT_DIR`: checkpoint and validation output directory
- `N_GPUS`, `CUDA_VISIBLE_DEVICES`: GPU configuration

Example:

```bash
MODEL_PATH=/path/to/sft_checkpoint \
DATASET_PATH=/path/to/rl_train.parquet \
VAL_DATASET_PATH=/path/to/rl_val.parquet \
bash src/run_grpo.sh
```
