# Directory Overview

This directory contains training and evaluation code for retrieval-augmented tool-calling models. Only two directory levels are shown below:

```text
.
├── Train/            # Reinforcement learning training project
│   ├── data/         # Training and validation datasets
│   ├── src/          # GRPO entry point, interactions, rewards, configs, and tool banks
│   └── verl/         # VERL reinforcement learning framework for large language models
└── RTCBench-main/    # RTCBench tool-calling evaluation project
    ├── RTCBench/     # Model inference, tool retrieval, rule evaluation, and LLM judge scripts
    └── test_data/    # Benchmark datasets and corresponding tool banks

```

The final GRPO checkpoint is uploaded to Hugging Face ([Url](https://huggingface.co/Qi9802/Qwen3_4B_RLckp/tree/main)).