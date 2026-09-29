# RTCBench: Evaluating Tool Use of Large Language Models Beyond Oracle Access

**This paper has been accepted to NeurIPS 2026 (Evaluations and Datasets Track)**

This repo contains training and evaluation benchmark code for retrieval-augmented tool-calling models. Only two directory levels are shown below:

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

The benchmark details can be found in ([Url](https://github.com/1998v7/RTCBench/tree/main/RTCBench-main/RTCBench)).

The training program can be found in ([Url](https://github.com/1998v7/RTCBench/tree/main/Train)).
