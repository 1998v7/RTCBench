import os
import socket
import sys

import hydra
import ray
from omegaconf import OmegaConf

# Ensure interaction.py and reward.py in this directory are importable by Ray workers
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.dirname(_HERE),
)
_VERL = os.environ.get("VERL_ROOT", os.path.join(_PROJECT_ROOT, "verl"))
for _p in [_HERE, _VERL]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Tool bank / retrieval paths consumed by interaction.py at worker runtime
_TOOL_BANK_DIR = os.path.join(_HERE, "tool_bank")
os.environ.setdefault("NON_LIVE_TOOL_BANK_PATH", os.path.join(_TOOL_BANK_DIR, "non_live.json"))
os.environ.setdefault("LIVE_TOOL_BANK_PATH", os.path.join(_TOOL_BANK_DIR, "live.json"))
os.environ.setdefault("RETRIEVAL_MODEL_PATH", "ToolBench/ToolBench_IR_bert_based_uncased")
os.environ.setdefault("RETRIEVAL_TOP_K", "10")
os.environ.setdefault("EMBEDDING_CACHE_DIR", "/tmp/bfcl_emb_cache")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.environ["HF_HOME"], "datasets"))


from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.device import auto_set_device

from ray_trainer import RayRetrievalTrainer


@hydra.main(config_path="config", config_name="retrieval_grpo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_grpo(config)


def run_grpo(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    try:
        runner = TaskRunner.remote()
        ray.get(runner.run.remote(config))
    finally:
        if ray.is_initialized():
            ray.shutdown()


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from verl.utils.fs import copy_to_local
        from verl.utils import hf_processor, hf_tokenizer
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download checkpoint
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # Tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Worker classes (FSDP only for now)
        assert config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}, \
            f"Only fsdp/fsdp2 supported, got {config.actor_rollout_ref.actor.strategy}"

        from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(AsyncActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {Role.ActorRollout: global_pool_id,}

        # Reference policy for KL loss
        if config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(AsyncActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # Build datasets and sampler (same as verl/trainer/main_ppo.py)
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayRetrievalTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
