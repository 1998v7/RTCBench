"""RayRetrievalTrainer: GRPO trainer for outcome-only retrieval tool-calling.

Training flow per step:
  1. Data      — load a batch of (system+user) prompts from parquet
  2. Rollout   — agent loop: model generates search call → interaction executes
                 retrieval → model generates final tool call (multi-turn)
  3. Reward    — comes from SearchToolInteraction.turn_scores:
                   outcome(0/1) + search_behavior(0/0.3) + format(-0.3)
  4. Advantage — GRPO (group relative policy optimization)
  5. Update    — actor update with PPO clip loss + KL loss
  6. Log       — standard metrics + retrieval-specific: outcome_rate, search_rate,
                 reward breakdown
  7. Validate  — periodic validation with outcome accuracy on val set
"""

import uuid
from collections import defaultdict
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from omegaconf import OmegaConf
from verl.utils.tracking import Tracking


class RayRetrievalTrainer(RayPPOTrainer):
    """GRPO trainer for outcome-only retrieval-augmented tool calling.

    Inherits all worker management, checkpointing, and data loading from
    RayPPOTrainer. Overrides fit() and _validate() to add retrieval-specific
    metric logging and simplified training loop (no REMAX, no multimodal,
    no rollout correction — not needed for this task).
    """

    # ── Retrieval-specific metric helpers ────────────────────────────────────

    def _compute_retrieval_metrics(self, batch: DataProto, reward_tensor: torch.Tensor,
                                    reward_extra_infos_dict: dict = None) -> dict:
        """Extract retrieval-specific training metrics from reward components.

        Components come directly from reward.py compute_score():
        """
        traj_rewards = reward_tensor.sum(dim=-1).float()
        metrics = {"retrieval/mean_reward": traj_rewards.mean().item()}

        if reward_extra_infos_dict:
            if "format_ok" in reward_extra_infos_dict:
                metrics["retrieval/format_score"] = float(np.mean(reward_extra_infos_dict["format_ok"]))
            if "outcome" in reward_extra_infos_dict:
                metrics["retrieval/outcome_score"] = float(np.mean(reward_extra_infos_dict["outcome"]))
            if "search_ok" in reward_extra_infos_dict:
                metrics["retrieval/search_behavior_score"] = float(np.mean(reward_extra_infos_dict["search_ok"]))
            if "search_tool_count" in reward_extra_infos_dict:
                metrics["retrieval/avg_search_tool_count"] = float(np.mean(reward_extra_infos_dict["search_tool_count"]))
            if "variant_is_none" in reward_extra_infos_dict:
                mask = np.array(reward_extra_infos_dict["variant_is_none"])
                none_mask = mask > 0.5
                gt_mask = ~none_mask
                if "search_tool_count" in reward_extra_infos_dict:
                    counts = np.array(reward_extra_infos_dict["search_tool_count"])
                    if none_mask.any():
                        metrics["retrieval/avg_search_count_none"] = float(counts[none_mask].mean())
                    if gt_mask.any():
                        metrics["retrieval/avg_search_count_with_gt"] = float(counts[gt_mask].mean())
                if "retrieval_recall" in reward_extra_infos_dict:
                    recalls = np.array(reward_extra_infos_dict["retrieval_recall"])
                    if none_mask.any():
                        metrics["retrieval/avg_recall_none"] = float(recalls[none_mask].mean())
                if "search_truncated" in reward_extra_infos_dict:
                    trunc = np.array(reward_extra_infos_dict["search_truncated"])
                    if none_mask.any():
                        metrics["retrieval/search_truncated_rate_none"] = float(trunc[none_mask].mean())

        return metrics

    # ── Training loop ─────────────────────────────────────────────────────────

    def fit(self):

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # Load checkpoint, sync weights to rollout engine
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        current_epoch = self.global_steps // max(len(self.train_dataloader), 1)

        # ── Initial validation ───────────────────────────────────────────────
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training",
        )

        self.global_steps += 1
        last_val_metrics = None

        # ── Main training loop ───────────────────────────────────────────────
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)

                metrics = {}
                timing_raw = defaultdict(float)

                # ── Step 1: Prepare batch ────────────────────────────────────
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # Assign unique IDs for GRPO group matching
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

                # Build generation input (keep reward-related keys, drop tensor keys)
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps

                # Repeat each prompt rollout_n times (GRPO samples K trajectories per prompt)
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):

                    # ── Step 2: Rollout (multi-turn agent loop) ──────────────
                    # SearchToolInteraction handles:
                    #   Turn 1: model outputs [search_tool(query="...")]
                    #           → interaction executes dense retrieval
                    #           → returns retrieved tools as user message
                    #   Turn 2: model outputs [func_call(params)]
                    #           → interaction computes reward, terminates
                    with marked_timer("gen", timing_raw, color="red"):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        self.checkpoint_manager.sleep_replicas()
                        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                        gen_batch_output.meta_info.pop("timing", None)

                    # Expand original batch to match rollout repeats, then merge outputs
                    batch = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    batch = batch.union(gen_batch_output)

                    # Compute response mask (1 = generated token, 0 = prompt/pad)
                    if "response_mask" not in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance token counts across DP ranks for even compute
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = (
                        torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    )

                    # ── Step 3: Extract reward ───────────────────────────────
                    # Rewards come from interaction turn_scores (set by SearchToolInteraction).
                    # extract_reward pulls them from batch.batch["token_level_scores"]
                    # which was populated by the agent loop.
                    with marked_timer("reward", timing_raw, color="yellow"):
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    # Log retrieval-specific metrics
                    metrics.update(self._compute_retrieval_metrics(batch, reward_tensor, reward_extra_infos_dict))

                    # ── Step 4: Compute old log probs + ref log probs ────────
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob, _ = self._compute_old_log_prob(batch)
                        # Log entropy
                        entropys = old_log_prob.batch["entropys"]
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=batch.batch["response_mask"],
                            loss_agg_mode=actor_config.loss_agg_mode,
                        )
                        metrics["actor/entropy"] = entropy_agg.detach().item()
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # ── Step 5: Compute advantages (GRPO) ───────────────────
                    with marked_timer("adv", timing_raw, color="brown"):
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # Apply KL penalty if configured (default: off)
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # GRPO: advantage = (reward - group_mean) / group_std
                        # group = all K rollouts from the same prompt
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                            config=self.config.algorithm,
                        )

                    # ── Step 6: Actor update ─────────────────────────────────
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Save checkpoint
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                        ):
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # Sync updated weights to rollout engine
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights()

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                # ── Step 7: Validation ───────────────────────────────────────
                if self.config.trainer.test_freq > 0 and (
                    is_last_step
                    or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # ── Step 8: Log all metrics ──────────────────────────────────
                metrics.update(compute_data_metrics(batch=batch, use_critic=False))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch,
                        timing_raw=timing_raw,
                        n_gpus=self.resource_pool_manager.get_n_gpus(),
                    )
                )

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1

        # Save if last checkpoint was not saved inside the loop
        checkpoint_dir = (
            f"{self.config.trainer.default_local_dir}/global_step_{self.global_steps}"
        )
        import os
        if not os.path.exists(checkpoint_dir):
            self._save_checkpoint()

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self, merged: bool = False) -> dict:
        """Run validation and return retrieval-specific metrics.

        Metrics reported:
          val/outcome_accuracy  — fraction of val samples where final tool call is correct
          val/mean_reward       — mean trajectory reward on val set
          val/search_rate       — fraction that called search_tool
          val/format_fail_rate  — fraction that hit format penalty
        """
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.ppo.ray_trainer import pad_dataproto_to_divisor

        all_rewards = []
        all_extra = defaultdict(list)
        all_batches = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # Use n=1 for validation (greedy/single sample)
            val_n = self.config.actor_rollout_ref.rollout.get("val_kwargs", {}).get("n", 1)
            test_batch = test_batch.repeat(repeat_times=val_n, interleave=True)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id":       self.tokenizer.eos_token_id,
                "pad_token_id":       self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample":          False,   # greedy for validation
                "validate":           True,
                "global_steps":       self.global_steps,
            }

            # Pad to divisor required by agent loop workers
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, size_divisor
            )

            # Run multi-turn rollout (same agent loop as training)
            output_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # Remove padding
            if pad_size > 0:
                output = output_padded[:-pad_size]
                test_batch = test_batch[:-pad_size] if pad_size > 0 else test_batch
            else:
                output = output_padded

            test_batch = test_batch.union(output)

            # Extract rewards and per-component scores from reward loop
            if "rm_scores" in test_batch.batch:
                traj_rewards = test_batch.batch["rm_scores"].sum(dim=-1).float()
                all_rewards.extend(traj_rewards.tolist())
            for key in ("outcome", "search_ok", "format_ok"):
                if key in test_batch.non_tensor_batch:
                    all_extra[key].extend(test_batch.non_tensor_batch[key].tolist())

            all_batches.append((output, test_batch))

        if not all_rewards:
            return {}

        rewards = torch.tensor(all_rewards)
        metrics = {"val/mean_reward": rewards.mean().item()}

        if all_extra["format_ok"]:
            metrics["val/format_score"] = float(np.mean(all_extra["format_ok"]))
        if all_extra["outcome"]:
            metrics["val/outcome_score"] = float(np.mean(all_extra["outcome"]))
        if all_extra["search_ok"]:
            metrics["val/search_behavior_score"] = float(np.mean(all_extra["search_ok"]))

        # ── Save sample trajectories for inspection ────────────────────────
        try:
            self._save_val_trajectories(all_batches)
        except Exception as e:
            pprint(f"Warning: failed to save val trajectories: {e}")

        return metrics

    def _save_val_trajectories(self, all_batches: list):
        """Decode and save all validation trajectories to JSON for inspection."""
        import json as _json
        from pathlib import Path

        from reward import compute_score as _compute_score

        all_entries: list[dict] = []

        for output, test_batch in all_batches:
            bsz = output.batch["responses"].shape[0]

            for i in range(bsz):
                # ── Extract variant / scope / ground_truth from extra_info ────
                ei = test_batch.non_tensor_batch.get("extra_info", {})
                if isinstance(ei, np.ndarray):
                    ei = ei[i]
                elif isinstance(ei, list):
                    ei = ei[i]
                if isinstance(ei, str):
                    ei = _json.loads(ei)
                ik = ei.get("interaction_kwargs", {}) if isinstance(ei, dict) else {}
                variant = ik.get("variant", "none")
                scope = ik.get("scope", "non_live")

                gt = ""
                rm = test_batch.non_tensor_batch.get("reward_model", None)
                if rm is not None:
                    rm_i = rm[i] if hasattr(rm, '__getitem__') else rm
                    if isinstance(rm_i, dict):
                        gt = rm_i.get("ground_truth", "")
                    elif isinstance(rm_i, str):
                        gt = _json.loads(rm_i).get("ground_truth", "")

                # ── Decode response ───────────────────────────────────────────
                resp_ids = output.batch["responses"][i]
                resp_mask = output.batch["response_mask"][i]
                attn_full = output.batch["attention_mask"][i]
                prompt_len = output.batch["prompts"][i].shape[0]

                resp_attn = attn_full[prompt_len:]
                valid_resp_ids = resp_ids[resp_attn.bool()]
                valid_resp_mask = resp_mask[resp_attn.bool()]

                response_text = self.tokenizer.decode(valid_resp_ids, skip_special_tokens=True)

                # ── Split into model / env turns by response_mask ─────────────
                turns = []
                mask_list = valid_resp_mask.tolist()
                ids_list = valid_resp_ids.tolist()
                cur_type, cur_ids = None, []
                for token_id, m in zip(ids_list, mask_list):
                    t = "model" if m == 1 else "env"
                    if t != cur_type and cur_ids:
                        text = self.tokenizer.decode(cur_ids, skip_special_tokens=True).strip()
                        if text:
                            turns.append({"role": cur_type, "content": text})
                        cur_ids = []
                    cur_type = t
                    cur_ids.append(token_id)
                if cur_ids:
                    text = self.tokenizer.decode(cur_ids, skip_special_tokens=True).strip()
                    if text:
                        turns.append({"role": cur_type, "content": text})

                # ── Compute reward locally (always available) ─────────────────
                reward_result = _compute_score(
                    data_source="bfcl_retrieval3",
                    solution_str=response_text,
                    ground_truth=gt,
                    extra_info=ei,
                )

                # ── Extract user query from prompt ────────────────────────────
                raw_prompt = test_batch.non_tensor_batch.get("raw_prompt", None)
                user_query = ""
                if raw_prompt is not None:
                    rp = raw_prompt[i] if hasattr(raw_prompt, '__getitem__') else raw_prompt
                    if isinstance(rp, (list, tuple)):
                        for msg in rp:
                            if isinstance(msg, dict) and msg.get("role") == "user":
                                user_query = msg["content"]
                                break

                entry = {
                    "variant": variant,
                    "scope": scope,
                    "user_query": user_query,
                    "ground_truth": gt,
                    "reward": reward_result.get("score", 0.0),
                    "outcome": reward_result.get("outcome", 0.0),
                    "search_ok": reward_result.get("search_ok", 0.0),
                    "format_ok": reward_result.get("format_ok", 0.0),
                    "turns": turns,
                    "response_raw": response_text,
                }

                entry["index"] = len(all_entries)
                all_entries.append(entry)

        # ── Save ──────────────────────────────────────────────────────────
        save_dir = Path(self.config.trainer.default_local_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"val_trajectories_step{self.global_steps}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            _json.dump(all_entries, f, ensure_ascii=False, indent=2)
        n_none = sum(1 for s in all_entries if s["variant"] == "none")
        n_gt = sum(1 for s in all_entries if s["variant"] == "with_gt")
        pprint(f"Saved {len(all_entries)} val trajectories (none={n_none}, with_gt={n_gt}) to {save_path}")
