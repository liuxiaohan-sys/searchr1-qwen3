"""TensorDict-native PPO control flow shared by V1 trainer modes."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from typing import Any, Callable

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.data_protocol_adapter import validate_training_tensordict


class PPOTrainer:
    """Backend-agnostic PPO transaction over versioned TransferQueue batches."""

    def __init__(
        self,
        config=None,
        *,
        actor_wg=None,
        critic_wg=None,
        ref_wg=None,
        queue=None,
        replay_buffer=None,
        checkpoint_manager=None,
        reward_fn: Callable[[TensorDict], Any] | None = None,
        reward_loop_manager=None,
        advantage_fn: Callable[[TensorDict], TensorDict] | None = None,
        validation_fn: Callable[[], dict] | None = None,
    ):
        self.config = config
        self.actor_wg = actor_wg
        self.critic_wg = critic_wg
        self.ref_wg = ref_wg
        self.queue = queue
        self.replay_buffer = replay_buffer
        self.checkpoint_manager = checkpoint_manager
        self.reward_fn = reward_fn
        self.reward_loop_manager = reward_loop_manager
        self.advantage_fn = advantage_fn
        self.validation_fn = validation_fn
        self.global_steps = 0

    def init(self):
        required = {"actor_wg": self.actor_wg, "queue": self.queue, "replay_buffer": self.replay_buffer}
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(f"V1 PPOTrainer is missing runtime components: {missing}")

    @staticmethod
    def _merge(batch: TensorDict, output: Any) -> TensorDict:
        if output is None:
            return batch
        if hasattr(output, "get") and not isinstance(output, dict | TensorDict):
            output = output.get()
        if isinstance(output, TensorDict):
            for key, value in output.items():
                batch[key] = value
        elif isinstance(output, dict):
            for key, value in output.items():
                if key != "metrics":
                    batch[key] = value
        else:
            raise TypeError(f"Worker returned unsupported output {type(output)!r}")
        return batch

    @staticmethod
    def _wait(value):
        if inspect.isawaitable(value):
            return asyncio.run(value)
        return value

    @staticmethod
    def _metrics(output: Any) -> dict:
        if output is None:
            return {}
        if hasattr(output, "get") and not isinstance(output, dict | TensorDict):
            output = output.get()
        if isinstance(output, TensorDict):
            return dict(tu.unwrap_non_tensor_data(output.get("metrics", {})))
        if isinstance(output, dict):
            return dict(tu.unwrap_non_tensor_data(output.get("metrics", {})))
        return {}

    def _default_advantage(self, batch: TensorDict) -> TensorDict:
        from verl.trainer.ppo import core_algos

        rewards = batch["token_level_rewards"]
        mask = batch["response_mask"].to(rewards.dtype)
        algorithm = getattr(self.config, "algorithm", None)
        estimator = str(getattr(algorithm, "adv_estimator", "gae"))
        if estimator == core_algos.AdvantageEstimator.GAE.value:
            if "values" not in batch:
                raise RuntimeError("GAE requires critic values; enable critic or choose a critic-free estimator")
            advantages, returns = core_algos.compute_gae_advantage_return(
                token_level_rewards=rewards,
                values=batch["values"],
                response_mask=mask,
                gamma=float(getattr(algorithm, "gamma", 1.0)),
                lam=float(getattr(algorithm, "lam", 1.0)),
            )
        else:
            estimator_fn = core_algos.get_adv_estimator_fn(estimator)
            kwargs = {
                "token_level_rewards": rewards,
                "response_mask": mask,
                "config": algorithm,
            }
            if "uid" in batch:
                uid = batch["uid"]
                kwargs["index"] = getattr(uid, "tolist", lambda: tu.unwrap_non_tensor_data(uid))()
            if "reward_baselines" in batch:
                kwargs["reward_baselines"] = batch["reward_baselines"]
            advantages, returns = estimator_fn(**kwargs)
        batch["advantages"] = advantages
        batch["returns"] = returns
        return batch

    def train_kv_batch(self, meta, *, global_steps: int) -> tuple[TensorDict, dict]:
        batch = self.queue.get_tensordict(meta)
        validate_training_tensordict(batch, require_versions=True)
        metrics: dict[str, Any] = {}
        batch = self._merge(batch, self.actor_wg.compute_log_prob(batch))
        if self.ref_wg is not None:
            batch = self._merge(batch, self.ref_wg.compute_ref_log_prob(batch))
        if self.critic_wg is not None:
            batch = self._merge(batch, self.critic_wg.infer_batch(batch))
        if self.reward_loop_manager is not None:
            batch = self._wait(self.reward_loop_manager.compute(batch))
        elif self.reward_fn is not None:
            reward_output = self.reward_fn(batch)
            if isinstance(reward_output, torch.Tensor):
                batch["token_level_rewards"] = reward_output
            else:
                batch = self._merge(batch, reward_output)
        if "token_level_rewards" not in batch:
            raise RuntimeError("PPO step did not produce token_level_rewards")
        batch = self.advantage_fn(batch) if self.advantage_fn is not None else self._default_advantage(batch)
        # TrainingWorker/Engine consume batching policy as TensorDict metadata.
        actor_cfg = getattr(getattr(self.config, "actor_rollout_ref", None), "actor", None)
        critic_cfg = getattr(self.config, "critic", None)
        if self.critic_wg is not None and critic_cfg is not None:
            tu.assign_non_tensor(
                batch,
                mini_batch_size=critic_cfg.ppo_mini_batch_size,
                epochs=critic_cfg.ppo_epochs,
                compute_loss=True,
            )
        validate_training_tensordict(batch, require_versions=True)
        if self.critic_wg is not None:
            critic_output = self.critic_wg.train_batch(batch)
            metrics.update(self._metrics(critic_output))
        if actor_cfg is not None:
            tu.assign_non_tensor(
                batch,
                mini_batch_size=actor_cfg.ppo_mini_batch_size,
                epochs=actor_cfg.ppo_epochs,
                compute_loss=True,
            )
        actor_output = self.actor_wg.update_actor(batch)
        metrics.update(self._metrics(actor_output))
        if self.should_sync_weights(global_steps):
            if self.checkpoint_manager is not None:
                self._wait(self.checkpoint_manager.update_weights(global_steps))
            else:
                self._wait(self.actor_wg.update_weights(global_steps=global_steps))
        self.queue.clear(meta)
        return batch, metrics

    def should_sync_weights(self, global_steps: int) -> bool:
        """Return whether this optimizer step publishes a new rollout version."""
        return True

    def step(self, *, global_steps: int, partition_id: str, batch_size: int):
        meta, metrics = self.replay_buffer.sample(global_steps, partition_id, batch_size)
        batch, train_metrics = self.train_kv_batch(meta, global_steps=global_steps)
        metrics.update(train_metrics)
        return batch, metrics

    def save_checkpoint(self, local_dir: str, global_steps: int) -> None:
        root = os.path.join(local_dir, f"global_step_{global_steps}")
        os.makedirs(root, exist_ok=True)
        self.actor_wg.save_checkpoint(os.path.join(root, "actor"), None, global_steps)
        if self.critic_wg is not None:
            self.critic_wg.save_checkpoint(os.path.join(root, "critic"), None, global_steps)
        with open(os.path.join(local_dir, "latest_checkpointed_iteration.txt"), "w", encoding="utf-8") as file:
            file.write(str(global_steps))
        with open(os.path.join(root, "v1_state.json"), "w", encoding="utf-8") as file:
            json.dump({"global_steps": global_steps}, file)

    def load_checkpoint(self, local_dir: str) -> int:
        marker = os.path.join(local_dir, "latest_checkpointed_iteration.txt")
        if not os.path.exists(marker):
            return 0
        with open(marker, encoding="utf-8") as file:
            global_steps = int(file.read().strip())
        root = os.path.join(local_dir, f"global_step_{global_steps}")
        self.actor_wg.load_checkpoint(os.path.join(root, "actor"))
        if self.critic_wg is not None:
            self.critic_wg.load_checkpoint(os.path.join(root, "critic"))
        self.global_steps = global_steps
        return global_steps

    def validate(self) -> dict:
        return self.validation_fn() if self.validation_fn is not None else {}
