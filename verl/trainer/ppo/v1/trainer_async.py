"""Unified asynchronous V1 trainer modes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .trainer_base import PPOTrainer


class PPOTrainerAsync(PPOTrainer):
    """Overlap rollout submission with replay-buffer consumption."""

    def __init__(self, *args, num_warmup_batches: int = 1, parameter_sync_step: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        if num_warmup_batches < 1 or parameter_sync_step < 1:
            raise ValueError("async warmup and parameter_sync_step must be positive")
        self.num_warmup_batches = num_warmup_batches
        self.parameter_sync_step = parameter_sync_step
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="v1-rollout")
        self._rollout_futures = []
        self._initialized_weights = False

    def init(self):
        super().init()
        self._validate_mode()
        if self.checkpoint_manager is not None:
            self._wait(self.checkpoint_manager.update_weights(self.global_steps))
            self._wait(self.checkpoint_manager.wake_up_replicas())
            self._initialized_weights = True

    def _validate_mode(self):
        """Mode-specific validation hook."""

    def submit_rollout(self, agent_loop_manager, prompts):
        future = self._executor.submit(agent_loop_manager.generate_sequences, prompts)
        self._rollout_futures.append(future)
        return future

    def drain_rollouts(self):
        for future in self._rollout_futures:
            future.result()
        self._rollout_futures.clear()

    def should_sync_weights(self, global_steps: int) -> bool:
        return global_steps % self.parameter_sync_step == 0

    def close(self):
        self.drain_rollouts()
        if self.checkpoint_manager is not None:
            self._wait(self.checkpoint_manager.wait_for_weight_update())
        self._executor.shutdown()

    def load_checkpoint(self, local_dir: str) -> int:
        global_steps = super().load_checkpoint(local_dir)
        if global_steps and self.checkpoint_manager is not None:
            self._wait(self.checkpoint_manager.update_weights(global_steps))
            self._wait(self.checkpoint_manager.wake_up_replicas())
        return global_steps

    def validate(self) -> dict:
        if self.checkpoint_manager is not None:
            self._wait(self.checkpoint_manager.wait_for_weight_update())
        return super().validate()


class PPOTrainerColocateAsync(PPOTrainerAsync):
    """Async sampling on rollout engines that share the actor GPU pool.

    Sampling is asynchronous at request level.  At every optimizer step the
    colocated replicas are quiesced and put to sleep so training can reclaim
    their weight/KV-cache memory, then woken after the new weights are loaded.
    """

    def _validate_mode(self):
        backend = getattr(getattr(self.checkpoint_manager, "config", None), "backend", "naive")
        if backend != "naive":
            raise ValueError("colocate_async requires checkpoint_engine.backend='naive'")

    def step(self, *, global_steps: int, partition_id: str, batch_size: int):
        # Do not suspend rollout while sample() is still waiting: in-flight
        # agent/tool turns may need another vLLM request to finish the batch.
        meta, metrics = self.replay_buffer.sample(global_steps, partition_id, batch_size)
        if self.checkpoint_manager is not None:
            self._wait(self.checkpoint_manager.abort_replicas())
            self._wait(self.checkpoint_manager.sleep_replicas())
        try:
            batch, train_metrics = self.train_kv_batch(meta, global_steps=global_steps)
            metrics.update(train_metrics)
            return batch, metrics
        finally:
            if self.checkpoint_manager is not None:
                self._wait(self.checkpoint_manager.wake_up_replicas())


class PPOTrainerSeparateAsync(PPOTrainerAsync):
    """Decoupled training with continuously resident standalone rollout GPUs."""

    def _validate_mode(self):
        backend = getattr(getattr(self.checkpoint_manager, "config", None), "backend", "naive")
        if backend == "naive":
            raise ValueError("separate_async requires a registered standalone checkpoint transport")
        if self.config is None:
            return
        data = getattr(self.config, "data", None)
        actor = getattr(getattr(self.config, "actor_rollout_ref", None), "actor", None)
        train_batch_size = getattr(data, "train_batch_size", None)
        mini_batch_size = getattr(actor, "ppo_mini_batch_size", None)
        if train_batch_size is not None and mini_batch_size is not None:
            expected = self.parameter_sync_step * mini_batch_size
            if train_batch_size != expected:
                raise ValueError(
                    "separate_async requires data.train_batch_size == "
                    "parameter_sync_step * actor.ppo_mini_batch_size; "
                    f"got {train_batch_size} != {self.parameter_sync_step} * {mini_batch_size}"
                )
        rollout = getattr(getattr(self.config, "actor_rollout_ref", None), "rollout", None)
        if self.parameter_sync_step > 1 and not bool(getattr(rollout, "calculate_log_probs", False)):
            raise ValueError(
                "separate_async with parameter_sync_step > 1 requires "
                "actor_rollout_ref.rollout.calculate_log_probs=true so PPO uses a stable behavior policy"
            )

    def compute_old_log_prob(self, batch):
        if self.parameter_sync_step == 1:
            return super().compute_old_log_prob(batch)
        if "rollout_log_probs" not in batch:
            raise RuntimeError(
                "Decoupled PPO requires rollout_log_probs when parameter_sync_step > 1"
            )
        batch["old_log_probs"] = batch["rollout_log_probs"].detach()
        return batch
