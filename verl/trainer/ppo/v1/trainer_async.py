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
        self._executor.shutdown()


class PPOTrainerColocateAsync(PPOTrainerAsync):
    """Async generation and training sharing the same resource pool."""


class PPOTrainerSeparateAsync(PPOTrainerAsync):
    """Async generation on standalone rollout replicas."""
