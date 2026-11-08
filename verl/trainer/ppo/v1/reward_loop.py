"""Reward computation boundary shared by colocated and streaming V1 trainers."""

from __future__ import annotations

import inspect
from typing import Any, Callable

import torch
from tensordict import TensorDict


class RewardLoopManager:
    """Dispatch reward computation without coupling PPO control flow to its placement."""

    def __init__(self, reward_fn: Callable[[TensorDict], Any] | None = None, worker_handles=None):
        self.reward_fn = reward_fn
        self.reward_loop_worker_handles = list(worker_handles or [])
        if reward_fn is None and not self.reward_loop_worker_handles:
            raise ValueError("RewardLoopManager requires reward_fn or worker_handles")
        self._next_worker = 0

    async def compute(self, batch: TensorDict):
        if self.reward_loop_worker_handles:
            worker = self.reward_loop_worker_handles[self._next_worker % len(self.reward_loop_worker_handles)]
            self._next_worker += 1
            method = worker.compute_reward
            result = method.remote(batch) if hasattr(method, "remote") else method(batch)
            if inspect.isawaitable(result):
                result = await result
            elif type(result).__module__.startswith("ray"):

                result = await result
        else:
            result = self.reward_fn(batch)
            if inspect.isawaitable(result):
                result = await result
        if isinstance(result, torch.Tensor):
            batch["token_level_rewards"] = result
        elif isinstance(result, dict):
            rewards = result.get("reward_tensor", result.get("token_level_rewards"))
            if rewards is None:
                raise ValueError("Reward result dict has no reward tensor")
            batch["token_level_rewards"] = rewards
        else:
            raise TypeError(f"Unsupported reward result {type(result)!r}")
        return batch
