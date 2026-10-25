from abc import ABC, abstractmethod
from typing import Generator, Optional

import torch


class CheckpointEngineRegistry:
    _registry: dict[str, type["CheckpointEngine"]] = {}

    @classmethod
    def register(cls, backend: str):
        def decorator(engine_cls: type["CheckpointEngine"]):
            cls._registry[backend] = engine_cls
            return engine_cls

        return decorator

    @classmethod
    def new(cls, backend: str, *args, **kwargs) -> "CheckpointEngine":
        if backend not in cls._registry:
            raise ValueError(f"Checkpoint engine {backend!r} is not registered")
        return cls._registry[backend](*args, **kwargs)


class CheckpointEngine(ABC):
    """Versioned transport boundary between a training engine and rollout."""

    @abstractmethod
    async def send_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None], global_steps: Optional[int] = None
    ):
        raise NotImplementedError

    @abstractmethod
    async def receive_weights(self, global_steps: Optional[int] = None):
        raise NotImplementedError


@CheckpointEngineRegistry.register("naive")
class ColocatedCheckpointEngine(CheckpointEngine):
    """In-process transport used by colocated synchronous rollout."""

    def __init__(self, bucket_size: int = 0, is_master: bool = False, **kwargs):
        self.bucket_size = bucket_size
        self.is_master = is_master
        self._weights = None
        self._global_steps: Optional[int] = None

    async def send_weights(self, weights, global_steps=None):
        if self._global_steps is not None and global_steps is not None and global_steps < self._global_steps:
            raise ValueError(f"Refusing stale weight version {global_steps}; current version is {self._global_steps}")
        self._weights = weights
        self._global_steps = global_steps

    async def receive_weights(self, global_steps=None):
        if self._weights is None:
            raise RuntimeError("No weights have been sent")
        if global_steps is not None and self._global_steps != global_steps:
            raise ValueError(f"Requested weight version {global_steps}; available version is {self._global_steps}")
        return self._weights
