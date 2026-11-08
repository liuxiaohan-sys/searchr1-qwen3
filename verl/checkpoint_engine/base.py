import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Any, Generator, Optional

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

    @classmethod
    def get(cls, backend: str):
        if backend not in cls._registry:
            raise ValueError(f"Checkpoint engine {backend!r} is not registered; available={sorted(cls._registry)}")
        return cls._registry[backend]


class CheckpointEngine(ABC):
    """Versioned transport boundary between a training engine and rollout."""

    support_publish_weights = False

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


class CheckpointEngineManager:
    """Coordinate versioned actor-to-rollout synchronization and replica memory state.

    The current repository ships the ``naive`` colocated transport.  The manager
    owns the stable Trainer API so NCCL/NIXL/HCCL transports can be registered
    without changing trainer control flow.
    """

    def __init__(self, config, actor_wg, replicas: list[Any] | None = None):
        from verl.utils.import_utils import import_external_libs

        self.config = config
        self.backend = config.backend
        import_external_libs(getattr(config, "custom_backend_module", None) or None)
        self.actor_wg = actor_wg
        self.replicas = list(replicas or [])
        self._last_submitted_version: int | None = None
        self._last_committed_version: int | None = None

    def add_replicas(self, replicas):
        self.replicas.extend(replicas)

    def remove_replicas(self, replicas):
        remove = set(replicas)
        self.replicas = [replica for replica in self.replicas if replica not in remove]

    async def sleep_replicas(self):
        await asyncio.gather(*(replica.sleep() for replica in self.replicas))

    async def wake_up_replicas(self):
        await asyncio.gather(*(replica.wake_up() for replica in self.replicas))

    async def abort_replicas(self, reject_request: bool = False):
        operations = []
        for replica in self.replicas:
            abort = getattr(replica, "abort_all_requests", None)
            operations.append(abort(reject_request=reject_request) if abort is not None else replica.sleep())
        await asyncio.gather(*operations)

    def _build_process_group(self):
        """Prepare and connect actor and standalone rollout transport ranks."""
        import ray

        from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup

        workers = [worker for replica in self.replicas for worker in getattr(replica, "workers", [])]
        if not workers:
            raise RuntimeError(f"Checkpoint backend {self.backend!r} requires rollout worker handles")
        rollout = RayWorkerGroup(worker_handles=workers, ray_cls_with_init=RayClassWithInitArgs(cls=None))
        backend_cls = CheckpointEngineRegistry.get(self.backend)
        actor_size, rollout_size = self.actor_wg.world_size, rollout.world_size
        metadata = ray.get(
            self.actor_wg.execute_checkpoint_engine(["prepare"] * actor_size)
            + rollout.execute_checkpoint_engine(["prepare"] * rollout_size)
        )
        actor_kwargs, rollout_kwargs = backend_cls.build_topology(actor_size, rollout_size, metadata)
        for name, values in actor_kwargs.items():
            if len(values) != actor_size:
                raise ValueError(f"actor topology {name!r} has {len(values)} entries, expected {actor_size}")
        for name, values in rollout_kwargs.items():
            if len(values) != rollout_size:
                raise ValueError(f"rollout topology {name!r} has {len(values)} entries, expected {rollout_size}")
        actor_kwargs["method"] = ["init_process_group"] * actor_size
        rollout_kwargs["method"] = ["init_process_group"] * rollout_size
        ray.get(
            self.actor_wg.execute_checkpoint_engine(**actor_kwargs)
            + rollout.execute_checkpoint_engine(**rollout_kwargs)
        )
        return rollout

    async def update_weights(self, global_steps: int | None = None):
        if global_steps is not None:
            if self._last_submitted_version is not None and global_steps <= self._last_submitted_version:
                raise ValueError(
                    f"Weight version must increase monotonically: {global_steps} <= {self._last_submitted_version}"
                )
            self._last_submitted_version = global_steps
        if self.backend != "naive":
            await self.abort_replicas()
            has_worker_handles = any(getattr(replica, "workers", None) for replica in self.replicas)
            rollout = self._build_process_group() if has_worker_handles else None
        result = self.actor_wg.update_weights(global_steps=global_steps, mode=self.backend)
        actor_result = await result if inspect.isawaitable(result) else result
        if isinstance(actor_result, list | tuple) and any(
            inspect.isawaitable(item) for item in actor_result
        ):
            actor_result = await asyncio.gather(*actor_result)
        if self.backend != "naive":
            if rollout is not None:
                import ray

                rollout_results = rollout.update_weights(global_steps=global_steps)
                ray.get(rollout_results)
                ray.get(
                    self.actor_wg.execute_checkpoint_engine(["finalize"] * self.actor_wg.world_size)
                    + rollout.execute_checkpoint_engine(["finalize"] * rollout.world_size)
                )
            else:
                # Lightweight plugin/test replicas may expose a direct update API.
                await asyncio.gather(
                    *(replica.update_weights(global_steps=global_steps, mode=self.backend) for replica in self.replicas)
                )
            await self.wake_up_replicas()
        self._last_committed_version = global_steps
        return actor_result

    async def wait_for_weight_update(self):
        return self._last_committed_version

    def close(self):
        """Compatibility hook for transports that own a background event loop."""
