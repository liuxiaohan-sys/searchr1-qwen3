import asyncio
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Generator, Optional

import torch


@dataclass
class TensorMeta:
    name: str
    shape: torch.Size
    dtype: torch.dtype
    chunk_offset: int
    chunk_size: int
    offset: int | None = None


async def _ensure_async_iterator(iterator):
    if hasattr(iterator, "__aiter__"):
        async for item in iterator:
            yield item
    else:
        for item in iterator:
            yield item


async def split_weight_chunks(weights, bucket_size: int) -> AsyncGenerator[tuple[TensorMeta, torch.Tensor], None]:
    """Split named tensors into byte chunks no larger than a transport bucket."""
    async for name, weight in _ensure_async_iterator(weights):
        buffer = weight.contiguous().view(-1).view(torch.uint8)
        for chunk_offset in range(0, weight.nbytes, bucket_size):
            chunk_size = min(bucket_size, weight.nbytes - chunk_offset)
            yield TensorMeta(name, weight.shape, weight.dtype, chunk_offset, chunk_size), buffer[
                chunk_offset : chunk_offset + chunk_size
            ]


async def merge_weight_chunks(chunks, bucket_size: int):
    """Reassemble byte chunks emitted by a checkpoint transport."""
    merge_name = merge_weight = merge_buffer = None
    merge_offset = 0
    async for tensor_meta, chunk in chunks:
        nbytes = tensor_meta.shape.numel() * tensor_meta.dtype.itemsize
        if nbytes <= bucket_size:
            yield tensor_meta.name, chunk.view(tensor_meta.dtype).view(tensor_meta.shape)
            continue
        if merge_weight is None:
            merge_name = tensor_meta.name
            merge_weight = torch.empty(tensor_meta.shape, dtype=tensor_meta.dtype, device=chunk.device)
            merge_buffer = merge_weight.view(-1).view(torch.uint8)
            merge_offset = 0
        assert tensor_meta.name == merge_name and tensor_meta.chunk_offset == merge_offset
        merge_buffer[merge_offset : merge_offset + tensor_meta.chunk_size] = chunk
        merge_offset += tensor_meta.chunk_size
        if merge_offset == nbytes:
            yield merge_name, merge_weight
            merge_name = merge_weight = merge_buffer = None
            merge_offset = 0


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


class _RolloutWorkerGroup:
    """Dispatch checkpoint calls to existing standalone rollout handles."""

    def __init__(self, handles):
        self.workers = list(handles)
        self.world_size = len(self.workers)

    def _per_rank(self, value, rank, name):
        if isinstance(value, list | tuple):
            if len(value) != self.world_size:
                raise ValueError(f"rollout argument {name!r} has {len(value)} entries, expected {self.world_size}")
            return value[rank]
        return value

    def execute_checkpoint_engine(self, method, *args, **kwargs):
        """Match RayWorkerGroup's per-rank list dispatch contract."""
        calls = []
        for rank, worker in enumerate(self.workers):
            rank_method = self._per_rank(method, rank, "method")
            rank_args = tuple(self._per_rank(value, rank, f"arg_{index}") for index, value in enumerate(args))
            rank_kwargs = {name: self._per_rank(value, rank, name) for name, value in kwargs.items()}
            calls.append(worker.execute_checkpoint_engine.remote(rank_method, *rank_args, **rank_kwargs))
        return calls

    def update_weights(self, **kwargs):
        return [worker.update_weights.remote(None, **kwargs) for worker in self.workers]


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
        self._update_in_progress = False

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
        """Prepare and connect actor and standalone rollout transport ranks.

        Standalone rollout actors are not ``ActorRolloutRefWorker`` instances,
        so wrapping their handles in a RayWorkerGroup cannot bind the latter's
        checkpoint methods.  Invoke the explicit receiver API implemented by
        the rollout actor instead.
        """
        import ray

        workers = [worker for replica in self.replicas for worker in getattr(replica, "workers", [])]
        if not workers:
            raise RuntimeError(f"Checkpoint backend {self.backend!r} requires rollout worker handles")

        rollout = _RolloutWorkerGroup(workers)
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

    async def _finalize_process_group(self, rollout):
        import ray

        await asyncio.to_thread(
            ray.get,
            self.actor_wg.execute_checkpoint_engine(["finalize"] * self.actor_wg.world_size)
            + rollout.execute_checkpoint_engine(["finalize"] * rollout.world_size),
        )

    async def update_weights(self, global_steps: int | None = None):
        if self._update_in_progress:
            raise RuntimeError("A checkpoint weight update is already in progress")
        if global_steps is not None:
            if self._last_committed_version is not None and global_steps <= self._last_committed_version:
                raise ValueError(
                    f"Weight version must increase monotonically: {global_steps} <= {self._last_committed_version}"
                )
            self._last_submitted_version = global_steps

        self._update_in_progress = True
        rollout = None
        transport_started = False
        try:
            if self.backend != "naive":
                await self.abort_replicas()
                has_worker_handles = any(getattr(replica, "workers", None) for replica in self.replicas)
                if has_worker_handles:
                    # Retain the handles before any remote prepare call so a
                    # partially initialized topology can still be finalized.
                    workers = [
                        worker for replica in self.replicas for worker in getattr(replica, "workers", [])
                    ]
                    rollout = _RolloutWorkerGroup(workers)
                    transport_started = True
                    rollout = self._build_process_group()

            # Submit both sides before waiting: collectives cannot complete if
            # the sender is awaited before receivers enter the broadcast.
            actor_call = self.actor_wg.update_weights(global_steps=global_steps, mode=self.backend)
            if self.backend != "naive":
                if rollout is not None:
                    rollout_call = rollout.update_weights(global_steps=global_steps)
                else:
                    rollout_call = [
                        replica.update_weights(global_steps=global_steps, mode=self.backend)
                        for replica in self.replicas
                    ]
            else:
                rollout_call = []

            actor_result, _ = await asyncio.gather(
                self._resolve_calls(actor_call), self._resolve_calls(rollout_call)
            )
        finally:
            try:
                if transport_started:
                    await self._finalize_process_group(rollout)
            finally:
                if self.backend != "naive":
                    await self.wake_up_replicas()
                self._update_in_progress = False
                self._last_submitted_version = self._last_committed_version
        # A version is committed only after transfer, transport finalization,
        # and replica restoration all complete successfully.
        self._last_committed_version = global_steps
        self._last_submitted_version = global_steps
        return actor_result

    @staticmethod
    async def _resolve_calls(calls):
        """Resolve coroutines and Ray ObjectRefs without blocking the event loop."""
        if calls is None:
            return None
        if inspect.isawaitable(calls):
            return await calls
        if isinstance(calls, list | tuple):
            if any(inspect.isawaitable(item) for item in calls):
                return await asyncio.gather(*calls)
            if calls:
                try:
                    import ray

                    if all(isinstance(item, ray.ObjectRef) for item in calls):
                        return await asyncio.to_thread(ray.get, calls)
                except (ImportError, TypeError):
                    pass
        return calls

    async def wait_for_weight_update(self):
        return self._last_committed_version

    def close(self):
        """Compatibility hook for transports that own a background event loop."""
