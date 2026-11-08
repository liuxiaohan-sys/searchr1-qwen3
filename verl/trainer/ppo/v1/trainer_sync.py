"""Synchronous TensorDict/TransferQueue trainer and V0 compatibility adapter."""

from __future__ import annotations

from typing import Any, Optional, Protocol

from tensordict import TensorDict

from verl import DataProto
from verl.utils.data_protocol_adapter import (
    dataproto_to_tensordict,
    tensordict_to_dataproto,
    validate_response_alignment,
    validate_training_tensordict,
)

from .trainer_base import PPOTrainer


class QueueClient(Protocol):
    def kv_batch_put(self, *, keys, fields, tags, partition_id: str): ...

    def kv_batch_get_by_meta(self, meta): ...

    def kv_clear(self, *, keys, partition_id: str): ...

    def close(self): ...


class TransferQueueAdapter:
    """Thin, injectable wrapper around TransferQueue's KV batch API."""

    def __init__(self, client: Optional[QueueClient] = None):
        if client is None:
            import transfer_queue as tq

            tq.init()
            client = tq
        self.client = client

    def put_tensordict(
        self,
        batch: TensorDict,
        *,
        keys: list[str],
        tags: list[dict[str, Any]],
        partition_id: str = "train",
        require_versions: bool = True,
    ):
        if len(keys) != batch.batch_size[0] or len(tags) != batch.batch_size[0]:
            raise ValueError("TransferQueue keys, tags and batch rows must have the same length")
        validate_training_tensordict(batch, require_versions=require_versions)
        self.client.kv_batch_put(keys=keys, fields=batch, tags=tags, partition_id=partition_id)

    def put(self, batch: DataProto, *, keys: list[str], tags: list[dict[str, Any]], partition_id: str = "train"):
        """V0 compatibility wrapper; V1 callers must use put_tensordict."""
        validate_response_alignment(batch)
        self.put_tensordict(
            dataproto_to_tensordict(batch),
            keys=keys,
            tags=tags,
            partition_id=partition_id,
            require_versions=False,
        )

    def get_tensordict(self, meta: Any) -> TensorDict:
        fields = self.client.kv_batch_get_by_meta(meta)
        if not isinstance(fields, TensorDict):
            raise TypeError(f"TransferQueue returned {type(fields)!r}; expected TensorDict")
        validate_training_tensordict(fields, require_versions=True)
        return fields

    def get(self, meta: Any) -> DataProto:
        """V0 compatibility wrapper; V1 callers must use get_tensordict."""
        return tensordict_to_dataproto(self.get_tensordict(meta))

    def close(self):
        self.client.close()

    def clear(self, meta: Any) -> None:
        """Clear a successfully consumed batch; failed steps intentionally retain data."""
        self.client.kv_clear(keys=list(meta.keys), partition_id=meta.partition_id)


class PPOTrainerSync(PPOTrainer):
    """Synchronous V1 trainer; rollout and training advance one policy version at a time."""

    def __init__(self, *args, actor_rollout_wg=None, **kwargs):
        if actor_rollout_wg is not None and "actor_wg" not in kwargs:
            kwargs["actor_wg"] = actor_rollout_wg
        super().__init__(*args, **kwargs)
        self.actor_rollout_wg = self.actor_wg

    def train_batch(self, batch: TensorDict, *, global_steps: int) -> TensorDict:
        validate_training_tensordict(batch, require_versions=True)
        result = self.actor_rollout_wg.update_actor(batch)
        self.actor_rollout_wg.update_weights(global_steps=global_steps)
        return result

    def consume_once(self, meta: Any, *, global_steps: int) -> TensorDict:
        if self.queue is None:
            raise RuntimeError("consume_once requires a TransferQueueAdapter")
        result = self.train_batch(self.queue.get_tensordict(meta), global_steps=global_steps)
        self.queue.clear(meta)
        return result
