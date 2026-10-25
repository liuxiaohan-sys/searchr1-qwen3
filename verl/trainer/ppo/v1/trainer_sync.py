"""Minimal V1 synchronous trainer boundary for the legacy Search-R1 stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from tensordict import TensorDict

from verl import DataProto
from verl.utils.data_protocol_adapter import dataproto_to_tensordict, tensordict_to_dataproto, validate_response_alignment


class QueueClient(Protocol):
    def kv_batch_put(self, *, keys, fields, tags, partition_id: str): ...

    def kv_batch_get_by_meta(self, meta): ...

    def close(self): ...


class TransferQueueAdapter:
    """Thin, injectable wrapper around TransferQueue's KV batch API."""

    def __init__(self, client: Optional[QueueClient] = None):
        if client is None:
            import transfer_queue as tq

            tq.init()
            client = tq
        self.client = client

    def put(self, batch: DataProto, *, keys: list[str], tags: list[dict[str, Any]], partition_id: str = "train"):
        if len(keys) != len(batch) or len(tags) != len(batch):
            raise ValueError("TransferQueue keys, tags and batch rows must have the same length")
        validate_response_alignment(batch)
        self.client.kv_batch_put(keys=keys, fields=dataproto_to_tensordict(batch), tags=tags, partition_id=partition_id)

    def get(self, meta: Any) -> DataProto:
        fields = self.client.kv_batch_get_by_meta(meta)
        if not isinstance(fields, TensorDict):
            raise TypeError(f"TransferQueue returned {type(fields)!r}; expected TensorDict")
        return tensordict_to_dataproto(fields)

    def close(self):
        self.client.close()


@dataclass
class PPOTrainerSync:
    """One-batch V1 control loop used before asynchronous overlap is enabled."""

    actor_rollout_wg: Any
    queue: Optional[TransferQueueAdapter] = None

    def train_batch(self, batch: DataProto, *, global_steps: int) -> DataProto:
        validate_response_alignment(batch)
        result = self.actor_rollout_wg.update_actor(batch)
        self.actor_rollout_wg.update_weights(global_steps=global_steps)
        return result

    def consume_once(self, meta: Any, *, global_steps: int) -> DataProto:
        if self.queue is None:
            raise RuntimeError("consume_once requires a TransferQueueAdapter")
        return self.train_batch(self.queue.get(meta), global_steps=global_steps)
