import pytest
import torch

from verl.trainer.ppo.v1.replay_buffer import KVBatchMeta
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync, TransferQueueAdapter
from verl.utils import tensordict_utils as tu


def _batch():
    return tu.get_tensordict(
        {
            "input_ids": torch.ones(2, 5, dtype=torch.long),
            "attention_mask": torch.ones(2, 5, dtype=torch.long),
            "responses": torch.ones(2, 2, dtype=torch.long),
            "response_mask": torch.ones(2, 2, dtype=torch.bool),
        },
        non_tensor_dict={"global_steps": 2, "weight_version": 2},
    )


class FakeQueue:
    def __init__(self):
        self.fields = None

    def kv_batch_put(self, *, keys, fields, tags, partition_id):
        self.fields = fields
        self.put_args = (keys, tags, partition_id)

    def kv_batch_get_by_meta(self, meta):
        return self.fields

    def close(self):
        self.closed = True

    def kv_clear(self, *, keys, partition_id):
        self.cleared = (keys, partition_id)


class FakeActor:
    def __init__(self):
        self.versions = []

    def update_actor(self, batch):
        self.batch = batch
        return batch

    def update_weights(self, *, global_steps):
        self.versions.append(global_steps)


def test_native_tensordict_queue_and_sync_trainer_roundtrip():
    backend = FakeQueue()
    queue = TransferQueueAdapter(backend)
    batch = _batch()
    queue.put_tensordict(batch, keys=["a", "b"], tags=[{}, {}])
    actor = FakeActor()
    meta = KVBatchMeta(partition_id="train", keys=["a", "b"], tags=[{}, {}])
    output = PPOTrainerSync(actor_rollout_wg=actor, queue=queue).consume_once(meta, global_steps=2)
    assert output is batch
    assert actor.batch is batch
    assert actor.versions == [2]
    assert backend.put_args == (["a", "b"], [{}, {}], "train")
    assert backend.cleared == (["a", "b"], "train")


def test_failed_update_retains_queue_data_for_recovery():
    class FailingActor(FakeActor):
        def update_actor(self, batch):
            raise RuntimeError("update failed")

    backend = FakeQueue()
    queue = TransferQueueAdapter(backend)
    queue.put_tensordict(_batch(), keys=["a", "b"], tags=[{}, {}])
    meta = KVBatchMeta(partition_id="train", keys=["a", "b"], tags=[{}, {}])
    with pytest.raises(RuntimeError, match="update failed"):
        PPOTrainerSync(actor_rollout_wg=FailingActor(), queue=queue).consume_once(meta, global_steps=2)
    assert not hasattr(backend, "cleared")
