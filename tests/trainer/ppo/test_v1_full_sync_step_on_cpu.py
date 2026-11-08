import torch

from verl.trainer.ppo.v1 import KVBatchMeta, PPOTrainerSync
from verl.utils import tensordict_utils as tu


class Queue:
    def __init__(self, batch): self.batch, self.cleared = batch, False
    def get_tensordict(self, meta): return self.batch
    def clear(self, meta): self.cleared = True


class Replay:
    def __init__(self, meta): self.meta = meta
    def sample(self, global_steps, partition_id, batch_size):
        return self.meta, {"sampled": batch_size}


class Actor:
    def compute_log_prob(self, batch):
        return {"old_log_probs": torch.zeros_like(batch["responses"], dtype=torch.float)}
    def update_actor(self, batch):
        assert "advantages" in batch and "returns" in batch
        return {"metrics": {"actor/loss": 1.0}}
    def update_weights(self, global_steps): self.version = global_steps


class Ref:
    def compute_ref_log_prob(self, batch):
        return {"ref_log_prob": torch.zeros_like(batch["responses"], dtype=torch.float)}


class Critic:
    def infer_batch(self, batch):
        return {"values": torch.zeros_like(batch["responses"], dtype=torch.float)}
    def train_batch(self, batch): return {"metrics": {"critic/loss": 2.0}}


def test_complete_tensor_dict_sync_ppo_transaction():
    batch = tu.get_tensordict(
        {
            "input_ids": torch.ones(2, 5, dtype=torch.long),
            "attention_mask": torch.ones(2, 5, dtype=torch.long),
            "responses": torch.ones(2, 2, dtype=torch.long),
            "response_mask": torch.ones(2, 2, dtype=torch.bool),
        },
        non_tensor_dict={"global_steps": 3, "weight_version": 2},
    )
    meta = KVBatchMeta("train", ["a", "b"], [{}, {}])
    queue, actor = Queue(batch), Actor()
    trainer = PPOTrainerSync(
        actor_wg=actor,
        critic_wg=Critic(),
        ref_wg=Ref(),
        queue=queue,
        replay_buffer=Replay(meta),
        reward_fn=lambda data: torch.ones_like(data["responses"], dtype=torch.float),
    )
    trainer.init()
    output, metrics = trainer.step(global_steps=3, partition_id="train", batch_size=2)
    assert queue.cleared
    assert actor.version == 3
    assert set(("old_log_probs", "ref_log_prob", "values", "advantages", "returns")) <= set(output.keys())
    assert metrics == {"sampled": 2, "critic/loss": 2.0, "actor/loss": 1.0}
