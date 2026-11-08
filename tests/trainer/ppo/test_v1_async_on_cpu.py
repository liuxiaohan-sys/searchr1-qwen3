import pytest

from verl.trainer.ppo.v1 import PPOTrainerColocateAsync, ReplayBufferAsync


class Store:
    def __init__(self):
        self.data = {"train": {"old": {"is_prompt": True, "status": "finished", "global_steps": 1}}}
        self.cleared = []
    def kv_list(self): return self.data
    def kv_clear(self, *, partition_id, keys):
        self.cleared.extend(keys)
        for key in keys:
            self.data[partition_id].pop(key, None)


def test_async_drop_evicts_stale_groups_before_polling():
    store = Store()
    replay = ReplayBufferAsync(store, timeout=0, max_off_policy_threshold=2, strategy="drop")
    with pytest.raises(TimeoutError):
        replay.sample(5, "train", 1)
    assert store.cleared == ["old"]


def test_async_rollout_runs_in_background_and_drains():
    class Manager:
        def generate_sequences(self, prompts): self.prompts = prompts
    manager = Manager()
    trainer = PPOTrainerColocateAsync(num_warmup_batches=1)
    trainer.submit_rollout(manager, "batch")
    trainer.close()
    assert manager.prompts == "batch"
