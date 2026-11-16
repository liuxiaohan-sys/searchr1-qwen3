from types import SimpleNamespace

import pytest

from verl.trainer.ppo.v1 import PPOTrainerColocateAsync, PPOTrainerSeparateAsync, ReplayBufferAsync


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


class CheckpointManager:
    def __init__(self, backend):
        self.config = SimpleNamespace(backend=backend)
        self.events = []

    async def update_weights(self, step): self.events.append(("update", step))
    async def wake_up_replicas(self): self.events.append("wake")
    async def abort_replicas(self): self.events.append("abort")
    async def sleep_replicas(self): self.events.append("sleep")
    async def wait_for_weight_update(self): self.events.append("wait")


def test_async_modes_validate_transport_boundaries():
    colocate = PPOTrainerColocateAsync(checkpoint_manager=CheckpointManager("nccl"))
    with pytest.raises(ValueError, match="naive"):
        colocate._validate_mode()

    separate = PPOTrainerSeparateAsync(checkpoint_manager=CheckpointManager("naive"))
    with pytest.raises(ValueError, match="standalone"):
        separate._validate_mode()


def test_colocate_step_quiesces_and_resumes_replicas(monkeypatch):
    manager = CheckpointManager("naive")
    events = manager.events

    class Replay:
        def sample(self, *args):
            events.append("sample")
            return "meta", {"sampled": True}

    trainer = PPOTrainerColocateAsync(checkpoint_manager=manager, replay_buffer=Replay())
    monkeypatch.setattr(
        trainer,
        "train_kv_batch",
        lambda meta, **kwargs: ("batch", {"trained": True}),
    )
    assert trainer.step(global_steps=1, partition_id="train", batch_size=1) == (
        "batch", {"sampled": True, "trained": True}
    )
    assert events == ["sample", "abort", "sleep", "wake"]


def test_separate_parameter_sync_period():
    trainer = PPOTrainerSeparateAsync(parameter_sync_step=3)
    assert not trainer.should_sync_weights(1)
    assert not trainer.should_sync_weights(2)
    assert trainer.should_sync_weights(3)


def test_separate_decoupled_ppo_uses_rollout_behavior_log_probs():
    torch = pytest.importorskip("torch")
    trainer = PPOTrainerSeparateAsync(parameter_sync_step=2)
    rollout_log_probs = torch.randn(2, 3)
    batch = {"rollout_log_probs": rollout_log_probs}
    assert torch.equal(trainer.compute_old_log_prob(batch)["old_log_probs"], rollout_log_probs)

    with pytest.raises(RuntimeError, match="rollout_log_probs"):
        trainer.compute_old_log_prob({})
