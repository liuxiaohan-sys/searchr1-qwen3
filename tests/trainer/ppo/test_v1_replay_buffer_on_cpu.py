import pytest

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer


class FakeStore:
    def __init__(self, items):
        self.items = items
        self.cleared = []

    def kv_list(self):
        return self.items

    def kv_clear(self, *, partition_id, keys):
        self.cleared.append((partition_id, keys))


def _store():
    return FakeStore(
        {
            "train": {
                "b": {"is_prompt": True, "status": "finished", "global_steps": 3},
                "a": {"is_prompt": True, "status": "finished", "global_steps": 1},
                "a_0_0": {"global_steps": 1, "reward": 1.0},
                "b_0_0": {"global_steps": 3, "reward": 2.0},
            }
        }
    )


def test_samples_oldest_group_and_clears_prompt_metadata():
    store = _store()
    meta, metrics = ReplayBuffer(store, timeout=0).sample(4, "train", 1)
    assert meta.keys == ["a_0_0"]
    assert meta.partition_id == "train"
    assert store.cleared == [("train", ["a"])]
    assert metrics["training/off_policy/sampled_staleness/mean"] == 3


def test_rejects_future_policy_samples():
    store = _store()
    with pytest.raises(ValueError, match="future weight version"):
        ReplayBuffer(store, timeout=0).sample(0, "train", 1)


def test_timeout_when_not_enough_terminal_groups():
    store = FakeStore({"train": {"a": {"is_prompt": True, "status": "running", "global_steps": 1}}})
    with pytest.raises(TimeoutError):
        ReplayBuffer(store, timeout=0).sample(2, "train", 1)


def test_terminal_group_without_trajectory_is_rejected():
    store = FakeStore({"train": {"a": {"is_prompt": True, "status": "failure", "global_steps": 1}}})
    with pytest.raises(RuntimeError, match="no materializable trajectories"):
        ReplayBuffer(store, timeout=0).sample(2, "train", 1)
