from pathlib import Path

from verl.trainer.ppo.v1 import PPOTrainerSync


class Worker:
    def __init__(self): self.calls = []
    def save_checkpoint(self, *args): self.calls.append(("save", args[0]))
    def load_checkpoint(self, *args): self.calls.append(("load", args[0]))


def test_v1_checkpoint_roundtrip(tmp_path: Path):
    actor, critic = Worker(), Worker()
    trainer = PPOTrainerSync(actor_wg=actor, critic_wg=critic, validation_fn=lambda: {"score": 1})
    trainer.save_checkpoint(str(tmp_path), 4)
    assert trainer.load_checkpoint(str(tmp_path)) == 4
    assert trainer.validate() == {"score": 1}
    assert (tmp_path / "global_step_4" / "v1_state.json").exists()
