"""TensorDict/TransferQueue V1 trainer entry points for Search-R1."""

from .agent_loop_tq import AgentLoopManagerTQ, AgentLoopTQProducer, AgentLoopWorkerTQ
from .replay_buffer import KVBatchMeta, ReplayBuffer, ReplayBufferAsync
from .reward_loop import RewardLoopManager
from .trainer_async import PPOTrainerColocateAsync, PPOTrainerSeparateAsync
from .trainer_base import PPOTrainer
from .trainer_sync import PPOTrainerSync, TransferQueueAdapter

_TRAINER_REGISTRY = {
    "sync": PPOTrainerSync,
    "colocate_async": PPOTrainerColocateAsync,
    "separate_async": PPOTrainerSeparateAsync,
}
_KNOWN_TRAINER_MODES = frozenset({"sync", "colocate_async", "separate_async"})


def get_trainer_cls(name: str):
    """Resolve a V1 trainer without silently falling back to the V0 trainer."""
    if name not in _KNOWN_TRAINER_MODES:
        raise ValueError(f"Unknown V1 trainer mode {name!r}; expected one of {sorted(_KNOWN_TRAINER_MODES)}")
    return _TRAINER_REGISTRY[name]


__all__ = [
    "PPOTrainerSync",
    "PPOTrainer",
    "TransferQueueAdapter",
    "KVBatchMeta",
    "ReplayBuffer",
    "ReplayBufferAsync",
    "AgentLoopTQProducer",
    "AgentLoopManagerTQ",
    "AgentLoopWorkerTQ",
    "RewardLoopManager",
    "PPOTrainerColocateAsync",
    "PPOTrainerSeparateAsync",
    "get_trainer_cls",
]
