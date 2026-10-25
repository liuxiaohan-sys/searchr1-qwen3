"""V1 trainer entry points available during the Search-R1 migration.

The complete upstream asynchronous trainer needs the upstream experimental
AgentLoop contract. This package starts with the synchronous, versioned queue
boundary so Search-R1 can migrate one execution concern at a time.
"""

from .trainer_sync import PPOTrainerSync, TransferQueueAdapter

__all__ = ["PPOTrainerSync", "TransferQueueAdapter"]
