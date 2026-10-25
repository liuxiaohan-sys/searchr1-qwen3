"""Pluggable actor-to-rollout weight transport.

Phase 4 starts with the colocated ``naive`` backend.  The registry preserves
the upstream extension point for NCCL/NIXL/HCCL backends without coupling them
to the legacy trainer.
"""

from .base import CheckpointEngine, CheckpointEngineRegistry, ColocatedCheckpointEngine

__all__ = ["CheckpointEngine", "CheckpointEngineRegistry", "ColocatedCheckpointEngine"]
