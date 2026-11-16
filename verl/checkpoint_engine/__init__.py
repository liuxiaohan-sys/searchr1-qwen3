"""Pluggable actor-to-rollout weight transport.

Phase 4 starts with the colocated ``naive`` backend.  The registry preserves
the upstream extension point for NCCL/NIXL/HCCL backends without coupling them
to the legacy trainer.
"""

from .base import CheckpointEngine, CheckpointEngineManager, CheckpointEngineRegistry, ColocatedCheckpointEngine, TensorMeta

__all__ = ["CheckpointEngine", "CheckpointEngineManager", "CheckpointEngineRegistry", "ColocatedCheckpointEngine", "TensorMeta"]

# NCCL is optional so CPU-only environments can still import the package and
# use the colocated backend. Importing this module performs backend registration.
try:
    from .nccl_checkpoint_engine import NCCLCheckpointEngine

    __all__.append("NCCLCheckpointEngine")
except ImportError:
    NCCLCheckpointEngine = None
