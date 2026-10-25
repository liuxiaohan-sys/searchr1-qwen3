"""Explicit boundary between legacy :class:`DataProto` and model-engine TensorDict.

Keep conversions here while the PPO trainer is migrated.  New worker code must
accept TensorDict; legacy Search-R1 code may keep DataProto until phase 6.
"""

from typing import Iterable, Optional

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.utils import tensordict_utils as tu


REQUIRED_TOKEN_KEYS = frozenset({"input_ids", "attention_mask", "responses"})


def dataproto_to_tensordict(data: DataProto, *, required_keys: Iterable[str] = ()) -> TensorDict:
    """Convert at the sole legacy-to-engine boundary and validate tensor keys."""
    _validate_dataproto(data, set(required_keys))
    return data.to_tensordict()


def tensordict_to_dataproto(data: TensorDict, *, meta_info: Optional[dict] = None) -> DataProto:
    """Return legacy output without losing tensors or batched non-tensor fields.

    Engine metrics are intentionally returned as ``meta_info``: they are global
    worker results rather than per-sample columns.
    """
    tensors: dict[str, torch.Tensor] = {}
    non_tensors: dict[str, np.ndarray] = {}
    merged_meta = dict(meta_info or {})
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            tensors[key] = value
            continue
        value = tu.unwrap_non_tensor_data(value)
        if key == "metrics":
            merged_meta["metrics"] = value
        elif isinstance(value, list):
            non_tensors[key] = np.asarray(value, dtype=object)
        else:
            merged_meta[key] = value
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=merged_meta)


def validate_response_alignment(data: DataProto) -> None:
    """Catch the most dangerous silent PPO shape mismatch before dispatch."""
    required = REQUIRED_TOKEN_KEYS & set(data.batch.keys())
    if required and required != REQUIRED_TOKEN_KEYS:
        missing = REQUIRED_TOKEN_KEYS - set(data.batch.keys())
        raise ValueError(f"Incomplete token contract; missing {sorted(missing)}")
    if "responses" in data.batch and "old_log_probs" in data.batch:
        if data.batch["responses"].shape != data.batch["old_log_probs"].shape:
            raise ValueError("responses and old_log_probs must have identical shapes")
    if "responses" in data.batch and "ref_log_prob" in data.batch:
        if data.batch["responses"].shape != data.batch["ref_log_prob"].shape:
            raise ValueError("responses and ref_log_prob must have identical shapes")


def _validate_dataproto(data: DataProto, required_keys: set[str]) -> None:
    missing = required_keys - set(data.batch.keys())
    if missing:
        raise ValueError(f"DataProto is missing required tensor keys: {sorted(missing)}")
    validate_response_alignment(data)
