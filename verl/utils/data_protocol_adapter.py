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
RESPONSE_ALIGNED_KEYS = frozenset(
    {
        "response_mask",
        "old_log_probs",
        "ref_log_prob",
        "token_level_scores",
        "token_level_rewards",
        "advantages",
        "returns",
        "values",
    }
)
VERSION_KEYS = frozenset({"global_steps", "weight_version"})


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
    validate_training_tensordict(data.to_tensordict())


def validate_training_tensordict(data: TensorDict, *, require_versions: bool = False) -> None:
    """Validate the canonical V1 PPO tensor and version contract."""
    keys = set(data.keys())
    required = REQUIRED_TOKEN_KEYS & keys
    if required and required != REQUIRED_TOKEN_KEYS:
        missing = REQUIRED_TOKEN_KEYS - keys
        raise ValueError(f"Incomplete token contract; missing {sorted(missing)}")
    if "responses" in data:
        response_shape = data["responses"].shape
        for key in RESPONSE_ALIGNED_KEYS & keys:
            if data[key].shape != response_shape:
                raise ValueError(f"responses and {key} must have identical shapes")
    if "response_mask" in data:
        mask = data["response_mask"]
        if mask.dtype != torch.bool and not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("response_mask must be boolean or contain only 0/1 values")
    if "attention_mask" in data and "input_ids" in data:
        if data["attention_mask"].shape != data["input_ids"].shape:
            raise ValueError("input_ids and attention_mask must have identical shapes")
    if require_versions:
        missing = VERSION_KEYS - keys
        if missing:
            raise ValueError(f"Versioned V1 batch is missing {sorted(missing)}")
        for key in VERSION_KEYS:
            value = tu.unwrap_non_tensor_data(data[key])
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{key} must be a non-negative integer")


def _validate_dataproto(data: DataProto, required_keys: set[str]) -> None:
    missing = required_keys - set(data.batch.keys())
    if missing:
        raise ValueError(f"DataProto is missing required tensor keys: {sorted(missing)}")
    validate_response_alignment(data)
