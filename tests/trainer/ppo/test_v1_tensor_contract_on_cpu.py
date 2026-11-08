import pytest
import torch

from verl.utils import tensordict_utils as tu
from verl.utils.data_protocol_adapter import validate_training_tensordict


def _batch():
    return tu.get_tensordict(
        {
            "input_ids": torch.ones(2, 7, dtype=torch.long),
            "attention_mask": torch.ones(2, 7, dtype=torch.long),
            "responses": torch.ones(2, 3, dtype=torch.long),
            "response_mask": torch.ones(2, 3, dtype=torch.bool),
            "old_log_probs": torch.zeros(2, 3),
            "ref_log_prob": torch.zeros(2, 3),
            "token_level_rewards": torch.zeros(2, 3),
            "advantages": torch.zeros(2, 3),
            "returns": torch.zeros(2, 3),
        },
        non_tensor_dict={"global_steps": 4, "weight_version": 3},
    )


def test_complete_versioned_contract():
    validate_training_tensordict(_batch(), require_versions=True)


@pytest.mark.parametrize("key", ["old_log_probs", "ref_log_prob", "advantages", "returns"])
def test_response_aligned_fields_reject_shifted_shape(key):
    batch = _batch()
    batch.set(key, torch.zeros(2, 2), inplace=False)
    with pytest.raises(ValueError, match=key):
        validate_training_tensordict(batch)


def test_non_binary_response_mask_is_rejected():
    batch = _batch()
    batch.set("response_mask", torch.full((2, 3), 2), inplace=False)
    with pytest.raises(ValueError, match="response_mask"):
        validate_training_tensordict(batch)


def test_missing_version_is_rejected_at_queue_boundary():
    batch = _batch()
    del batch["weight_version"]
    with pytest.raises(ValueError, match="weight_version"):
        validate_training_tensordict(batch, require_versions=True)
