import pytest
from omegaconf import OmegaConf

from verl.trainer.main_ppo import validate_execution_mode
from verl.trainer.ppo.v1 import PPOTrainerSync, get_trainer_cls


def _config(*, use_v1=False, worker_impl="auto", trainer_mode="sync", checkpoint_backend="nccl"):
    return OmegaConf.create(
        {
            "trainer": {
                "use_v1": use_v1,
                "use_legacy_worker_impl": worker_impl,
                "v1": {"trainer_mode": trainer_mode},
            },
            "actor_rollout_ref": {"rollout": {"checkpoint_engine": {"backend": checkpoint_backend}}},
        }
    )


@pytest.mark.parametrize("worker_impl", ["auto", "enable", "disable"])
def test_v0_keeps_all_worker_routing_options(worker_impl):
    validate_execution_mode(_config(worker_impl=worker_impl))


def test_v1_requires_unified_workers():
    with pytest.raises(ValueError, match="requires trainer.use_legacy_worker_impl=disable"):
        validate_execution_mode(_config(use_v1=True, worker_impl="auto"))


def test_sync_v1_factory_is_wired():
    config = _config(use_v1=True, worker_impl="disable")
    validate_execution_mode(config)
    assert get_trainer_cls("sync") is PPOTrainerSync


@pytest.mark.parametrize("trainer_mode", ["colocate_async", "separate_async"])
def test_async_modes_are_registered(trainer_mode):
    validate_execution_mode(_config(use_v1=True, worker_impl="disable", trainer_mode=trainer_mode))
    assert get_trainer_cls(trainer_mode).__name__.lower().endswith(trainer_mode.replace("_", ""))


def test_unknown_v1_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown V1 trainer mode"):
        get_trainer_cls("unknown")


def test_separate_async_rejects_colocated_naive_weight_transport():
    with pytest.raises(ValueError, match="standalone checkpoint transport"):
        validate_execution_mode(
            _config(
                use_v1=True,
                worker_impl="disable",
                trainer_mode="separate_async",
                checkpoint_backend="naive",
            )
        )
