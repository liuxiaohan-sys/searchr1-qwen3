import logging
import os
from functools import partial
from copy import deepcopy
from typing import Any, Optional

import torch
from omegaconf import DictConfig, open_dict
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.data_protocol_adapter import dataproto_to_tensordict, tensordict_to_dataproto
from verl.utils.device import get_device_name, get_torch_device, set_expandable_segments
from verl.utils.distributed import initialize_global_process_group_ray, set_numa_affinity
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig
from verl.workers.config import ActorConfig, HFModelConfig, RolloutConfig, TrainingWorkerConfig
from verl.workers.roles.utils.losses import ppo_loss
from verl.workers.roles.utils.padding import left_right_2_no_padding, no_padding_2_padding
from verl.workers.rollout.base import BaseRollout, get_rollout_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _mean_metrics(outputs: list[TensorDict]) -> dict[str, float]:
    """Aggregate replicated engine metrics without exposing per-rank tensors."""
    values: dict[str, list[float]] = {}
    for output in outputs:
        metrics = tu.unwrap_non_tensor_data(output.get("metrics", {}))
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean().item()
            if isinstance(value, (int, float)):
                values.setdefault(key, []).append(float(value))
    return {key: sum(items) / len(items) for key, items in values.items()}


class TrainingWorker(Worker, DistProfilerExtension):
    """Backend-independent training facade used by Actor, Critic and Ref.

    This is the phase-1 compatibility implementation.  It deliberately accepts
    TensorDict at its public boundary, while the legacy PPO trainer remains on
    DataProto until the phase-2 adapter is introduced.
    """

    def __init__(self, config: TrainingWorkerConfig):
        Worker.__init__(self)
        from verl.workers.engine import BaseEngine, EngineRegistry

        initialize_global_process_group_ray(timeout_second=None)
        set_numa_affinity()
        self.config = config
        self.model_config = config.model_config
        self.engine_config = config.engine_config
        self.optimizer_config = config.optimizer_config
        self.checkpoint_config = config.checkpoint_config
        if self.engine_config is None:
            if config.auto_select_engine_optim_fn is None:
                raise ValueError("TrainingWorker needs engine_config or auto_select_engine_optim_fn")
            self.engine_config, self.optimizer_config = config.auto_select_engine_optim_fn(
                self.model_config, get_device_name()
            )
        self.profiler_config = config.profiler_config
        tool_config = self.profiler_config.tool_config if self.profiler_config is not None else None
        DistProfilerExtension.__init__(self, DistProfiler(rank=self.rank, config=self.profiler_config, tool_config=tool_config))
        self.engine: BaseEngine = EngineRegistry.new(
            model_type=config.model_type,
            backend=self.engine_config.strategy,
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )
        self._register_dispatch_collect_info(
            mesh_name="train",
            dp_rank=self.engine.get_data_parallel_rank(),
            is_collect=self.engine.is_mp_src_rank_with_outputs(),
        )
        self.loss_fn = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset(self):
        self.engine.initialize()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        # Current engines do not yet expose independent gradient offload.
        del grad
        self.engine.to(get_device_name() if device == "device" else device, model=model, optimizer=optimizer)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"))
    def train_batch(self, data: TensorDict, step_scheduler: bool = True):
        if self.loss_fn is None:
            raise RuntimeError("set_loss_fn must be called before train_batch")
        with self.engine.train_mode():
            output = self.engine.train_batch(data, loss_function=self.loss_fn)
        if self.engine.is_mp_src_rank_with_outputs():
            if step_scheduler:
                output.setdefault("metrics", {})["lr"] = self.engine.lr_scheduler_step()
            return tu.get_tensordict(output.get("model_output", {}), {"metrics": output["metrics"]}).cpu()
        return None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def step_scheduler(self):
        """Advance the scheduler once per PPO update, not once per micro batch."""
        return self.engine.lr_scheduler_step()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"))
    def infer_batch(self, data: TensorDict):
        with self.engine.eval_mode():
            output = self.engine.infer_batch(data, loss_function=self.loss_fn)
        if self.engine.is_mp_src_rank_with_outputs():
            return tu.get_tensordict(output.get("model_output", {}), {"metrics": output.get("metrics", {})}).cpu()
        return None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        return self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        return self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """Unified-engine actor plus rollout and optional reference policy."""

    actor_worker_cls = TrainingWorker
    ref_worker_cls = TrainingWorker

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)
        self.config = config
        self.role = role
        self.actor: Optional[TrainingWorker] = None
        self.ref: Optional[TrainingWorker] = None
        self.rollout: Optional[BaseRollout] = None
        if role not in {"actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"}:
            raise ValueError(f"Unsupported role: {role}")
        role_config = config.actor if "actor" in role else config.rollout if "rollout" in role else config.ref
        profiler_config = omega_conf_to_dataclass(role_config.get("profiler", {}), dataclass_type=ProfilerConfig)
        tool_config = profiler_config.tool_config
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)
        actor_config: Optional[ActorConfig] = None
        if "actor" in self.role:
            actor_config = omega_conf_to_dataclass(self.config.actor)
            actor_config.model_config = model_config

        if "ref" in self.role:
            with open_dict(self.config.ref):
                self.config.ref.ppo_mini_batch_size = self.config.actor.ppo_mini_batch_size
                self.config.ref.ppo_micro_batch_size = self.config.ref.pop("log_prob_micro_batch_size", None)
                self.config.ref.ppo_micro_batch_size_per_gpu = self.config.ref.pop(
                    "log_prob_micro_batch_size_per_gpu", None
                )
                self.config.ref.use_dynamic_bsz = self.config.ref.pop("log_prob_use_dynamic_bsz", False)
                self.config.ref.ppo_max_token_len_per_gpu = self.config.ref.pop("log_prob_max_token_len_per_gpu", None)
            ref_config: ActorConfig = omega_conf_to_dataclass(self.config.ref)
            ref_config.model_config = model_config
            # Historical ref YAMLs do not carry the new ``engine`` section.
            # Reuse actor backend settings while keeping a distinct config object.
            if not getattr(ref_config.engine, "strategy", None):
                if actor_config is None:
                    actor_config = omega_conf_to_dataclass(self.config.actor)
                    actor_config.model_config = model_config
                ref_config.engine = deepcopy(actor_config.engine)
            self.ref = self.ref_worker_cls(
                TrainingWorkerConfig(
                    model_type=ref_config.model_config.get("model_type", "language_model"),
                    model_config=ref_config.model_config,
                    engine_config=ref_config.engine,
                    optimizer_config=ref_config.optim,
                    checkpoint_config=ref_config.checkpoint,
                )
            )
            self.ref.reset()
            self.ref.engine.to("cpu")
            self.set_dispatch_collect(mesh_name="ref", **self.ref.get_dispatch_collect())

        if "actor" in self.role:
            assert actor_config is not None
            self.actor = self.actor_worker_cls(
                TrainingWorkerConfig(
                    model_type=actor_config.model_config.get("model_type", "language_model"),
                    model_config=actor_config.model_config,
                    engine_config=actor_config.engine,
                    optimizer_config=actor_config.optim,
                    checkpoint_config=actor_config.checkpoint,
                )
            )
            self.actor.set_loss_fn(partial(ppo_loss, config=actor_config))
            self.actor.reset()
            self.actor.engine.to("cpu")
            self.set_dispatch_collect(mesh_name="actor", **self.actor.get_dispatch_collect())

        if "rollout" in self.role:
            rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
            infer_tp = rollout_config.tensor_model_parallel_size * rollout_config.data_parallel_size
            infer_pp = rollout_config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            if self.world_size % infer_world_size != 0:
                raise ValueError(f"rollout world_size {self.world_size} is not divisible by {infer_world_size}")
            rollout_device_mesh = init_device_mesh(
                get_device_name(),
                mesh_shape=(self.world_size // infer_world_size, infer_tp, infer_pp),
                mesh_dim_names=["dp", "infer_tp", "infer_pp"],
            )
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().manual_seed(rollout_device_mesh["dp"].get_local_rank() + 1000)
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
            rollout_cls: type[BaseRollout] = get_rollout_class(rollout_config.name, rollout_config.mode)
            self.rollout = rollout_cls(config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh)
            self.base_sync_done = "dummy" not in self.config.rollout.load_format
            self.layered_summon = self.config.rollout.get("layered_summon", False)
            checkpoint_config = rollout_config.checkpoint_engine
            self.checkpoint_engine = CheckpointEngineRegistry.new(
                checkpoint_config.backend,
                bucket_size=checkpoint_config.update_weights_bucket_megabytes << 20,
                is_master=self.rank == 0,
                **checkpoint_config.engine_kwargs.get(checkpoint_config.backend, {}),
            )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        tensor_data = left_right_2_no_padding(dataproto_to_tensordict(data))
        output = self.ref.infer_batch(tensor_data)
        if output is None:
            return None
        log_probs = no_padding_2_padding(output["log_probs"], tensor_data)
        return DataProto.from_dict(tensors={"ref_log_prob": log_probs.float()})

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data: DataProto):
        tensor_data = left_right_2_no_padding(dataproto_to_tensordict(data))
        output = self.actor.infer_batch(tensor_data)
        if output is None:
            return None
        tensors = {"old_log_probs": no_padding_2_padding(output["log_probs"], tensor_data).float()}
        entropy = output.get("entropy")
        if entropy is not None:
            tensors["entropy"] = no_padding_2_padding(entropy, tensor_data).float()
        return DataProto.from_dict(tensors=tensors)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        # The old PPO worker performs ``ppo_epochs`` passes over minibatches. A
        # single Engine.train_batch call silently changed this contract, so keep
        # the loop at the RPC boundary until the V1 trainer owns batching.
        mini_batch_size = self.config.actor.ppo_mini_batch_size
        if not isinstance(mini_batch_size, int) or mini_batch_size <= 0:
            raise ValueError(f"Invalid actor.ppo_mini_batch_size: {mini_batch_size!r}")
        outputs = []
        for _ in range(self.config.actor.ppo_epochs):
            for mini_batch in data.split(mini_batch_size):
                tensor_data = left_right_2_no_padding(
                    dataproto_to_tensordict(mini_batch, required_keys={"input_ids"})
                )
                output = self.actor.train_batch(tensor_data, step_scheduler=False)
                if output is not None:
                    outputs.append(output)
        if not outputs:
            return None
        lr = self.actor.step_scheduler()
        metrics = _mean_metrics(outputs)
        metrics["lr"] = lr
        return DataProto(meta_info={"metrics": metrics})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        return self.actor.load_checkpoint(local_path, hdfs_path, del_local_after_load)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        return self.actor.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        if self.config.rollout.free_cache_engine:
            await self.rollout.release()
        aggressive_empty_cache(force_sync=True)
        set_expandable_segments(True)
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: Optional[int] = None, mode: str = "auto"):
        """Synchronize actor parameters to rollout through CheckpointEngine.

        ``naive`` keeps the existing colocated semantics.  Other registered
        transports retain the payload for a disaggregated rollout consumer.
        """
        effective_mode = mode if mode != "auto" else self.config.rollout.checkpoint_engine.backend
        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param()
        await self.checkpoint_engine.send_weights(per_tensor_param, global_steps=global_steps)
        if effective_mode != "naive":
            return
        aggressive_empty_cache(force_sync=True)
        set_expandable_segments(False)
        per_tensor_param = await self.checkpoint_engine.receive_weights(global_steps=global_steps)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        await self.rollout.update_weights(
            per_tensor_param, peft_config=peft_config, base_sync_done=self.base_sync_done, global_steps=global_steps
        )
        self.actor.engine.to("cpu", model=True, optimizer=False)
        aggressive_empty_cache(force_sync=True)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        self.base_sync_done = True
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        """Legacy alias retained while RayPPOTrainer is migrated to V1."""
        await self.update_weights(global_steps=None, mode="naive")

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        return await self.rollout.chat_completion(json_request)

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> list[int]:
        return await self.rollout.generate(prompt_ids, sampling_params, request_id, image_data=image_data)
