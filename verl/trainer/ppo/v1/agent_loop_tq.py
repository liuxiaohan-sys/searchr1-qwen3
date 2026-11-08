"""TransferQueue publishing boundary for request-level agent loops."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

import numpy as np
import ray
import torch

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorkerBase, get_trajectory_info
from verl.utils import tensordict_utils as tu
from verl.utils.data_protocol_adapter import validate_training_tensordict


class AsyncQueueWriter(Protocol):
    async def async_kv_put(self, *, key, partition_id, tag): ...

    async def async_kv_batch_put(self, *, keys, fields, tags, partition_id): ...


class AgentLoopTQProducer:
    """Publish prompts and completed trajectories using the upstream V1 tag model."""

    def __init__(self, client: AsyncQueueWriter):
        self.client = client

    async def mark_prompt(self, uid: str, *, global_steps: int, status: str = "pending", partition_id="train"):
        if status not in {"pending", "running", "finished", "failure"}:
            raise ValueError(f"Unknown prompt status {status!r}")
        await self.client.async_kv_put(
            key=uid,
            partition_id=partition_id,
            tag={"is_prompt": True, "status": status, "global_steps": global_steps},
        )

    async def publish_output(
        self,
        *,
        uid: str,
        session_id: int,
        output: Any,
        global_steps: int,
        weight_version: int,
        index: int = 0,
        partition_id: str = "train",
    ) -> str:
        prompts = torch.as_tensor(output.prompt_ids, dtype=torch.long).unsqueeze(0)
        responses = torch.as_tensor(output.response_ids, dtype=torch.long).unsqueeze(0)
        response_mask = torch.as_tensor(output.response_mask, dtype=torch.bool).unsqueeze(0)
        input_ids = torch.cat((prompts, responses), dim=-1)
        batch = tu.get_tensordict(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "responses": responses,
                "response_mask": response_mask,
            },
            non_tensor_dict={"global_steps": global_steps, "weight_version": weight_version},
        )
        validate_training_tensordict(batch, require_versions=True)
        key = f"{uid}_{session_id}_{index}"
        tag = {
            "status": "success",
            "global_steps": global_steps,
            "min_global_steps": weight_version,
            "max_global_steps": weight_version,
            "prompt_len": prompts.shape[-1],
            "response_len": responses.shape[-1],
            "seq_len": input_ids.shape[-1],
        }
        await self.client.async_kv_batch_put(
            keys=[key], fields=batch, tags=[tag], partition_id=partition_id
        )
        return key


@ray.remote
class AgentLoopWorkerTQ(AgentLoopWorkerBase):
    """Fire-and-forget request worker that streams completed trajectories to TQ."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import transfer_queue as tq

        tq.init()
        self._tq = tq
        self._background_tasks = set()

    async def generate_sequences(self, batch: DataProto):
        config = self.config.actor_rollout_ref.rollout
        sampling_params = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repetition_penalty": 1.0,
            "logprobs": config.calculate_log_probs,
        }
        validate = batch.meta_info.get("validate", False)
        if validate:
            sampling_params.update(top_p=config.val_kwargs.top_p, temperature=config.val_kwargs.temperature)
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(
                [config.agent.default_agent_loop] * len(batch), dtype=object
            )
        if "uid" not in batch.non_tensor_batch:
            raise ValueError("AgentLoopWorkerTQ requires uid")
        indexes = batch.non_tensor_batch.get("index", np.arange(len(batch)))
        trajectories = await get_trajectory_info(
            batch.meta_info["global_steps"], indexes.tolist(), validate
        )
        groups = {}
        for index in range(len(batch)):
            kwargs = {key: value[index] for key, value in batch.non_tensor_batch.items()}
            groups.setdefault(str(kwargs["uid"]), []).append((trajectories[index], kwargs))
        for uid, entries in groups.items():
            task = asyncio.create_task(self._run_group(uid, entries, sampling_params, batch.meta_info))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _run_group(self, uid, entries, sampling_params, meta_info):
        partition = "val" if entries[0][0]["validate"] else "train"
        await self._tq.async_kv_put(
            key=uid,
            partition_id=partition,
            tag={"is_prompt": True, "status": "running", "global_steps": meta_info["weight_version"]},
        )
        status = "failure"
        try:
            results = await asyncio.gather(
                *[
                    self._run_and_publish(uid, sampling_params, trajectory, meta_info, kwargs, partition)
                    for trajectory, kwargs in entries
                ],
                return_exceptions=True,
            )
            status = "failure" if any(isinstance(result, BaseException) for result in results) else "finished"
        finally:
            await self._tq.async_kv_put(
                key=uid,
                partition_id=partition,
                tag={"is_prompt": True, "status": status, "global_steps": meta_info["weight_version"]},
            )

    async def _run_and_publish(self, uid, sampling_params, trajectory, meta_info, kwargs, partition):
        try:
            output = await self._run_agent_loop(
                sampling_params, trajectory, agent_name=kwargs.pop("agent_name"), **kwargs
            )
            proto = self._postprocess([output])
            fields = proto.to_tensordict()
            tu.assign_non_tensor(
                fields,
                global_steps=meta_info["global_steps"],
                weight_version=meta_info["weight_version"],
                uid=uid,
            )
            validate_training_tensordict(fields, require_versions=True)
            key = f"{uid}_{trajectory['rollout_n']}_0"
            tag = {
                "status": "success",
                "global_steps": meta_info["global_steps"],
                "min_global_steps": meta_info["weight_version"],
                "max_global_steps": meta_info["weight_version"],
            }
            await self._tq.async_kv_batch_put(
                keys=[key], fields=fields, tags=[tag], partition_id=partition
            )
        except Exception as error:
            return error


class _StreamingAgentLoopManager(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = AgentLoopWorkerTQ
        super().__init__(*args, **kwargs)

    def generate_sequences(self, prompts: DataProto):
        chunks = prompts.chunk(len(self.agent_loop_workers))
        ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunks, strict=False)
            ]
        )


class AgentLoopManagerTQ:
    """TensorDict facade over streaming Ray agent-loop workers."""

    def __init__(self, config, worker_group=None, rm_wg=None, queue_client=None):
        self.config = config
        self._manager = _StreamingAgentLoopManager(config=config, worker_group=worker_group, rm_wg=rm_wg)
        if queue_client is None:
            import transfer_queue as tq

            queue_client = tq
        self.queue = queue_client

    def generate_sequences(self, prompts):
        prompt_proto = DataProto.from_tensordict(prompts)
        prompt_proto.meta_info["global_steps"] = int(tu.unwrap_non_tensor_data(prompts["global_steps"]))
        prompt_proto.meta_info["weight_version"] = int(tu.unwrap_non_tensor_data(prompts["weight_version"]))
        return self._manager.generate_sequences(prompt_proto)
