"""NCCL actor-to-rollout weight transport.

Adapted from the upstream verl checkpoint engine. Tensor metadata travels over
ZeroMQ while byte buckets are broadcast through Ray's NCCL collective group.
"""

import asyncio
import socket
from dataclasses import dataclass
from typing import AsyncGenerator
import ray
import ray.util.collective as collective
import torch
import zmq

from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
    TensorMeta,
    merge_weight_chunks,
    split_weight_chunks,
)
from verl.utils.net_utils import is_ipv6


def _get_free_port(ip: str):
    family = socket.AF_INET6 if is_ipv6(ip) else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((ip, 0))
        return sock.getsockname()[1]


@dataclass
class MasterMetadata:
    zmq_ip: str
    zmq_port: int


class BroadcastOperation:
    def __init__(self, rank, group_name, bucket, metadata, socket, topic):
        self.rank = rank
        self.group_name = group_name
        self.bucket = bucket
        self.metadata = metadata
        self.socket = socket
        self.topic = topic
        self._task = asyncio.get_running_loop().run_in_executor(None, self._run)

    def _run(self):
        if self.rank == 0:
            self.socket.send_string(self.topic, flags=zmq.SNDMORE)
            self.socket.send_pyobj(self.metadata)
        else:
            self.socket.recv_string()
            self.metadata = self.socket.recv_pyobj()
            self.bucket = self.bucket[: self.metadata["length"]]
        collective.broadcast(self.bucket, src_rank=0, group_name=self.group_name)

    async def wait(self):
        await self._task
        return self.metadata


@CheckpointEngineRegistry.register("nccl")
class NCCLCheckpointEngine(CheckpointEngine):
    support_publish_weights = False

    def __init__(self, bucket_size, group_name="verl_checkpoint", rebuild_group=False, is_master=False, **kwargs):
        self.bucket_size = bucket_size
        self.group_name = group_name
        self.rebuild_group = rebuild_group
        self.is_master = is_master
        self.topic = "bucket_metadata"
        self.rank = self.world_size = None
        if is_master:
            self._start_zmq_server()

    def _start_zmq_server(self):
        self.ip = ray.util.get_node_ip_address().strip("[]")
        self.listen_port = _get_free_port(self.ip)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        if is_ipv6(self.ip):
            self.socket.setsockopt(zmq.IPV6, 1)
            address = f"tcp://[{self.ip}]:{self.listen_port}"
        else:
            address = f"tcp://{self.ip}:{self.listen_port}"
        self.socket.bind(address)

    def _connect(self, master):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        if is_ipv6(master.zmq_ip):
            self.socket.setsockopt(zmq.IPV6, 1)
            address = f"tcp://[{master.zmq_ip}]:{master.zmq_port}"
        else:
            address = f"tcp://{master.zmq_ip}:{master.zmq_port}"
        self.socket.connect(address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)

    def prepare(self):
        if self.is_master:
            # Delay the optional CUDA dependency until NCCL is actually used;
            # importing verl.checkpoint_engine must still register the backend
            # in CPU-only controller processes.
            from unittest.mock import patch

            with patch("importlib.metadata.distributions", return_value=[]):
                import cupy as cp

            self.cp = cp
            self.send_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
            self.recv_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
            return MasterMetadata(self.ip, self.listen_port)
        self.send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")
        self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="cuda")
        return None

    @classmethod
    def build_topology(cls, actor_size, rollout_size, metadata):
        master = metadata[0]
        if not isinstance(master, MasterMetadata):
            raise RuntimeError("actor rank 0 must be the NCCL checkpoint master")
        world_size = rollout_size + 1
        actor = {
            "rank": [0] + [-1] * (actor_size - 1),
            "world_size": [world_size] * actor_size,
            "master_metadata": [master] * actor_size,
        }
        rollout = {
            "rank": list(range(1, world_size)),
            "world_size": [world_size] * rollout_size,
            "master_metadata": [master] * rollout_size,
        }
        return actor, rollout

    def init_process_group(self, rank, world_size, master_metadata):
        self.rank, self.world_size = rank, world_size
        if rank < 0:
            return
        if self.rebuild_group or not collective.is_group_initialized(self.group_name):
            collective.init_collective_group(world_size, rank, "nccl", self.group_name)
        if rank > 0:
            self._connect(master_metadata)
        collective.barrier(self.group_name)

    def finalize(self):
        if self.rebuild_group and self.rank is not None and self.rank >= 0:
            collective.destroy_collective_group(self.group_name)
        self.send_buf = self.recv_buf = None
        torch.cuda.empty_cache()

    async def send_weights(self, weights, global_steps=None):
        if self.rank < 0:
            for _ in weights:
                pass
            return
        send_buf, recv_buf = self.send_buf, self.recv_buf
        bucket_meta, offset = {}, 0
        previous = None
        async for meta, chunk in split_weight_chunks(weights, self.bucket_size):
            if offset + meta.chunk_size > self.bucket_size:
                if previous is not None:
                    await previous.wait()
                previous = BroadcastOperation(0, self.group_name, send_buf[:offset], {
                    "bucket_meta": bucket_meta, "is_last": False, "length": offset
                }, self.socket, self.topic)
                send_buf, recv_buf = recv_buf, send_buf
                bucket_meta, offset = {}, 0
            meta.offset = offset
            bucket_meta[f"{meta.name}:{meta.chunk_offset}"] = meta
            send_buf[offset : offset + meta.chunk_size] = self.cp.asarray(chunk)
            offset += meta.chunk_size
        if previous is not None:
            await previous.wait()
        await BroadcastOperation(0, self.group_name, send_buf[:offset], {
            "bucket_meta": bucket_meta, "is_last": True, "length": offset
        }, self.socket, self.topic).wait()
        torch.cuda.synchronize()

    async def receive_weights(self, global_steps=None):
        async for item in merge_weight_chunks(self._receive_chunks(), self.bucket_size):
            yield item

    async def _receive_chunks(self) -> AsyncGenerator[tuple[TensorMeta, torch.Tensor], None]:
        send_buf, recv_buf = self.send_buf, self.recv_buf
        metadata = await BroadcastOperation(
            self.rank, self.group_name, recv_buf, None, self.socket, self.topic
        ).wait()
        torch.cuda.synchronize()
        send_buf, recv_buf = recv_buf, send_buf
        while True:
            next_op = None if metadata["is_last"] else BroadcastOperation(
                self.rank, self.group_name, recv_buf, None, self.socket, self.topic
            )
            for meta in metadata["bucket_meta"].values():
                yield meta, send_buf[meta.offset : meta.offset + meta.chunk_size]
            if next_op is None:
                break
            metadata = await next_op.wait()
            torch.cuda.synchronize()
            send_buf, recv_buf = recv_buf, send_buf
