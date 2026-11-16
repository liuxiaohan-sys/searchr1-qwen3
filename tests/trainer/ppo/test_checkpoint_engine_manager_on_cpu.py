import asyncio
from types import SimpleNamespace

import pytest

from verl.checkpoint_engine import CheckpointEngineManager


class Actor:
    async def update_weights(self, **kwargs):
        self.kwargs = kwargs
        return {"synced": True}


class Replica:
    def __init__(self):
        self.events = []

    async def sleep(self): self.events.append("sleep")
    async def wake_up(self): self.events.append("wake")
    async def abort_all_requests(self, reject_request=False): self.events.append(("abort", reject_request))
    async def update_weights(self, **kwargs): self.events.append(("update", kwargs))


def test_manager_versions_and_replica_lifecycle():
    async def run():
        actor, replica = Actor(), Replica()
        manager = CheckpointEngineManager(SimpleNamespace(backend="naive"), actor, [replica])
        await manager.sleep_replicas()
        await manager.wake_up_replicas()
        await manager.abort_replicas(reject_request=True)
        assert await manager.update_weights(1) == {"synced": True}
        assert await manager.wait_for_weight_update() == 1
        assert actor.kwargs == {"global_steps": 1, "mode": "naive"}
        assert replica.events == ["sleep", "wake", ("abort", True)]
        with pytest.raises(ValueError, match="monotonically"):
            await manager.update_weights(1)

    asyncio.run(run())


def test_registered_transport_updates_rollout_after_actor_publish():
    from verl.checkpoint_engine import CheckpointEngine, CheckpointEngineRegistry

    @CheckpointEngineRegistry.register("test_transport")
    class TestTransport(CheckpointEngine):
        async def send_weights(self, weights, global_steps=None): pass
        async def receive_weights(self, global_steps=None): pass

    async def run():
        actor, replica = Actor(), Replica()
        manager = CheckpointEngineManager(SimpleNamespace(backend="test_transport"), actor, [replica])
        await manager.update_weights(2)
        assert replica.events == [
            ("abort", False),
            ("update", {"global_steps": 2, "mode": "test_transport"}),
            "wake",
        ]

    asyncio.run(run())


def test_transport_starts_sender_and_receiver_before_waiting():
    from verl.checkpoint_engine import CheckpointEngine, CheckpointEngineRegistry

    @CheckpointEngineRegistry.register("rendezvous_transport")
    class RendezvousTransport(CheckpointEngine):
        async def send_weights(self, weights, global_steps=None): pass
        async def receive_weights(self, global_steps=None): pass

    async def run():
        sender_started = asyncio.Event()
        receiver_started = asyncio.Event()

        class RendezvousActor:
            async def update_weights(self, **kwargs):
                sender_started.set()
                await asyncio.wait_for(receiver_started.wait(), timeout=1)
                return "sent"

        class RendezvousReplica(Replica):
            async def update_weights(self, **kwargs):
                receiver_started.set()
                await asyncio.wait_for(sender_started.wait(), timeout=1)

        manager = CheckpointEngineManager(
            SimpleNamespace(backend="rendezvous_transport"), RendezvousActor(), [RendezvousReplica()]
        )
        assert await manager.update_weights(3) == "sent"

    asyncio.run(run())


def test_failed_version_can_retry_and_replica_is_restored():
    from verl.checkpoint_engine import CheckpointEngine, CheckpointEngineRegistry

    @CheckpointEngineRegistry.register("retry_transport")
    class RetryTransport(CheckpointEngine):
        async def send_weights(self, weights, global_steps=None): pass
        async def receive_weights(self, global_steps=None): pass

    async def run():
        class FlakyActor(Actor):
            attempts = 0

            async def update_weights(self, **kwargs):
                self.attempts += 1
                if self.attempts == 1:
                    raise RuntimeError("transfer failed")
                return {"synced": True}

        actor, replica = FlakyActor(), Replica()
        manager = CheckpointEngineManager(SimpleNamespace(backend="retry_transport"), actor, [replica])
        with pytest.raises(RuntimeError, match="transfer failed"):
            await manager.update_weights(4)
        assert replica.events[-1] == "wake"
        assert await manager.wait_for_weight_update() is None
        assert await manager.update_weights(4) == {"synced": True}
        assert await manager.wait_for_weight_update() == 4

    asyncio.run(run())
