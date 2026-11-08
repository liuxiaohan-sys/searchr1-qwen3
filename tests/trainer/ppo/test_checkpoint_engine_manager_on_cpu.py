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
