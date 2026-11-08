import asyncio
from types import SimpleNamespace

from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopTQProducer


class FakeAsyncQueue:
    async def async_kv_put(self, **kwargs):
        self.prompt = kwargs

    async def async_kv_batch_put(self, **kwargs):
        self.batch = kwargs


def test_agent_loop_output_is_versioned_and_published():
    async def run():
        queue = FakeAsyncQueue()
        producer = AgentLoopTQProducer(queue)
        await producer.mark_prompt("uid", global_steps=7)
        output = SimpleNamespace(prompt_ids=[1, 2], response_ids=[3, 4], response_mask=[1, 0])
        key = await producer.publish_output(
            uid="uid", session_id=0, output=output, global_steps=7, weight_version=6
        )
        await producer.mark_prompt("uid", global_steps=7, status="finished")
        assert key == "uid_0_0"
        assert queue.batch["fields"]["weight_version"] == 6
        assert queue.batch["tags"][0]["min_global_steps"] == 6
        assert queue.prompt["tag"]["status"] == "finished"

    asyncio.run(run())
