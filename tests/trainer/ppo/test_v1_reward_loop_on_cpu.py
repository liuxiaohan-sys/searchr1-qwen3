import asyncio

import torch

from verl.trainer.ppo.v1 import RewardLoopManager
from verl.utils import tensordict_utils as tu


def test_colocated_reward_loop_attaches_token_rewards():
    async def run():
        batch = tu.get_tensordict({"responses": torch.ones(2, 3, dtype=torch.long)})
        manager = RewardLoopManager(reward_fn=lambda data: torch.ones_like(data["responses"], dtype=torch.float))
        output = await manager.compute(batch)
        assert output["token_level_rewards"].shape == (2, 3)

    asyncio.run(run())
