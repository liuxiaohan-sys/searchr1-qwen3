import asyncio

import torch

from verl.workers.rollout.vllm_rollout.weight_utils import (
    load_vllm_weights,
    materialize_async_weights,
)


class _Model:
    def __init__(self):
        self.calls = []

    def load_weights(self, weights):
        self.calls.append([(name, tensor.clone()) for name, tensor in weights])


def test_streaming_weights_are_loaded_before_iterator_advances():
    model = _Model()
    buffer = torch.zeros(1)

    async def weights():
        buffer.fill_(1)
        yield "first", buffer
        buffer.fill_(2)
        yield "second", buffer

    loaded = asyncio.run(load_vllm_weights(model, weights()))

    assert loaded == 2
    assert [call[0][0] for call in model.calls] == ["first", "second"]
    assert [call[0][1].item() for call in model.calls] == [1, 2]


def test_sync_weights_keep_single_loader_call():
    model = _Model()
    asyncio.run(load_vllm_weights(model, iter((("a", torch.ones(1)), ("b", torch.ones(1))))))
    assert len(model.calls) == 1


def test_async_adapter_weights_are_cloned_from_transport_buffer():
    buffer = torch.zeros(1)

    async def weights():
        buffer.fill_(1)
        yield "a", buffer
        buffer.fill_(2)
        yield "b", buffer

    materialized = asyncio.run(materialize_async_weights(weights()))
    assert materialized["a"].item() == 1
    assert materialized["b"].item() == 2
