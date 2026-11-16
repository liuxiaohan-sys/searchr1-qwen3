"""Weight iterator adapters shared by vLLM rollout implementations."""

import torch


async def load_vllm_weights(model, weights) -> int | None:
    """Load sync weights or consume an async transport stream without prefetch."""
    if not hasattr(weights, "__aiter__"):
        model.load_weights(weights)
        return None
    loaded = 0
    async for name, tensor in weights:
        # The tensor may alias a reusable NCCL receive buffer. The synchronous
        # loader must finish before requesting the next item from the stream.
        model.load_weights(iter(((name, tensor),)))
        # Transport buckets are reused as soon as the async iterator advances.
        # vLLM loaders may enqueue copies on a different CUDA stream, so make
        # completion explicit before allowing the receiver to reuse the buffer.
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        loaded += 1
    return loaded


async def materialize_async_weights(weights):
    """Materialize adapter weights and detach them from transport buffers."""
    if not hasattr(weights, "__aiter__"):
        return dict(weights)
    result = {}
    async for name, tensor in weights:
        result[name] = tensor.clone()
    return result
