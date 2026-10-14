"""V4 functional single-token KV graph and local TorchScript validation."""
import math

import torch
from torch import nn

from vimeml.deployment.graph import trace_graph

CACHE_SHAPE = (7, 1, 6, 128, 64)


class KVStepGraphV4(nn.Module):
    """Explicit functional FP32 buffers; caller owns snapshots and rollback."""
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
        self.register_buffer("slots", torch.arange(128))

    def forward(self, input_ids, position, key_cache, value_cache):
        position = position.to(torch.int64)
        hidden = self.model.token_embedding(input_ids) + self.model.position_embedding(position).unsqueeze(0)
        insert = (self.slots == position[0]).reshape(1, 1, 128, 1)
        valid = (self.slots <= position[0]).reshape(1, 1, 1, 128)
        keys, values = [], []
        for layer, block in enumerate(self.model.blocks):
            qkv = block.attention.qkv(block.attention_norm(hidden)).reshape(1, 1, 3, 6, 64)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            updated_key = torch.where(insert, key, key_cache[layer])
            updated_value = torch.where(insert, value, value_cache[layer])
            scores = torch.matmul(query, updated_key.transpose(-2, -1)) / math.sqrt(64)
            probabilities = torch.softmax(scores.masked_fill(~valid, -torch.inf), dim=-1)
            attended = torch.matmul(probabilities, updated_value).transpose(1, 2).reshape(1, 1, 384)
            hidden = hidden + block.attention.projection(attended)
            hidden = hidden + block.ffn(block.ffn_norm(hidden))
            keys.append(updated_key)
            values.append(updated_value)
        return self.model.lm_head(self.model.final_norm(hidden)), torch.stack(keys), torch.stack(values)


@torch.inference_mode()
def trace_kv_graph(model):
    graph = KVStepGraphV4(model).eval()
    args = (torch.tensor([[2]], dtype=torch.int32), torch.tensor([0], dtype=torch.int32),
            torch.zeros(CACHE_SHAPE), torch.zeros(CACHE_SHAPE))
    traced = torch.jit.trace(graph, args)
    # Different positions from the tracing example, including last cache slot.
    ids = torch.randint(7, 8192, (1, 128), generator=torch.Generator().manual_seed(940))
    cache = model.new_cache()
    for position in (0, 7, 127):
        if position > cache.length:
            model.forward_cached(ids[:, cache.length:position], cache)
        key, value = cache.key.clone(), cache.value.clone()
        token = ids[:, position:position + 1].int()
        reference = model.forward_cached(token, cache)
        actual = traced(token, torch.tensor([position], dtype=torch.int32), key, value)
        for a, b in zip(actual, (reference, cache.key, cache.value), strict=True):
            torch.testing.assert_close(a, b, atol=3e-4, rtol=3e-4)
    return traced
