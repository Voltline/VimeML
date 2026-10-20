"""Fixed V5 capacity allocation: the V4 recipe and 8K tokenizer scaled to d512/L8.

Same operations as V2/V4 (RMSNorm, SwiGLU, bias-free, tied head, learned
positions, head width 64); only width, depth and FFN width change. V4 stays frozen.
"""
import math
from dataclasses import dataclass, fields

import torch

from vimeml.training.model_v2 import GPTV2Config, TinyGPTV2
from vimeml.training.model_v4 import KVCacheV4

PARAMETERS = 28_385_792
LAYER_PARAMETERS = 3_015_680
HEAD_WIDTH = 64
CACHE_SHAPE = (8, 1, 8, 128, HEAD_WIDTH)


@dataclass(frozen=True)
class GPTV5Config(GPTV2Config):
    vocab_size: int = 8192
    context_length: int = 128
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 1280
    dropout: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        for field in fields(self):
            if type(getattr(self, field.name)) is not type(field.default) or getattr(self, field.name) != field.default:
                raise ValueError(f"V5 {field.name} is fixed to {field.default}")


class TinyGPTV5(TinyGPTV2):
    """V2 operations at the fixed V5 size, with the FP32 research KV path of V4."""
    def __init__(self, config=GPTV5Config()):
        if type(config) is not GPTV5Config:
            raise ValueError("V5 requires its fixed GPTV5Config")
        super().__init__(config)
        if self.parameter_count() != PARAMETERS or any(
                sum(p.numel() for p in block.parameters()) != LAYER_PARAMETERS for block in self.blocks):
            raise ValueError("V5 parameter budget does not match its contract")

    def cache_shape(self, batch_size=1):
        layers, _, heads, length, width = CACHE_SHAPE
        return (layers, batch_size, heads, length, width)

    def new_cache(self, batch_size=1):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Positive integer cache batch required")
        shape = self.cache_shape(batch_size)
        device = self.token_embedding.weight.device
        return KVCacheV4(torch.zeros(shape, device=device, dtype=torch.float32),
                         torch.zeros(shape, device=device, dtype=torch.float32))

    @torch.inference_mode()
    def forward_cached(self, input_ids, cache):
        if self.training or self.token_embedding.weight.dtype != torch.float32:
            raise ValueError("Research KV path requires eval mode and FP32 weights")
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected integer [batch,time] tokens")
        batch, size = input_ids.shape
        shape = self.cache_shape(batch)
        heads, width, context = shape[2], shape[4], shape[3]
        if (cache.key.shape != shape or cache.value.shape != shape
                or cache.key.dtype != torch.float32 or cache.value.dtype != torch.float32
                or cache.key.device != input_ids.device or cache.value.device != input_ids.device
                or input_ids.device != self.token_embedding.weight.device
                or type(cache.length) is not int or cache.length < 0
                or size < 1 or cache.length + size > context):
            raise ValueError("Invalid KV cache or context overflow")
        if int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size:
            raise ValueError("Token outside fixed vocabulary")
        start, end = cache.length, cache.length + size
        positions = torch.arange(start, end, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal = positions[:, None] >= torch.arange(end, device=input_ids.device)[None, :]
        for layer, block in enumerate(self.blocks):
            normalized = block.attention_norm(hidden)
            qkv = block.attention.qkv(normalized).reshape(batch, size, 3, heads, width)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            cache.key[layer, :, :, start:end].copy_(key)
            cache.value[layer, :, :, start:end].copy_(value)
            scores = torch.matmul(query, cache.key[layer, :, :, :end].transpose(-2, -1)) / math.sqrt(width)
            probabilities = torch.softmax(scores.masked_fill(~causal, -torch.inf), dim=-1)
            attended = torch.matmul(probabilities, cache.value[layer, :, :, :end])
            attended = attended.transpose(1, 2).reshape(batch, size, self.config.d_model)
            hidden = hidden + block.attention.projection(attended)
            hidden = hidden + block.ffn(block.ffn_norm(hidden))
        cache.length = end
        return self.lm_head(self.final_norm(hidden))
