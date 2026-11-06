"""Vime 1.3 Chinese 20M decoder: the V4 block family with two registered shapes.

d384, 6 heads, SwiGLU 960, learned positions 128, tied embeddings; only the
vocabulary and layer count vary, fixed to the 8K/L10 and 12K/L9 pairs of
docs/zh-CN/training-zh-v13-plan.md.
"""
import math
from dataclasses import dataclass

import torch

from vimeml.training.model_v2 import GPTV2Config, TinyGPTV2

LAYER_PARAMETERS = 1_696_512
# vocab_size -> (n_layers, total parameters)
SHAPES = {8192: (10, 20_160_384), 12288: (9, 20_036_736)}


@dataclass(frozen=True)
class GPTZh13Config(GPTV2Config):
    vocab_size: int = 12288
    context_length: int = 128
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 9
    d_ff: int = 960
    dropout: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.vocab_size not in SHAPES or self.n_layers != SHAPES[self.vocab_size][0]:
            raise ValueError("Vime 1.3 zh shapes are 8192/L10 and 12288/L9")
        fixed = {"context_length": 128, "d_model": 384, "n_heads": 6, "d_ff": 960, "dropout": 0.0}
        for name, value in fixed.items():
            if type(getattr(self, name)) is not type(value) or getattr(self, name) != value:
                raise ValueError(f"Vime 1.3 zh {name} is fixed to {value}")


@dataclass
class KVCacheZh13:
    key: torch.Tensor
    value: torch.Tensor
    length: int = 0

    def clone(self):
        return KVCacheZh13(self.key.clone(), self.value.clone(), self.length)


class TinyGPTZh13(TinyGPTV2):
    def __init__(self, config=GPTZh13Config()):
        if type(config) is not GPTZh13Config:
            raise ValueError("Vime 1.3 zh requires GPTZh13Config")
        super().__init__(config)
        if self.parameter_count() != SHAPES[config.vocab_size][1] or any(
                sum(p.numel() for p in block.parameters()) != LAYER_PARAMETERS for block in self.blocks):
            raise ValueError("Vime 1.3 zh parameter budget does not match its contract")

    def new_cache(self, batch_size=1):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Positive integer cache batch required")
        shape = (self.config.n_layers, batch_size, 6, 128, 64)
        device = self.token_embedding.weight.device
        return KVCacheZh13(torch.zeros(shape, device=device), torch.zeros(shape, device=device))

    @torch.inference_mode()
    def forward_cached(self, input_ids, cache):
        """FP32 research KV path; matches forward() on the same prefix."""
        if self.training or self.token_embedding.weight.dtype != torch.float32:
            raise ValueError("KV path requires eval mode and FP32 weights")
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected integer [batch,time] tokens")
        batch, size = input_ids.shape
        if (cache.key.shape != (self.config.n_layers, batch, 6, 128, 64) or size < 1
                or cache.length + size > 128):
            raise ValueError("Invalid KV cache or context overflow")
        if int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size:
            raise ValueError("Token outside vocabulary")
        start, end = cache.length, cache.length + size
        positions = torch.arange(start, end, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal = positions[:, None] >= torch.arange(end, device=input_ids.device)[None, :]
        for layer, block in enumerate(self.blocks):
            qkv = block.attention.qkv(block.attention_norm(hidden)).reshape(batch, size, 3, 6, 64)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            cache.key[layer, :, :, start:end].copy_(key)
            cache.value[layer, :, :, start:end].copy_(value)
            scores = torch.matmul(query, cache.key[layer, :, :, :end].transpose(-2, -1)) / math.sqrt(64)
            attended = torch.matmul(torch.softmax(scores.masked_fill(~causal, -torch.inf), dim=-1),
                                    cache.value[layer, :, :, :end])
            hidden = hidden + block.attention.projection(attended.transpose(1, 2).reshape(batch, size, 384))
            hidden = hidden + block.ffn(block.ffn_norm(hidden))
        cache.length = end
        return self.lm_head(self.final_norm(hidden))
