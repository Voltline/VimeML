"""200M Chinese desktop-model decoder: the V4 block family with four registered shapes.

d1024, 16 heads, SwiGLU 2560, learned positions, tied embeddings; the
vocabulary, layer count and context length vary, fixed to the pairs of the
200M training plan (docs/zh-CN/training-zh-200m-plan.md): 16K/L15/c512,
24K/L14/c512, 32K/L14/c512 and 24K/L14/c128.
"""
import math
from dataclasses import dataclass

import torch

from vimeml.training.model_v2 import GPTV2Config, TinyGPTV2

LAYER_PARAMETERS = 12_060_672
# (vocab_size, context_length) -> (n_layers, total parameters)
# total = V·d (tied embedding) + ctx·d (positions) + L·LAYER_PARAMETERS + d (final norm)
SHAPES = {(16_384, 512): (15, 198_212_608),
          (24_576, 512): (14, 194_540_544),
          (32_768, 512): (14, 202_929_152),
          (24_576, 128): (14, 194_147_328)}


@dataclass(frozen=True)
class GPTZh200mConfig(GPTV2Config):
    vocab_size: int = 24576
    context_length: int = 512
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 14
    d_ff: int = 2560
    dropout: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if ((self.vocab_size, self.context_length) not in SHAPES
                or self.n_layers != SHAPES[(self.vocab_size, self.context_length)][0]):
            raise ValueError("zh-200m shapes are 16K/L15/c512, 24K/L14/c512, 32K/L14/c512 and 24K/L14/c128")
        fixed = {"d_model": 1024, "n_heads": 16, "d_ff": 2560, "dropout": 0.0}
        for name, value in fixed.items():
            if type(getattr(self, name)) is not type(value) or getattr(self, name) != value:
                raise ValueError(f"zh-200m {name} is fixed to {value}")


@dataclass
class KVCacheZh200m:
    key: torch.Tensor
    value: torch.Tensor
    length: int = 0

    def clone(self):
        return KVCacheZh200m(self.key.clone(), self.value.clone(), self.length)


class TinyGPTZh200m(TinyGPTV2):
    def __init__(self, config=GPTZh200mConfig()):
        if type(config) is not GPTZh200mConfig:
            raise ValueError("zh-200m requires GPTZh200mConfig")
        super().__init__(config)
        if self.parameter_count() != SHAPES[(config.vocab_size, config.context_length)][1] or any(
                sum(p.numel() for p in block.parameters()) != LAYER_PARAMETERS for block in self.blocks):
            raise ValueError("zh-200m parameter budget does not match its contract")

    def new_cache(self, batch_size=1):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Positive integer cache batch required")
        context = self.config.context_length
        shape = (self.config.n_layers, batch_size, 16, context, 64)
        device = self.token_embedding.weight.device
        return KVCacheZh200m(torch.zeros(shape, device=device), torch.zeros(shape, device=device))

    @torch.inference_mode()
    def forward_cached(self, input_ids, cache):
        """FP32 research KV path; matches forward() on the same prefix."""
        if self.training or self.token_embedding.weight.dtype != torch.float32:
            raise ValueError("KV path requires eval mode and FP32 weights")
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected integer [batch,time] tokens")
        context = self.config.context_length
        batch, size = input_ids.shape
        if (cache.key.shape != (self.config.n_layers, batch, 16, context, 64) or size < 1
                or cache.length + size > context):
            raise ValueError("Invalid KV cache or context overflow")
        if int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size:
            raise ValueError("Token outside vocabulary")
        start, end = cache.length, cache.length + size
        positions = torch.arange(start, end, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal = positions[:, None] >= torch.arange(end, device=input_ids.device)[None, :]
        for layer, block in enumerate(self.blocks):
            qkv = block.attention.qkv(block.attention_norm(hidden)).reshape(batch, size, 3, 16, 64)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            cache.key[layer, :, :, start:end].copy_(key)
            cache.value[layer, :, :, start:end].copy_(value)
            scores = torch.matmul(query, cache.key[layer, :, :, :end].transpose(-2, -1)) / math.sqrt(64)
            attended = torch.matmul(torch.softmax(scores.masked_fill(~causal, -torch.inf), dim=-1),
                                    cache.value[layer, :, :, :end])
            hidden = hidden + block.attention.projection(attended.transpose(1, 2).reshape(batch, size, 1024))
            hidden = hidden + block.ffn(block.ffn_norm(hidden))
        cache.length = end
        return self.lm_head(self.final_norm(hidden))
