"""800M Chinese desktop-model decoder (docs/zh-CN/training-zh-800m-plan.md).

The zh-200m block family scaled up: d1536, 24 heads of 64, SwiGLU 4096, RMSNorm,
no bias, learned positions, tied embeddings; 27 layers, the 20M's 12K tokenizer and
context 512 (784,157,184 parameters). One registered shape, so a config that drifts
from the plan fails at construction.
"""
import math
from dataclasses import dataclass

import torch

from vimeml.training.model_v2 import GPTV2Config, TinyGPTV2

D_MODEL, HEADS, HEAD_DIM, D_FF = 1536, 24, 64, 4096
# 4·d² attention + 3·d·d_ff SwiGLU + two RMSNorm weights
LAYER_PARAMETERS = 4 * D_MODEL ** 2 + 3 * D_MODEL * D_FF + 2 * D_MODEL
# (vocab_size, context_length) -> (n_layers, total parameters)
# total = V·d (tied embedding) + ctx·d (positions) + L·LAYER_PARAMETERS + d (final norm)
SHAPES = {(12_288, 512): (27, 784_157_184)}


@dataclass(frozen=True)
class GPTZh800mConfig(GPTV2Config):
    vocab_size: int = 12288
    context_length: int = 512
    d_model: int = D_MODEL
    n_heads: int = HEADS
    n_layers: int = 27
    d_ff: int = D_FF
    dropout: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if ((self.vocab_size, self.context_length) not in SHAPES
                or self.n_layers != SHAPES[(self.vocab_size, self.context_length)][0]):
            raise ValueError("zh-800m shape is 12K/L27/c512")
        fixed = {"d_model": D_MODEL, "n_heads": HEADS, "d_ff": D_FF, "dropout": 0.0}
        for name, value in fixed.items():
            if type(getattr(self, name)) is not type(value) or getattr(self, name) != value:
                raise ValueError(f"zh-800m {name} is fixed to {value}")


@dataclass
class KVCacheZh800m:
    key: torch.Tensor
    value: torch.Tensor
    length: int = 0

    def clone(self):
        return KVCacheZh800m(self.key.clone(), self.value.clone(), self.length)


class TinyGPTZh800m(TinyGPTV2):
    def __init__(self, config=GPTZh800mConfig()):
        if type(config) is not GPTZh800mConfig:
            raise ValueError("zh-800m requires GPTZh800mConfig")
        super().__init__(config)
        if self.parameter_count() != SHAPES[(config.vocab_size, config.context_length)][1] or any(
                sum(p.numel() for p in block.parameters()) != LAYER_PARAMETERS for block in self.blocks):
            raise ValueError("zh-800m parameter budget does not match its contract")

    def new_cache(self, batch_size=1):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Positive integer cache batch required")
        shape = (self.config.n_layers, batch_size, HEADS, self.config.context_length, HEAD_DIM)
        device = self.token_embedding.weight.device
        return KVCacheZh800m(torch.zeros(shape, device=device), torch.zeros(shape, device=device))

    @torch.inference_mode()
    def forward_cached(self, input_ids, cache):
        """FP32 research KV path; matches forward() on the same prefix."""
        if self.training or self.token_embedding.weight.dtype != torch.float32:
            raise ValueError("KV path requires eval mode and FP32 weights")
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected integer [batch,time] tokens")
        context = self.config.context_length
        batch, size = input_ids.shape
        if (cache.key.shape != (self.config.n_layers, batch, HEADS, context, HEAD_DIM) or size < 1
                or cache.length + size > context):
            raise ValueError("Invalid KV cache or context overflow")
        if int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size:
            raise ValueError("Token outside vocabulary")
        start, end = cache.length, cache.length + size
        positions = torch.arange(start, end, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal = positions[:, None] >= torch.arange(end, device=input_ids.device)[None, :]
        for layer, block in enumerate(self.blocks):
            qkv = block.attention.qkv(block.attention_norm(hidden)).reshape(batch, size, 3, HEADS, HEAD_DIM)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            cache.key[layer, :, :, start:end].copy_(key)
            cache.value[layer, :, :, start:end].copy_(value)
            scores = torch.matmul(query, cache.key[layer, :, :, :end].transpose(-2, -1)) / math.sqrt(HEAD_DIM)
            attended = torch.matmul(torch.softmax(scores.masked_fill(~causal, -torch.inf), dim=-1),
                                    cache.value[layer, :, :, :end])
            hidden = hidden + block.attention.projection(attended.transpose(1, 2).reshape(batch, size, D_MODEL))
            hidden = hidden + block.ffn(block.ffn_norm(hidden))
        cache.length = end
        return self.lm_head(self.final_norm(hidden))
