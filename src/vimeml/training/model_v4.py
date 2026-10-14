"""Fixed V4-a capacity allocation, with a separate FP32 research KV path."""
import math
from dataclasses import dataclass, fields

import torch

from vimeml.training.model_v2 import GPTV2Config, TinyGPTV2

PARAMETERS = 15_070_848
LAYER_PARAMETERS = 1_696_512


@dataclass(frozen=True)
class GPTV4Config(GPTV2Config):
    vocab_size: int = 8192
    context_length: int = 128
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 7
    d_ff: int = 960
    dropout: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        for field in fields(self):
            if type(getattr(self, field.name)) is not type(field.default) or getattr(self, field.name) != field.default:
                raise ValueError(f"V4-a {field.name} is fixed to {field.default}")


@dataclass
class KVCacheV4:
    key: torch.Tensor
    value: torch.Tensor
    length: int = 0

    def clone(self):
        return KVCacheV4(self.key.clone(), self.value.clone(), self.length)

    @torch.inference_mode()
    def restore(self, snapshot):
        if (self.key.shape != snapshot.key.shape or self.value.shape != snapshot.value.shape
                or self.key.device != snapshot.key.device or self.value.device != snapshot.value.device
                or self.key.dtype != snapshot.key.dtype or self.value.dtype != snapshot.value.dtype
                or type(snapshot.length) is not int or not 0 <= snapshot.length <= 128):
            raise ValueError("Incompatible KV snapshot")
        self.key.copy_(snapshot.key)
        self.value.copy_(snapshot.value)
        self.length = snapshot.length


class TinyGPTV4(TinyGPTV2):
    """Same proven operations as V2; independent fixed architecture identity."""
    def __init__(self, config=GPTV4Config()):
        if type(config) is not GPTV4Config:
            raise ValueError("V4 requires its fixed GPTV4Config")
        super().__init__(config)
        if self.parameter_count() != PARAMETERS or any(
                sum(p.numel() for p in block.parameters()) != LAYER_PARAMETERS for block in self.blocks):
            raise ValueError("V4 parameter budget does not match its contract")

    def new_cache(self, batch_size=1):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Positive integer cache batch required")
        shape = (7, batch_size, 6, 128, 64)
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
        shape = (7, batch, 6, 128, 64)
        if (cache.key.shape != shape or cache.value.shape != shape
                or cache.key.dtype != torch.float32 or cache.value.dtype != torch.float32
                or cache.key.device != input_ids.device or cache.value.device != input_ids.device
                or input_ids.device != self.token_embedding.weight.device
                or type(cache.length) is not int or cache.length < 0
                or size < 1 or cache.length + size > 128):
            raise ValueError("Invalid KV cache or context overflow")
        if int(input_ids.min()) < 0 or int(input_ids.max()) >= 8192:
            raise ValueError("Token outside fixed vocabulary")
        start, end = cache.length, cache.length + size
        positions = torch.arange(start, end, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal = positions[:, None] >= torch.arange(end, device=input_ids.device)[None, :]
        for layer, block in enumerate(self.blocks):
            normalized = block.attention_norm(hidden)
            qkv = block.attention.qkv(normalized).reshape(batch, size, 3, 6, 64)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            cache.key[layer, :, :, start:end].copy_(key)
            cache.value[layer, :, :, start:end].copy_(value)
            scores = torch.matmul(query, cache.key[layer, :, :, :end].transpose(-2, -1)) / math.sqrt(64)
            probabilities = torch.softmax(scores.masked_fill(~causal, -torch.inf), dim=-1)
            attended = torch.matmul(probabilities, cache.value[layer, :, :, :end])
            attended = attended.transpose(1, 2).reshape(batch, size, 384)
            hidden = hidden + block.attention.projection(attended)
            hidden = hidden + block.ffn(block.ffn_norm(hidden))
        cache.length = end
        return self.lm_head(self.final_norm(hidden))
