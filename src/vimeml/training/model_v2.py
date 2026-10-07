"""V2 decoder: primitive RMSNorm, bias-free attention and SwiGLU."""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from vimeml.training.model import GPTConfig


@dataclass(frozen=True)
class GPTV2Config(GPTConfig):
    d_model: int = 320
    n_heads: int = 5
    n_layers: int = 6
    d_ff: int = 832
    norm: str = "rmsnorm"
    norm_eps: float = 1e-5
    activation: str = "swiglu"
    bias: bool = False
    tie_embeddings: bool = True
    position_embedding: str = "learned"

    def __post_init__(self):
        super().__post_init__()
        if (
            self.norm != "rmsnorm"
            or self.activation != "swiglu"
            or self.bias is not False
            or self.tie_embeddings is not True
            or self.position_embedding != "learned"
        ):
            raise ValueError(
                "V2 requires RMSNorm, SwiGLU, bias-free Linear, tied weights and learned positions."
            )
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive.")


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, hidden):
        # Accumulate the mean square in FP32 during BF16 training.
        normalized = hidden.float()
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return normalized.to(hidden.dtype) * self.weight


class CausalAttentionV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.n_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.projection = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, hidden):
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).view(batch, length, 3, self.heads, width // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.projection(attended.transpose(1, 2).reshape(batch, length, width))


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, hidden):
        return self.down(F.silu(self.gate(hidden)) * self.up(hidden))


class DecoderBlockV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalAttentionV2(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden):
        hidden = hidden + self.dropout(self.attention(self.attention_norm(hidden)))
        return hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))


class TinyGPTV2(nn.Module):
    def __init__(self, config=GPTV2Config()):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(DecoderBlockV2(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.lm_head.weight = self.token_embedding.weight
        for block in self.blocks:
            for projection in (block.attention.projection, block.ffn.down):
                nn.init.normal_(projection.weight, std=0.02 / math.sqrt(2 * config.n_layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids, labels=None):
        if input_ids.ndim != 2 or not 1 <= input_ids.shape[1] <= self.config.context_length:
            raise ValueError("Expected [batch, time] input within model context.")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        if labels is None:
            return self.lm_head(hidden)
        if labels.shape != input_ids.shape:
            raise ValueError("Labels must match inputs; labels are already shifted by DataLoader.")
        valid = labels != -100
        logits = self.lm_head(hidden[valid])
        return {
            "loss_sum": F.cross_entropy(logits, labels[valid], reduction="sum"),
            "token_count": valid.sum(),
        }

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def configuration(self):
        return asdict(self.config)
