"""Small pre-norm decoder-only GPT, with tied embedding/output weights."""
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 16384
    context_length: int = 128
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.0

    def __post_init__(self):
        if min(self.vocab_size, self.context_length, self.d_model, self.n_heads, self.n_layers, self.d_ff) < 1:
            raise ValueError("Model dimensions must be positive.")
        if self.d_model % self.n_heads or not 0 <= self.dropout < 1:
            raise ValueError("Invalid attention heads or dropout.")


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.n_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.projection = nn.Linear(config.d_model, config.d_model)

    def forward(self, hidden):
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).view(batch, length, 3, self.heads, width // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # Right padding is always AFTER valid tokens. Causality prevents a valid
        # query from seeing PAD, so no dense per-sample mask is necessary.
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        return self.projection(attended.transpose(1, 2).reshape(batch, length, width))


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalAttention(config)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(nn.Linear(config.d_model, config.d_ff),
                                nn.GELU(), nn.Linear(config.d_ff, config.d_model))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden):
        hidden = hidden + self.dropout(self.attention(self.attention_norm(hidden)))
        return hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))


class TinyGPT(nn.Module):
    def __init__(self, config=GPTConfig()):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.lm_head.weight = self.token_embedding.weight
        for block in self.blocks:
            for projection in (block.attention.projection, block.ffn[2]):
                nn.init.normal_(projection.weight, std=0.02 / math.sqrt(2 * config.n_layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

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
        # Avoid allocating [B,T,V] logits for PAD positions; supervision is unchanged.
        logits = self.lm_head(hidden[valid])
        return {"loss_sum": F.cross_entropy(logits, labels[valid], reduction="sum"),
                "token_count": valid.sum()}

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def configuration(self):
        return asdict(self.config)
