"""Explicit attention conversion adapter. Does not edit or retrain TinyGPT."""
import math

import torch
from torch import nn


class ConversionGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
        config = model.config
        self.heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.width = config.d_model
        mask = torch.full((config.context_length, config.context_length), -float("inf"))
        self.register_buffer("causal_mask", torch.triu(mask, diagonal=1))

    def forward(self, input_ids):
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.model.token_embedding(input_ids) + self.model.position_embedding(positions)
        for block in self.model.blocks:
            normalized = block.attention_norm(hidden)
            qkv = block.attention.qkv(normalized).reshape(batch, length, 3, self.heads, self.head_dim)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            attention = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attention = torch.softmax(attention + self.causal_mask[:length, :length], dim=-1)
            attended = torch.matmul(attention, value).transpose(1, 2).reshape(batch, length, self.width)
            hidden = hidden + block.attention.projection(attended)
            hidden = hidden + block.ffn(block.ffn_norm(hidden))
        return self.model.lm_head(self.model.final_norm(hidden))


@torch.inference_mode()
def trace_graph(model):
    graph = ConversionGraph(model).eval()
    sample = torch.zeros((1, min(16, model.config.context_length)), dtype=torch.int32)
    traced = torch.jit.trace(graph, sample)
    # Trace must generalize lengths, including non-example sizes and 128.
    generator = torch.Generator().manual_seed(716)
    for length in sorted({1, min(7, model.config.context_length), model.config.context_length}):
        ids = torch.randint(model.config.vocab_size, (1, length), generator=generator, dtype=torch.int32)
        original = model(ids.long())
        for adapted in (graph(ids), traced(ids)):
            torch.testing.assert_close(adapted, original, atol=3e-5, rtol=3e-4)
    return traced
