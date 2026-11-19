"""zh-800m arch2 decoder (docs/training-zh-800m-arch2-plan.md, design 01ab0c8).

d1792, 23 layers, GQA with 28 query heads and 7 KV heads of 64, per-head RMSNorm on Q and K (QK-norm),
RoPE (theta 10000, all 64 dimensions) instead of learned positions, SwiGLU 4608, RMSNorm pre-norm, no
bias, tied embeddings; the 12K vocabulary and a segment limit of 512 (776,524,928 parameters).

forward(ids) is the ordinary causal forward (rows up to 512). forward(ids, segments=S) is the packed
training forward: rows up to PACK_LENGTH, S [batch, time] gives each position's segment id (contiguous
runs; padding is a segment of its own and carries no labels); positions restart at 0 in each segment and
attention is causal within a segment only, so every segment's tokens see exactly what they would see as a
row of their own. On CUDA the mask runs through flex_attention, elsewhere through a dense SDPA mask.
forward_cached is the FP32 KV path (cache [layers, batch, 7, 512, 64]); layer_qkv and embed let the
shape-generic search backend run the model.

Arch2Decoder takes any shape (tests use small ones); TinyGPTZh800mArch2 is the registered shape only.
"""
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from vimeml.training.model_v2 import RMSNorm, SwiGLU, token_loss

D_MODEL, HEADS, KV_HEADS, HEAD_DIM, D_FF, LAYERS = 1792, 28, 7, 64, 4608, 23
VOCAB, CONTEXT, PACK_LENGTH = 12288, 512, 2048
# Q and O d·d each, K and V d·(KV·64) each, SwiGLU 3·d·d_ff, two RMSNorms, two QK-norms of 64
LAYER_PARAMETERS = 2 * D_MODEL ** 2 + 2 * D_MODEL * KV_HEADS * HEAD_DIM + 3 * D_MODEL * D_FF + 2 * D_MODEL + 2 * HEAD_DIM
PARAMETERS = 776_524_928  # V·d (tied) + L·LAYER_PARAMETERS + d (final norm)


@dataclass(frozen=True)
class Arch2Config:
    vocab_size: int = VOCAB
    context_length: int = CONTEXT
    d_model: int = D_MODEL
    n_heads: int = HEADS
    n_kv_heads: int = KV_HEADS
    n_layers: int = LAYERS
    d_ff: int = D_FF
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    pack_length: int = PACK_LENGTH
    dropout: float = 0.0

    def __post_init__(self):
        if min(self.vocab_size, self.context_length, self.d_model, self.n_heads, self.n_kv_heads, self.n_layers,
               self.d_ff) < 1 or self.d_model % self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError("arch2 needs positive sizes, d_model divisible by heads, heads by KV heads")
        if (self.d_model // self.n_heads) % 2 or self.pack_length < self.context_length or self.dropout != 0.0:
            raise ValueError("arch2 needs an even head size, pack_length >= context and no dropout")

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


@dataclass(frozen=True)
class GPTZh800mArch2Config(Arch2Config):
    def __post_init__(self):
        super().__post_init__()
        if asdict(self) != asdict(Arch2Config()):
            raise ValueError("zh-800m arch2 shape is fixed (12K, d1792, L23, Q28/KV7, SwiGLU 4608, c512, pack 2048)")


def rope(x, positions, theta):
    """x [batch, heads, time, dim]; positions [batch, time] (or [time]); rotate-half RoPE in FP32."""
    dim = x.shape[-1]
    inverse = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    angles = positions.to(torch.float32)[..., None] * inverse              # [..., time, dim/2]
    angles = torch.cat([angles, angles], -1)
    if angles.ndim == 2:
        angles = angles[None]
    cos, sin = angles.cos()[:, None], angles.sin()[:, None]                # [batch|1, 1, time, dim]
    xf = x.float()
    half = dim // 2
    rotated = torch.cat([-xf[..., half:], xf[..., :half]], -1)
    return (xf * cos + rotated * sin).to(x.dtype)


def segment_positions(segments):
    """Position of every token inside its segment (contiguous runs of equal ids), [batch, time]."""
    time = segments.shape[1]
    index = torch.arange(time, device=segments.device).expand_as(segments)
    start = torch.ones_like(segments, dtype=torch.bool)
    start[:, 1:] = segments[:, 1:] != segments[:, :-1]
    first = torch.where(start, index, torch.zeros_like(index)).cummax(1).values
    return index - first


def flex_block_mask(segments):
    from torch.nn.attention.flex_attention import create_block_mask

    def same_segment_causal(b, h, q, k):
        return (segments[b, q] == segments[b, k]) & (q >= k)
    batch, time = segments.shape
    return create_block_mask(same_segment_causal, batch, None, time, time, device=segments.device)


class GQAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = config.n_heads, config.n_kv_heads, config.head_dim
        self.theta = config.rope_theta
        self.query = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.key = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.value = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.projection = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.query_norm = RMSNorm(self.head_dim, config.norm_eps)
        self.key_norm = RMSNorm(self.head_dim, config.norm_eps)

    def qkv(self, hidden, positions):
        """query [B, heads, T, D], key and value [B, kv_heads, T, D]; QK-norm, then RoPE on query and key."""
        batch, time = hidden.shape[:2]
        query = self.query_norm(self.query(hidden).view(batch, time, self.heads, self.head_dim)).transpose(1, 2)
        key = self.key_norm(self.key(hidden).view(batch, time, self.kv_heads, self.head_dim)).transpose(1, 2)
        value = self.value(hidden).view(batch, time, self.kv_heads, self.head_dim).transpose(1, 2)
        # RMSNorm multiplies its FP32 weight, so under BF16 autocast query and key come out FP32 while value
        # stays BF16; flex_attention and SDPA need one dtype. rope keeps its input dtype, so cast after it.
        return (rope(query, positions, self.theta).to(value.dtype),
                rope(key, positions, self.theta).to(value.dtype), value)

    def forward(self, hidden, positions, mask=None):
        """mask None: causal; a BlockMask: flex_attention; a bool tensor [B, 1, T, T]: dense SDPA."""
        batch, time = hidden.shape[:2]
        query, key, value = self.qkv(hidden, positions)
        if mask is None:
            out = F.scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True)
        elif isinstance(mask, torch.Tensor):
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, enable_gqa=True)
        else:
            from torch.nn.attention.flex_attention import flex_attention
            out = flex_attention(query, key, value, block_mask=mask, enable_gqa=True)
        return self.projection(out.transpose(1, 2).reshape(batch, time, self.heads * self.head_dim))


class Arch2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = GQAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, hidden, positions, mask=None):
        hidden = hidden + self.attention(self.attention_norm(hidden), positions, mask)
        return hidden + self.ffn(self.ffn_norm(hidden))


@dataclass
class KVCacheArch2:
    key: torch.Tensor
    value: torch.Tensor
    length: int = 0

    def clone(self):
        return KVCacheArch2(self.key.clone(), self.value.clone(), self.length)


class Arch2Decoder(nn.Module):
    packed_training = True  # joint_zh200m packs this model's training rows (segments)

    def __init__(self, config=Arch2Config()):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Arch2Block(config) for _ in range(config.n_layers))
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

    def attention_mask(self, segments):
        if segments.is_cuda:
            return flex_block_mask(segments)
        time = segments.shape[1]
        causal = torch.ones(time, time, dtype=torch.bool, device=segments.device).tril()
        return ((segments[:, :, None] == segments[:, None, :]) & causal)[:, None]

    def positions_and_mask(self, input_ids, segments=None):
        if input_ids.ndim != 2:
            raise ValueError("Expected [batch, time] tokens")
        time = input_ids.shape[1]
        if segments is None:
            if not 1 <= time <= self.config.context_length:
                raise ValueError("Expected [batch, time] input within model context")
            return torch.arange(time, device=input_ids.device), None
        if segments.shape != input_ids.shape or not 1 <= time <= self.config.pack_length:
            raise ValueError("Packed rows need segments shaped like the tokens, within pack_length")
        return segment_positions(segments), self.attention_mask(segments)

    def hidden_states(self, input_ids, positions, mask):
        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden, positions, mask)
        return self.final_norm(hidden)

    def forward(self, input_ids, labels=None, token_weights=None, segments=None):
        hidden = self.hidden_states(input_ids, *self.positions_and_mask(input_ids, segments))
        if labels is None:
            return self.lm_head(hidden)
        return token_loss(self.lm_head, hidden, input_ids, labels, token_weights)

    # ---- inference ---------------------------------------------------------------------------------------

    def embed(self, tokens, positions=None):
        return self.token_embedding(tokens)

    def layer_qkv(self, block, hidden, positions):
        """Normalized input of `block` -> query, key, value as the attention computes them (for KV backends)."""
        return block.attention.qkv(block.attention_norm(hidden), positions)

    def new_cache(self, batch_size=1):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Positive integer cache batch required")
        c = self.config
        shape = (c.n_layers, batch_size, c.n_kv_heads, c.context_length, c.head_dim)
        device = self.token_embedding.weight.device
        return KVCacheArch2(torch.zeros(shape, device=device), torch.zeros(shape, device=device))

    @torch.inference_mode()
    def forward_cached(self, input_ids, cache):
        """FP32 research KV path; matches forward() on the same prefix."""
        if self.training or self.token_embedding.weight.dtype != torch.float32:
            raise ValueError("KV path requires eval mode and FP32 weights")
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected integer [batch,time] tokens")
        c = self.config
        batch, size = input_ids.shape
        if (cache.key.shape != (c.n_layers, batch, c.n_kv_heads, c.context_length, c.head_dim) or size < 1
                or cache.length + size > c.context_length):
            raise ValueError("Invalid KV cache or context overflow")
        if int(input_ids.min()) < 0 or int(input_ids.max()) >= c.vocab_size:
            raise ValueError("Token outside vocabulary")
        start, end = cache.length, cache.length + size
        positions = torch.arange(start, end, device=input_ids.device)
        causal = positions[:, None] >= torch.arange(end, device=input_ids.device)[None, :]
        hidden = self.token_embedding(input_ids)
        for layer, block in enumerate(self.blocks):
            query, key, value = self.layer_qkv(block, hidden, positions)
            cache.key[layer, :, :, start:end].copy_(key)
            cache.value[layer, :, :, start:end].copy_(value)
            attended = F.scaled_dot_product_attention(query, cache.key[layer, :, :, :end], cache.value[layer, :, :, :end],
                                                      attn_mask=causal, enable_gqa=True)
            hidden = hidden + block.attention.projection(attended.transpose(1, 2).reshape(batch, size, c.d_model))
            hidden = hidden + block.ffn(block.ffn_norm(hidden))
        cache.length = end
        return self.lm_head(self.final_norm(hidden))

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def configuration(self):
        return asdict(self.config)


class TinyGPTZh800mArch2(Arch2Decoder):
    def __init__(self, config=GPTZh800mArch2Config()):
        if type(config) is not GPTZh800mArch2Config:
            raise ValueError("zh-800m arch2 requires GPTZh800mArch2Config")
        super().__init__(config)
        if self.parameter_count() != PARAMETERS or any(
                sum(p.numel() for p in block.parameters()) != LAYER_PARAMETERS for block in self.blocks):
            raise ValueError("zh-800m arch2 parameter budget does not match its contract")


def compile_backbone(model):
    """Compile the hidden-state stack (positions and the mask are built eagerly and passed in, so the ordinary,
    dense-mask and flex BlockMask forwards each get a graph); the valid-token head and loss stay eager."""
    from types import MethodType
    compiled = torch.compile(model.hidden_states, fullgraph=True, dynamic=True)

    def forward(module, input_ids, labels=None, token_weights=None, segments=None):
        hidden = compiled(input_ids, *module.positions_and_mask(input_ids, segments))
        if labels is None:
            return module.lm_head(hidden)
        return token_loss(module.lm_head, hidden, input_ids, labels, token_weights)

    model.forward = MethodType(forward, model)
    return {"implementation": "torch.compile hidden stack (positions, mask eager; flex_attention on CUDA); "
                              "eager valid-token head and CE",
            "dynamic_shapes": True, "state_dict_unchanged": True, "checkpoint_inference": "Plain Arch2Decoder"}


def pack(sequences, length=PACK_LENGTH, limit=CONTEXT):
    """First-fit-decreasing packing of [(input_ids, labels), ...] (1-D tensors of at most `limit` tokens) into
    rows of `length`. Returns (input_ids, labels, segments, placement): sequence i is segment i, padding is
    id 0 / label -100 / segment -1 (one run at the end of a row, a segment of its own), and
    placement[i] = (row, offset)."""
    order = sorted(range(len(sequences)), key=lambda i: (-len(sequences[i][0]), i))
    free, placement = [], [None] * len(sequences)
    for i in order:
        size = len(sequences[i][0])
        if not 1 <= size <= min(length, limit):
            raise ValueError("A sequence empty or longer than the segment limit")
        row = next((r for r, left in enumerate(free) if left >= size), None)
        if row is None:
            free.append(length)
            row = len(free) - 1
        placement[i] = (row, length - free[row])
        free[row] -= size
    rows = len(free)
    ids = torch.zeros(rows, length, dtype=torch.long)
    labels = torch.full((rows, length), -100, dtype=torch.long)
    segments = torch.full((rows, length), -1, dtype=torch.long)
    for i, (row, offset) in enumerate(placement):
        tokens, targets = sequences[i][0], sequences[i][1]
        ids[row, offset:offset + len(tokens)] = tokens
        labels[row, offset:offset + len(tokens)] = targets
        segments[row, offset:offset + len(tokens)] = i
    return ids, labels, segments, placement
