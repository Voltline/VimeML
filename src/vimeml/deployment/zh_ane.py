"""Neural Engine probe for vimeml-tiny-zh-v1 (step 0; not a deployment interface).

The Neural Engine runs FP16 programs with static shapes. The v3 graph computes in FP32, with flexible shapes
and a -1e9 mask, so CPU_AND_NE never placed it on the ANE. Two FP16 layouts, both with fixed shapes:

  bt   v3's math and v3's cache layouts, with an additive mask of -1e4 built with `where` (FP16 has no
       -1e9, and 0 * -inf is NaN). It shows what a direct FP16 conversion gets.
  ane  channels-first hidden states [B,384,1,T], every Linear as a 1x1 conv, attention per head with the key
       axis on dim 1, and RMSNorm on inputs scaled by 1/16 (the zh-v1 residual stream reaches about 53, so
       the sum of squares over 384 channels would overflow FP16). Caches: key (9,B,S,384) key-major, value
       (9,B,384,S) channel-major; the prefix uses the same layouts with batch 1.

Functions (each shape is its own function of one multifunction package):
  score_b{B}_t{T}_p{P}   input_ids, targets [B,T], prefix_length [1], prefix_key/value -> target_logprobs [B,T]
  step_b{B}_p{P}_s{S}    input_ids [B,1], position [B], key_cache/value_cache, prefix_length, prefix_key/value,
                         allowed [K] -> logsumexp [B], allowed_logits [B,K], new_key, new_value
Semantics are those of zh_v1_batch3 (shared prefix, row caches after it, masked slots finite).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from vimeml.deployment import zh_v1_batch3 as z3

LAYERS, HEADS, HEAD_DIM, VOCAB, WIDTH = z3.LAYERS, z3.HEADS, z3.HEAD_DIM, z3.VOCAB, z3.WIDTH
MASK = -1e4
RMS_SCALE = 1.0 / 16
ALLOWED = 256
LAYOUTS = ("bt", "ane")
SCORE_SHAPES = ((16, 8, 32), (16, 16, 32), (16, 32, 32), (16, 16, 64))       # (B, T, P)
STEP_SHAPES = ((16, 32, 16), (16, 64, 32))                                   # (B, P, S); T = 1


def score_name(b, t, p):
    return f"score_b{b}_t{t}_p{p}"


def step_name(b, p, s):
    return f"step_b{b}_p{p}_s{s}"


def functions():
    """{name: (kind, shape)} for the probe grid."""
    out = {score_name(*s): ("score", s) for s in SCORE_SHAPES}
    out.update({step_name(*s): ("step", s) for s in STEP_SHAPES})
    return out


def additive(condition, dtype=torch.float32):
    return torch.where(condition, torch.tensor(MASK, dtype=dtype), torch.tensor(0.0, dtype=dtype))


# ---- layout conversion (numpy or torch) -------------------------------------------------------------------

def key_to_ane(key):
    """(9,B,6,S,64) -> (9,B,S,384)."""
    n, b, h, s, d = key.shape
    if torch.is_tensor(key):
        return key.permute(0, 1, 3, 2, 4).reshape(n, b, s, h * d)
    return key.transpose(0, 1, 3, 2, 4).reshape(n, b, s, h * d)


def value_to_ane(value):
    """(9,B,6,S,64) -> (9,B,384,S)."""
    n, b, h, s, d = value.shape
    if torch.is_tensor(value):
        return value.permute(0, 1, 2, 4, 3).reshape(n, b, h * d, s)
    return value.transpose(0, 1, 2, 4, 3).reshape(n, b, h * d, s)


# ---- bt layout: v3 math, FP16-safe mask -------------------------------------------------------------------

def _bt_layers(model, input_ids, start, prefix, private):
    batch, tokens = input_ids.shape
    steps = torch.arange(tokens, dtype=torch.int32)
    hidden = model.token_embedding(input_ids) + model.position_embedding(start[:, None] + steps[None, :])
    length, prefix_key, prefix_value = prefix
    p_index = torch.arange(prefix_key.shape[3], dtype=torch.int32)
    dt = hidden.dtype
    prefix_mask = additive(p_index >= length, dt)[None, None, None, :]
    causal = additive(steps[None, :] > steps[:, None], dt)[None, None]
    if private is not None:
        position, key_cache, value_cache = private
        s_index = torch.arange(key_cache.shape[3], dtype=torch.int32)
        cache_mask = additive(s_index[None, :] >= position[:, None], dt)[:, None, None, :]
    keys, values = [], []
    for layer, block in enumerate(model.blocks):
        qkv = block.attention.qkv(block.attention_norm(hidden)).reshape(batch, tokens, 3, HEADS, HEAD_DIM)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query = query * (1.0 / math.sqrt(HEAD_DIM))
        scores = [torch.matmul(query, prefix_key[layer].transpose(-2, -1)) + prefix_mask,
                  torch.matmul(query, key.transpose(-2, -1)) + causal]
        vals = [prefix_value[layer].expand(batch, -1, -1, -1), value]
        if private is not None:
            scores.append(torch.matmul(query, key_cache[layer].transpose(-2, -1)) + cache_mask)
            vals.append(value_cache[layer])
        weights = torch.softmax(torch.cat(scores, -1), -1)
        attended = torch.matmul(weights, torch.cat(vals, 2))
        hidden = hidden + block.attention.projection(attended.transpose(1, 2).reshape(batch, tokens, WIDTH))
        hidden = hidden + block.ffn(block.ffn_norm(hidden))
        keys.append(key)
        values.append(value)
    return model.final_norm(hidden), torch.stack(keys), torch.stack(values)


# ---- ane layout ------------------------------------------------------------------------------------------

def _conv(x, linear):
    return F.conv2d(x, linear.weight[:, :, None, None])


def _rms(x, norm):
    xs = x * RMS_SCALE
    return xs * torch.rsqrt((xs * xs).mean(1, keepdim=True) + norm.eps * RMS_SCALE ** 2) * norm.weight[None, :, None, None]


def _ane_layers(model, input_ids, start, prefix, private):
    """Hidden [B,384,1,T] after the final norm; new keys (9,B,T,384) and values (9,B,384,T)."""
    batch, tokens = input_ids.shape
    steps = torch.arange(tokens, dtype=torch.int32)
    x = model.token_embedding(input_ids) + model.position_embedding(start[:, None] + steps[None, :])
    x = x.transpose(1, 2).unsqueeze(2)                                                    # [B,C,1,T]
    length, prefix_key, prefix_value = prefix                                             # (9,1,P,C), (9,1,C,P)
    p_index = torch.arange(prefix_key.shape[2], dtype=torch.int32)
    dt = x.dtype
    masks = [additive(p_index >= length, dt)[None, :, None], additive(steps[:, None] > steps[None, :], dt)[None]]  # [1,K,T]
    if private is not None:
        position, key_cache, value_cache = private                                        # (9,B,S,C), (9,B,C,S)
        s_index = torch.arange(key_cache.shape[2], dtype=torch.int32)
        masks.append(additive(s_index[None, :] >= position[:, None], dt)[:, :, None])         # [B,S,1]
    keys, values = [], []
    for layer, block in enumerate(model.blocks):
        a = block.attention
        qkv = _conv(_rms(x, block.attention_norm), a.qkv).squeeze(2)                       # [B,3C,T]
        q, k, v = qkv[:, :WIDTH] * (1.0 / math.sqrt(HEAD_DIM)), qkv[:, WIDTH:2 * WIDTH], qkv[:, 2 * WIDTH:]
        k_rows = k.transpose(1, 2)                                                         # [B,T,C]
        sources = [(prefix_key[layer], prefix_value[layer]), (k_rows, v)]
        if private is not None:
            sources.append((key_cache[layer], value_cache[layer]))
        heads = []
        for h in range(HEADS):
            c = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
            qh = q[:, c]                                                                   # [B,64,T]
            scores = torch.cat([torch.matmul(kk[:, :, c], qh) + m for (kk, _), m in zip(sources, masks)], 1)  # [B,K,T]
            w = torch.softmax(scores, 1)
            out, at = 0, 0
            for kk, vv in sources:
                n = kk.shape[1]
                out = out + torch.matmul(vv[:, c], w[:, at:at + n])                        # [B,64,T]
                at += n
            heads.append(out)
        x = x + _conv(torch.cat(heads, 1).unsqueeze(2), a.projection)
        f = _rms(x, block.ffn_norm)
        x = x + _conv(F.silu(_conv(f, block.ffn.gate)) * _conv(f, block.ffn.up), block.ffn.down)
        keys.append(k_rows)
        values.append(v)
    return _rms(x, model.final_norm), torch.stack(keys), torch.stack(values)


# ---- graphs ------------------------------------------------------------------------------------------------

class ScoreProbe(nn.Module):
    def __init__(self, model, layout):
        super().__init__()
        if layout not in LAYOUTS:
            raise ValueError(layout)
        self.model, self.layout = model.eval(), layout

    def forward(self, input_ids, targets, prefix_length, prefix_key, prefix_value):
        m = self.model
        start = prefix_length.expand(input_ids.shape[0])
        if self.layout == "bt":
            h, _, _ = _bt_layers(m, input_ids, start, (prefix_length, prefix_key, prefix_value), None)  # [B,T,C]
            logits = m.lm_head(h)
            picked = (h * m.token_embedding(targets)).sum(-1)
            top = logits.amax(-1)
            return picked - top - torch.log(torch.exp(logits - top[..., None]).sum(-1))
        h, _, _ = _ane_layers(m, input_ids, start, (prefix_length, prefix_key, prefix_value), None)      # [B,C,1,T]
        logits = _conv(h, m.lm_head)                                                                     # [B,V,1,T]
        picked = (h * m.token_embedding(targets).transpose(1, 2).unsqueeze(2)).sum(1)                    # [B,1,T]
        top = logits.amax(1, keepdim=True)
        lse = top + torch.log(torch.exp(logits - top).sum(1, keepdim=True))
        return (picked - lse.squeeze(1)).squeeze(1)


class StepProbe(nn.Module):
    def __init__(self, model, layout):
        super().__init__()
        if layout not in LAYOUTS:
            raise ValueError(layout)
        self.model, self.layout = model.eval(), layout

    def forward(self, input_ids, position, key_cache, value_cache, prefix_length, prefix_key, prefix_value, allowed):
        m = self.model
        layers = _bt_layers if self.layout == "bt" else _ane_layers
        h, keys, values = layers(m, input_ids, prefix_length + position, (prefix_length, prefix_key, prefix_value),
                                 (position, key_cache, value_cache))
        last = h[:, -1] if self.layout == "bt" else h[:, :, 0, -1]                                       # [B,C]
        logits = m.lm_head(last)
        top = logits.amax(-1, keepdim=True)
        lse = (top + torch.log(torch.exp(logits - top).sum(-1, keepdim=True))).squeeze(-1)
        return lse, torch.index_select(logits, 1, allowed), keys, values


def example(kind, shape, layout, seed=7):
    """Fixed-shape example inputs (torch) for tracing and checks."""
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g)
    if kind == "score":
        b, t, p = shape
        ids = torch.randint(8, VOCAB, (b, t), generator=g, dtype=torch.int32)
        tg = torch.randint(8, VOCAB, (b, t), generator=g, dtype=torch.int32)
        pk, pv = rnd(LAYERS, 1, HEADS, p, HEAD_DIM), rnd(LAYERS, 1, HEADS, p, HEAD_DIM)
        plen = torch.tensor([p - 3], dtype=torch.int32)
        if layout == "ane":
            pk, pv = key_to_ane(pk), value_to_ane(pv)
        return ids, tg, plen, pk, pv
    b, p, s = shape
    ids = torch.randint(8, VOCAB, (b, 1), generator=g, dtype=torch.int32)
    position = torch.tensor([i % (s + 1) for i in range(b)], dtype=torch.int32)
    kc, vc = rnd(LAYERS, b, HEADS, s, HEAD_DIM), rnd(LAYERS, b, HEADS, s, HEAD_DIM)
    pk, pv = rnd(LAYERS, 1, HEADS, p, HEAD_DIM), rnd(LAYERS, 1, HEADS, p, HEAD_DIM)
    plen = torch.tensor([p - 5], dtype=torch.int32)
    if layout == "ane":
        kc, vc, pk, pv = key_to_ane(kc), value_to_ane(vc), key_to_ane(pk), value_to_ane(pv)
    allowed = torch.randperm(VOCAB - 8, generator=g)[:ALLOWED].to(torch.int32) + 8
    return ids, position, kc, vc, plen, pk, pv, allowed


def graph(model, kind, layout):
    return (ScoreProbe if kind == "score" else StepProbe)(model, layout).eval()


@torch.inference_mode()
def trace(model, name, layout):
    kind, shape = functions()[name]
    g = graph(model, kind, layout)
    return torch.jit.trace(g, example(kind, shape, layout))
