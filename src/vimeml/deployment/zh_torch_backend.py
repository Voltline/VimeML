"""Shape-generic PyTorch backend for `zh_search.search` (any tiny_gpt zh model: 20M, 200M, 800M).

`zh_v1_batch.TorchBatchBackend` is tied to the vimeml-tiny-zh-v1 shape because it reuses the Core ML
graphs. This backend reads layers, heads and widths from the model and runs the same two calls in
FP32:

- `prefill(ids)`: one causal pass over the prompt; returns the last logits and the row state
  (key, value), each [layers, heads, n, head_dim].
- `decode(tokens, states)`: one batched step. Rows may sit at different positions; caches are
  right-padded to the longest row, and padded slots are masked out.

Evaluation only (no export). Both calls match the model's full forward on the same sequence. A model with
`layer_qkv` and `embed` (zh-800m arch2: RoPE, GQA, QK-norm) supplies its own query/key/value per layer;
row states then hold its KV heads only, and attention repeats each KV head for its query group.
"""
import math

import torch

MASKED = -1e9


class ModelBatchBackend:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device).float().eval()
        self.device = device
        config = model.config
        self.heads = config.n_heads
        self.kv_heads = getattr(config, "n_kv_heads", config.n_heads)
        self.width = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.context = config.context_length
        self.native = hasattr(model, "layer_qkv")
        self.calls = self.rows = 0

    def _embed(self, tokens, positions):
        m = self.model
        return m.embed(tokens, positions) if self.native else m.token_embedding(tokens) + m.position_embedding(positions)

    def _qkv(self, block, hidden, positions):
        """query [B,H,T,D]; key, value [B,KV,T,D] (KV = H unless the model groups its heads)."""
        if self.native:
            return self.model.layer_qkv(block, hidden, positions)
        batch, size = hidden.shape[:2]
        qkv = block.attention.qkv(block.attention_norm(hidden)).reshape(batch, size, 3, self.heads, self.head_dim)
        return qkv.permute(2, 0, 3, 1, 4).unbind(0)                       # each [B,H,T,D]

    def _expand(self, kv):
        return kv if self.kv_heads == self.heads else kv.repeat_interleave(self.heads // self.kv_heads, dim=-3)

    @torch.inference_mode()
    def prefill(self, ids):
        if not 1 <= len(ids) <= self.context:
            raise ValueError("Prompt outside the model context")
        m = self.model
        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        positions = torch.arange(len(ids), device=self.device)
        hidden = self._embed(tokens, positions)
        causal = positions[:, None] >= positions[None, :]
        keys, values = [], []
        for block in m.blocks:
            query, key, value = self._qkv(block, hidden, positions)
            scores = torch.matmul(query, self._expand(key).transpose(-2, -1)) / math.sqrt(self.head_dim)
            attended = torch.matmul(torch.softmax(scores.masked_fill(~causal, -torch.inf), -1), self._expand(value))
            hidden = hidden + block.attention.projection(attended.transpose(1, 2).reshape(1, len(ids), self.width))
            hidden = hidden + block.ffn(block.ffn_norm(hidden))
            keys.append(key[0])
            values.append(value[0])
        logits = m.lm_head(m.final_norm(hidden[0, -1]))
        return logits.float(), (torch.stack(keys), torch.stack(values))

    @torch.inference_mode()
    def decode(self, tokens, states):
        m = self.model
        batch = len(states)
        lengths = [s[0].shape[2] for s in states]
        if max(lengths) + 1 > self.context:
            raise ValueError("Context overflow")
        slots = max(lengths) + 1
        layers = len(m.blocks)
        key_cache = torch.zeros(layers, batch, self.kv_heads, slots, self.head_dim, device=self.device)
        value_cache = torch.zeros_like(key_cache)
        for r, (k, v) in enumerate(states):
            key_cache[:, r, :, :lengths[r]] = k
            value_cache[:, r, :, :lengths[r]] = v
        position = torch.tensor(lengths, device=self.device)
        index = torch.arange(slots, device=self.device)
        write = (index[None, :] == position[:, None])[:, None, :, None]           # [B,1,S,1]
        mask = (index[None, :] > position[:, None]).float()[:, None, None, :] * MASKED  # [B,1,1,S]
        hidden = self._embed(torch.tensor(tokens, device=self.device), position)[:, None]
        new_keys, new_values = [], []
        for layer, block in enumerate(m.blocks):
            query, key, value = self._qkv(block, hidden, position[:, None])          # [B,H|KV,1,D]
            full_key = torch.where(write, key, key_cache[layer])
            full_value = torch.where(write, value, value_cache[layer])
            scores = torch.matmul(query, self._expand(full_key).transpose(-2, -1)) / math.sqrt(self.head_dim) + mask
            attended = torch.matmul(torch.softmax(scores, -1), self._expand(full_value))
            hidden = hidden + block.attention.projection(attended.transpose(1, 2).reshape(batch, 1, self.width))
            hidden = hidden + block.ffn(block.ffn_norm(hidden))
            new_keys.append(key[:, :, 0])
            new_values.append(value[:, :, 0])
        logits = m.lm_head(m.final_norm(hidden[:, 0]))
        nk, nv = torch.stack(new_keys), torch.stack(new_values)                     # [L,B,H,D]
        self.calls += 1
        self.rows += batch
        new = [(torch.cat([k, nk[:, r, :, None]], 2), torch.cat([v, nv[:, r, :, None]], 2))
               for r, (k, v) in enumerate(states)]
        return logits.float(), new
