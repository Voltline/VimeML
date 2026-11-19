"""CUDA Graph backend for `zh_search.search`: the decode step of any tiny_gpt zh model replayed from
captured graphs (desktop latency; evaluation only, no export).

Eager decode is bound by kernel launches (about 25 per layer), not by arithmetic. This backend
captures one graph per cache width (`widths`, default 32/64/128 slots) over a fixed batch of
`rows` (default 16, the beam): static token, position and cache buffers go in, the logits and the
caches with each row's new key written at its position come out. Unused rows are padding and their
outputs are ignored. Prefill stays eager (once per search).

A row state is (keys, values, row, length) as views into one call's output caches
([layers, rows, heads, width, head_dim], cloned after each replay); decode copies each source
call's rows into the static buffers with one indexed copy per source. Slots past a row's length are
masked, so the math is the eager backend's; only the summation order of the wider kernels differs.

dtype (float32, float16, bfloat16) sets the weights and caches. The step is bound by reading the
weights, so half precision roughly halves it; attention scores and softmax stay in float32 (the
additive mask is -1e9) and the logits are returned in float32.
"""
import math

import torch

MASKED = -1e9


class GraphBackend:
    def __init__(self, model, device="cuda", rows=16, widths=(32, 64, 128), dtype=torch.float32):
        if not str(device).startswith("cuda"):
            raise ValueError("CUDA graphs need a CUDA device")
        self.dtype = dtype
        self.model = model.to(device=device, dtype=dtype).eval()
        self.device = device
        c = model.config
        self.heads, self.width, self.layers = c.n_heads, c.d_model, c.n_layers
        self.head_dim = c.d_model // c.n_heads
        self.context = c.context_length
        self.rows = rows
        self.widths = tuple(w for w in widths if w <= self.context) or (self.context,)
        if self.widths[-1] < self.context:
            self.widths += (self.context,)
        self.calls = self.used_rows = 0
        self.graphs = {w: self._capture(w) for w in self.widths}

    # ---- shared math ---------------------------------------------------------------------------------

    def _qkv(self, block, hidden):
        batch, size, _ = hidden.shape
        qkv = block.attention.qkv(block.attention_norm(hidden)).reshape(batch, size, 3, self.heads, self.head_dim)
        return qkv.permute(2, 0, 3, 1, 4).unbind(0)

    def _step(self, tokens, position, key_cache, value_cache):
        m, slots = self.model, key_cache.shape[3]
        index = torch.arange(slots, device=self.device)
        write = (index[None, :] == position[:, None])[:, None, :, None]
        mask = (index[None, :] > position[:, None]).float()[:, None, None, :] * MASKED
        hidden = (m.token_embedding(tokens) + m.position_embedding(position))[:, None]
        keys, values = [], []
        for layer, block in enumerate(m.blocks):
            query, key, value = self._qkv(block, hidden)
            full_key = torch.where(write, key, key_cache[layer])
            full_value = torch.where(write, value, value_cache[layer])
            scores = torch.matmul(query, full_key.transpose(-2, -1)).float() / math.sqrt(self.head_dim) + mask
            attended = torch.matmul(torch.softmax(scores, -1).to(self.dtype), full_value)
            hidden = hidden + block.attention.projection(attended.transpose(1, 2).reshape(self.rows, 1, self.width))
            hidden = hidden + block.ffn(block.ffn_norm(hidden))
            keys.append(full_key)
            values.append(full_value)
        logits = m.lm_head(m.final_norm(hidden[:, 0])).float()
        return logits, torch.stack(keys), torch.stack(values)

    @torch.inference_mode()
    def _capture(self, slots):
        shape = (self.layers, self.rows, self.heads, slots, self.head_dim)
        io = {"tokens": torch.zeros(self.rows, dtype=torch.long, device=self.device),
              "position": torch.zeros(self.rows, dtype=torch.long, device=self.device),
              "keys": torch.zeros(shape, device=self.device, dtype=self.dtype),
              "values": torch.zeros(shape, device=self.device, dtype=self.dtype)}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._step(io["tokens"], io["position"], io["keys"], io["values"])
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            io["out"] = self._step(io["tokens"], io["position"], io["keys"], io["values"])
        io["graph"] = graph
        return io

    # ---- search interface ----------------------------------------------------------------------------

    @torch.inference_mode()
    def prefill(self, ids):
        if not 1 <= len(ids) <= self.context:
            raise ValueError("Prompt outside the model context")
        m = self.model
        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        positions = torch.arange(len(ids), device=self.device)
        hidden = m.token_embedding(tokens) + m.position_embedding(positions)
        causal = positions[:, None] >= positions[None, :]
        keys, values = [], []
        for block in m.blocks:
            query, key, value = self._qkv(block, hidden)
            scores = torch.matmul(query, key.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
            attended = torch.matmul(torch.softmax(scores.masked_fill(~causal, -torch.inf), -1).to(self.dtype), value)
            hidden = hidden + block.attention.projection(attended.transpose(1, 2).reshape(1, len(ids), self.width))
            hidden = hidden + block.ffn(block.ffn_norm(hidden))
            keys.append(key[0])
            values.append(value[0])
        logits = m.lm_head(m.final_norm(hidden[0, -1]))
        return logits.float(), (torch.stack(keys)[:, None], torch.stack(values)[:, None], 0, len(ids))

    @torch.inference_mode()
    def decode(self, tokens, states):
        batch = len(states)
        if batch > self.rows:
            raise ValueError(f"More than {self.rows} rows")
        lengths = [s[3] for s in states]
        if max(lengths) + 1 > self.context:
            raise ValueError("Context overflow")
        slots = next(w for w in self.widths if w >= max(lengths) + 1)
        io = self.graphs[slots]
        groups = {}
        for r, s in enumerate(states):
            g = groups.setdefault(id(s[0]), (s[0], s[1], [], []))
            g[2].append(s[2])
            g[3].append(r)
        # One host-to-device copy carries the tokens, positions and every group's source and target rows.
        pad = self.rows - batch
        flat = list(tokens) + [0] * pad + lengths + [0] * pad
        for _, _, rows, places in groups.values():
            flat += rows + places
        flat = torch.tensor(flat, dtype=torch.long).to(self.device, non_blocking=True)
        io["tokens"].copy_(flat[:self.rows])
        io["position"].copy_(flat[self.rows:2 * self.rows])
        at = 2 * self.rows
        for keys, values, rows, _ in groups.values():
            width, size = min(keys.shape[3], slots), len(rows)
            src, dst = flat[at:at + size], flat[at + size:at + 2 * size]
            at += 2 * size
            io["keys"][:, dst, :, :width] = keys[:, src, :, :width]
            io["values"][:, dst, :, :width] = values[:, src, :, :width]
        io["graph"].replay()
        logits, keys, values = io["out"]
        keys, values = keys.clone(), values.clone()
        self.calls += 1
        self.used_rows += batch
        return logits[:batch].clone(), [(keys, values, r, lengths[r] + 1) for r in range(batch)]
