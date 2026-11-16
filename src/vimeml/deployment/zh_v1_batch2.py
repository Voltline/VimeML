"""vimeml-tiny-zh-v1 batched KV step, version 2: read-only caches, optional prefill, optional scored head.

Changes against `zh_v1_batch` (v1), after Vime's on-device memory report (2026-11-15):

1. The caches are never merged with the new keys. v1 built `cache*(1-write) + key*write` per layer, a
   full (B,6,S,64) copy of each cache per layer that Core ML may allocate at the largest shape. v2 attends
   to the cache (slots >= position[b] masked) and to the new tokens separately and combines the two parts
   with one shared softmax normalizer, so no cache-sized intermediate exists.
2. T new tokens per row (input_ids [B,T]), causal among themselves. T=1 is the decode step; B=1 with T>1
   is the prompt prefill, so the chunk package is no longer needed. Logits are returned for the last new
   token only.
3. Optional `scored` head: a shared `allowed` int32 [K] list (all rows of a search bucket sit at the same
   key position, so they share one allowed set; median 149 tokens, max 536 on the r2 fixture). Outputs
   `logsumexp` [B] over all 12,288 logits and `allowed_logits` [B,K]; log P(token) = allowed_logits -
   logsumexp. The head matmul still runs in full (the normalizer needs every logit).
4. Optional `sequence` head (2026-11-16, sentence scoring for typo correction): `targets` int32 [B,T], the
   token that follows each new token. Output `target_logprobs` [B,T], log P(targets[b,t] | row b up to and
   including input_ids[b,t]), at every new position. Scoring a known continuation is then one call per
   batch of rows instead of one call per token (see `score_continuations`). Rows of different lengths are
   right-padded (any token, any target); padded positions come after the real ones, so causality keeps
   them out of the real scores.

Interface (Core ML names):
  input_ids    int32 [B,T]            T new tokens per row (T=1 for decode)
  position     int32 [B]              row cache length before these tokens; position[b]+T <= 128
  key_cache    float32 (9,B,6,S,64)   slots >= position[b] of row b are masked but must be finite (zero new buffers): NaN/Inf there reaches the outputs; S >= max(position), S >= 1
  value_cache  float32 (9,B,6,S,64)
  allowed      int32 [K]              scored head only
  targets      int32 [B,T]            sequence head only
  -> logits    float32 [B,12288]      logits head: last new token, no softmax
  -> logsumexp float32 [B], allowed_logits float32 [B,K]     scored head
  -> target_logprobs float32 [B,T]                           sequence head
  -> new_key, new_value float32 (9,B,6,T,64)   keys/values of the new tokens; the caller writes them into
                                               slots position[b]..position[b]+T-1 of row b
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from vimeml.deployment import v2, zh_v1
from vimeml.deployment import zh_v1_batch as v1
from vimeml.deployment.bundle import read_json, write_json
from vimeml.deployment.coreml import coremltools, inspect_spec
from vimeml.training.data import file_sha

FORMAT = "vimeml_coreml_zh_v1_kv_batch_v2"
ALIGNMENT_FORMAT = "vimeml_kv_batch_alignment_zh_v1_v2"
VOCAB, LAYERS, CONTEXT, HEADS, HEAD_DIM, MAX_BATCH = v1.VOCAB, v1.LAYERS, v1.CONTEXT, v1.HEADS, v1.HEAD_DIM, v1.MAX_BATCH
MAX_ALLOWED = 1024
TOLERANCE = v1.TOLERANCE
MASKED = v1.MASKED
HEADS_V2 = ("logits", "scored", "sequence")
SEQUENCE_ROWS = 16


class KVBatchGraphZhV2(nn.Module):
    """See the module docstring. head: "logits", "scored" or "sequence"."""

    def __init__(self, model, head="logits"):
        super().__init__()
        if head not in HEADS_V2:
            raise ValueError(head)
        self.model = model.eval()
        self.head = head

    def forward(self, input_ids, position, key_cache, value_cache, extra=None):
        """extra: `allowed` [K] for the scored head, `targets` [B,T] for the sequence head."""
        batch, tokens = input_ids.shape
        slots = key_cache.shape[3]
        steps = torch.arange(tokens, dtype=torch.int32, device=input_ids.device)
        index = torch.arange(slots, dtype=torch.int32, device=input_ids.device)
        hidden = self.model.token_embedding(input_ids) + self.model.position_embedding(position[:, None] + steps[None, :])
        cache_mask = (index[None, :] >= position[:, None]).to(hidden.dtype)[:, None, None, :] * MASKED   # [B,1,1,S]
        causal = (steps[None, :] > steps[:, None]).to(hidden.dtype)[None, None] * MASKED                # [1,1,T,T]
        scale = 1.0 / math.sqrt(HEAD_DIM)
        keys, values = [], []
        for layer, block in enumerate(self.model.blocks):
            qkv = block.attention.qkv(block.attention_norm(hidden)).reshape(batch, tokens, 3, HEADS, HEAD_DIM)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)                    # [B,6,T,64]
            old = torch.matmul(query, key_cache[layer].transpose(-2, -1)) * scale + cache_mask   # [B,6,T,S]
            new = torch.matmul(query, key.transpose(-2, -1)) * scale + causal                    # [B,6,T,T]
            top = torch.maximum(old.amax(-1, keepdim=True), new.amax(-1, keepdim=True))
            old, new = torch.exp(old - top), torch.exp(new - top)
            total = old.sum(-1, keepdim=True) + new.sum(-1, keepdim=True)
            attended = (torch.matmul(old, value_cache[layer]) + torch.matmul(new, value)) / total  # [B,6,T,64]
            hidden = hidden + block.attention.projection(attended.transpose(1, 2).reshape(batch, tokens, 384))
            hidden = hidden + block.ffn(block.ffn_norm(hidden))
            keys.append(key)
            values.append(value)
        if self.head == "sequence":
            logits = self.model.lm_head(self.model.final_norm(hidden))                 # [B,T,V]
            picked = torch.gather(logits, 2, extra.long().unsqueeze(-1)).squeeze(-1)    # [B,T]
            return picked - torch.logsumexp(logits, -1), torch.stack(keys), torch.stack(values)
        logits = self.model.lm_head(self.model.final_norm(hidden[:, -1]))              # [B,V]
        if self.head == "scored":
            return torch.logsumexp(logits, -1), torch.index_select(logits, 1, extra), torch.stack(keys), torch.stack(values)
        return logits, torch.stack(keys), torch.stack(values)


def step_view(fn):
    """v1.check_graph-compatible view of a logits-head v2 step: new keys (9,B,6,1,64) -> (9,B,6,64)."""
    def run(tokens, position, k, v):
        logits, nk, nv = fn(tokens, position, k, v)
        return logits, torch.as_tensor(np.asarray(nk))[:, :, :, 0], torch.as_tensor(np.asarray(nv))[:, :, :, 0]
    return run


PREFILL_LENGTHS = lambda batch: [2 + (5 * b) % 17 for b in range(batch)]       # first chunk, per row


@torch.inference_mode()
def check_prefill(model, fn, batch=4, tokens=6, seed=1116, strict=True):
    """Two multi-token calls per case. First: each row alone (B=1, S=1, position 0) prefills a prompt of
    its own length. Second: all rows together advance `tokens` more tokens from their different
    positions. Last-token logits of each call vs `model`'s full forward. fn: logits-head v2 step."""
    lengths = PREFILL_LENGTHS(batch)
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(8, VOCAB, (batch, max(lengths) + tokens), generator=g, dtype=torch.int32)
    worst, outside, argmax, rows = 0.0, 0, 0, 0
    caches = []

    def judge(got, want):
        nonlocal worst, outside, argmax, rows
        worst = max(worst, float((got - want).abs().max()))
        rows += 1
        argmax += int(got.argmax() == want.argmax())
        if not torch.allclose(got, want, atol=TOLERANCE, rtol=TOLERANCE):
            outside += 1
            if strict:
                torch.testing.assert_close(got, want, atol=TOLERANCE, rtol=TOLERANCE)

    for b in range(batch):
        n = lengths[b]
        junk = torch.full((LAYERS, 1, HEADS, 1, HEAD_DIM), 30.0)
        logits, nk, nv = (torch.as_tensor(np.asarray(x)).float() for x in
                          fn(ids[b:b + 1, :n], torch.tensor([0], dtype=torch.int32), junk, -junk))
        judge(logits[0], torch.as_tensor(model(ids[b:b + 1, :n].long()))[0, -1].float())
        caches.append((nk[:, 0], nv[:, 0]))                                          # (9,6,n,64)
    width = max(lengths) + 2
    k = torch.full((LAYERS, batch, HEADS, width, HEAD_DIM), 30.0)
    v = torch.full_like(k, -30.0)
    for b, (ck, cv) in enumerate(caches):
        k[:, b, :, :lengths[b]] = ck
        v[:, b, :, :lengths[b]] = cv
    chunk = torch.stack([ids[b, lengths[b]:lengths[b] + tokens] for b in range(batch)])
    logits, _, _ = (torch.as_tensor(np.asarray(x)).float() for x in
                    fn(chunk, torch.tensor(lengths, dtype=torch.int32), k, v))
    for b in range(batch):
        n = lengths[b] + tokens
        judge(logits[b], torch.as_tensor(model(ids[b:b + 1, :n].long()))[0, -1].float())
    return {"prefill_rows": batch, "tokens_second_call": tokens, "rows": rows, "max_abs": worst,
            "rows_outside_tolerance": outside, "argmax_equal": argmax}


@torch.inference_mode()
def check_scored(model, seed=1117):
    """The scored head equals log-softmax of the logits head on the allowed ids (PyTorch)."""
    full, scored = KVBatchGraphZhV2(model), KVBatchGraphZhV2(model, "scored")
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(8, VOCAB, (3, 1), generator=g, dtype=torch.int32)
    position = torch.tensor([0, 4, 9], dtype=torch.int32)
    k = torch.randn(LAYERS, 3, HEADS, 10, HEAD_DIM, generator=g)
    v = torch.randn(LAYERS, 3, HEADS, 10, HEAD_DIM, generator=g)
    allowed = torch.randint(0, VOCAB, (149,), generator=g, dtype=torch.int32)
    logits, _, _ = full(ids, position, k, v)
    lse, picked, _, _ = scored(ids, position, k, v, allowed)
    error = float((picked - lse[:, None] - logits.log_softmax(-1)[:, allowed.long()]).abs().max())
    if error > TOLERANCE:
        raise AssertionError(f"scored head differs by {error}")
    return {"allowed": len(allowed), "max_abs_log_prob": error}


def _as(x):
    return torch.as_tensor(np.asarray(x)).float()


def torch_sequence_step(graph):
    """fn(ids [B,T], targets [B,T], position [B], k, v) -> (target_logprobs, new_key, new_value)."""
    return lambda ids, targets, position, k, v: graph(ids, position, k, v, targets)


def _expected_logprobs(reference, ids):
    """log P(ids[j+1] | ids[:j+1]) for j < len-1 from a full forward (`reference(ids [1,n]) -> [1,n,V]`)."""
    full = _as(reference(ids[None].long()))[0].log_softmax(-1)
    return full[torch.arange(len(ids) - 1), ids[1:].long()]


@torch.inference_mode()
def check_sequence(reference, fn, batch=4, seed=1119, strict=True):
    """Sequence head, the way sentence scoring uses it. First call per row: prefill a prompt of its own
    length (B=1, position 0) with targets = the next tokens. Second call: all rows together, each from its
    own position, a continuation of its own length right-padded to the longest (token 0, target 0), with
    the cache tails filled with large finite junk. Every real position's log-prob must match the
    reference's full forward."""
    prompt = PREFILL_LENGTHS(batch)
    more = [1 + (3 * b) % 7 for b in range(batch)]
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(8, VOCAB, (batch, max(p + m for p, m in zip(prompt, more)) + 1), generator=g, dtype=torch.int32)
    worst, outside, positions = 0.0, 0, 0

    def judge(got, want):
        nonlocal worst, outside, positions
        worst = max(worst, float((got - want).abs().max()))
        positions += len(want)
        bad = int((~torch.isclose(got, want, atol=TOLERANCE, rtol=TOLERANCE)).sum())
        outside += bad
        if bad and strict:
            torch.testing.assert_close(got, want, atol=TOLERANCE, rtol=TOLERANCE)

    caches = []
    for b in range(batch):
        n = prompt[b]
        junk = torch.full((LAYERS, 1, HEADS, 1, HEAD_DIM), 30.0)
        lp, nk, nv = (_as(x) for x in fn(ids[b:b + 1, :n], ids[b:b + 1, 1:n + 1], torch.tensor([0], dtype=torch.int32),
                                         junk, -junk))
        judge(lp[0], _expected_logprobs(reference, ids[b, :n + 1]))
        caches.append((nk[:, 0], nv[:, 0]))
    width = max(prompt) + 3
    k = torch.full((LAYERS, batch, HEADS, width, HEAD_DIM), 30.0)
    v = torch.full_like(k, -30.0)
    for b, (ck, cv) in enumerate(caches):
        k[:, b, :, :prompt[b]] = ck
        v[:, b, :, :prompt[b]] = cv
    tokens = torch.zeros((batch, max(more)), dtype=torch.int32)
    targets = torch.zeros_like(tokens)
    for b in range(batch):
        tokens[b, :more[b]] = ids[b, prompt[b]:prompt[b] + more[b]]
        targets[b, :more[b]] = ids[b, prompt[b] + 1:prompt[b] + more[b] + 1]
    lp, _, _ = (_as(x) for x in fn(tokens, targets, torch.tensor(prompt, dtype=torch.int32), k, v))
    for b in range(batch):
        n = prompt[b] + more[b]
        judge(lp[b, :more[b]], _expected_logprobs(reference, ids[b, :n + 1])[prompt[b]:])
    return {"rows": batch, "prompt_lengths": prompt, "continuation_lengths": more, "positions": positions,
            "max_abs": worst, "positions_outside_tolerance": outside}


def score_continuations(fn, prefix, continuations, rows=SEQUENCE_ROWS, slots=None):
    """Reference algorithm for sentence scoring with the sequence head.

    prefix: the shared token ids (BOS + context + the part every candidate shares), at least one token.
    continuations: each candidate's remaining token ids (non-empty). Returns (sum of log P over each
    continuation's tokens given prefix, number of calls). One prefill call over prefix[:-1] (skipped when
    the prefix is one token), then one call per `rows` continuations: input = [prefix[-1]] + c[:-1],
    targets = c, position = len(prefix) - 1, every row reading the same cache. `slots` pads the cache
    width (for packages with enumerated or fixed S)."""
    prefix = [int(t) for t in prefix]
    if not prefix or any(not c for c in continuations):
        raise ValueError("Need a non-empty prefix and non-empty continuations.")
    if len(prefix) - 1 + max(len(c) for c in continuations) > CONTEXT:
        raise ValueError(f"prefix + continuation exceed {CONTEXT} tokens")
    calls, n = 0, len(prefix) - 1
    if n:
        zeros = np.zeros((LAYERS, 1, HEADS, 1, HEAD_DIM), np.float32)
        _, nk, nv = fn(np.array([prefix[:-1]], np.int32), np.array([prefix[1:]], np.int32), np.array([0], np.int32),
                       zeros, zeros)
        ck, cv = np.asarray(nk, np.float32)[:, 0], np.asarray(nv, np.float32)[:, 0]          # (9,6,n,64)
        calls += 1
    width = max(n, 1, slots or 0)
    scores = []
    for start in range(0, len(continuations), rows):
        part = continuations[start:start + rows]
        t = max(len(c) for c in part)
        ids = np.zeros((len(part), t), np.int32)
        targets = np.zeros_like(ids)
        for r, c in enumerate(part):
            ids[r, :len(c)] = [prefix[-1]] + list(c[:-1])
            targets[r, :len(c)] = c
        k = np.zeros((LAYERS, len(part), HEADS, width, HEAD_DIM), np.float32)
        v = np.zeros_like(k)
        if n:
            k[:, :, :, :n] = ck[:, None]
            v[:, :, :, :n] = cv[:, None]
        lp, _, _ = fn(ids, targets, np.full(len(part), n, np.int32), k, v)
        lp = np.asarray(lp, np.float64)
        calls += 1
        scores += [float(lp[r, :len(c)].sum()) for r, c in enumerate(part)]
    return scores, calls


@torch.inference_mode()
def check_continuations(reference, fn, seed=1120, count=37):
    """score_continuations vs the full forward: one shared prefix, `count` continuations of 1..12 tokens."""
    g = torch.Generator().manual_seed(seed)
    prefix = torch.randint(8, VOCAB, (9,), generator=g, dtype=torch.int32).tolist()
    conts = [torch.randint(8, VOCAB, (1 + i % 12,), generator=g, dtype=torch.int32).tolist() for i in range(count)]
    got, calls = score_continuations(lambda *a: tuple(np.asarray(x) for x in fn(*(torch.as_tensor(y) for y in a))),
                                     prefix, conts)
    want = [float(_expected_logprobs(reference, torch.tensor(prefix + c, dtype=torch.int32))[len(prefix) - 1:].double().sum())
            for c in conts]
    worst = max(abs(a - b) for a, b in zip(got, want))
    return {"continuations": count, "calls": calls, "max_abs_sum": worst,
            "passed": worst <= 1e-3, "tolerance_sum": 1e-3}


def run_checks(model, fn_factory):
    """All PyTorch checks for one step function factory (eager or traced)."""
    out = {"decode": [], "prefill": []}
    for case in v1.ALIGN_CASES:
        out["decode"].append(v1.check_graph(model, step_view(fn_factory()), **case))
    for case in ({"batch": 4}, {"batch": 1, "tokens": 1}, {"batch": 16, "tokens": 3}):
        out["prefill"].append(check_prefill(model, fn_factory(), **case))
    return out


def example_inputs(head, tokens=1):
    ex = (torch.tensor([[7] * tokens, [9] * tokens], dtype=torch.int32), torch.tensor([0, 2], dtype=torch.int32),
          torch.zeros(LAYERS, 2, HEADS, 3, HEAD_DIM), torch.zeros(LAYERS, 2, HEADS, 3, HEAD_DIM))
    if head == "scored":
        return ex + (torch.tensor([5, 6, 7], dtype=torch.int32),)
    if head == "sequence":
        return ex + (torch.tensor([[8] * tokens, [10] * tokens], dtype=torch.int32),)
    return ex


@torch.inference_mode()
def trace(model, head="logits", tokens=1):
    if head == "sequence" and tokens == 1:
        raise ValueError("The sequence head is built with a flexible T.")
    graph = KVBatchGraphZhV2(model, head).eval()
    traced = torch.jit.trace(graph, example_inputs(head, 1 if tokens == 1 else 3))
    if head == "logits":
        run_checks(model, lambda: traced)
    elif head == "sequence":
        for case in ({"batch": 4}, {"batch": 1}, {"batch": 16, "seed": 2026}):
            check_sequence(model, torch_sequence_step(traced), **case)
    else:
        lse, picked, _, _ = traced(*example_inputs(head))
        ref, _, _ = KVBatchGraphZhV2(model)(*example_inputs("logits"))
        torch.testing.assert_close(picked - lse[:, None], ref.log_softmax(-1)[:, [5, 6, 7]], atol=TOLERANCE, rtol=TOLERANCE)
    return traced


def shape_spec(ct, batch, tokens, slots):
    """batch/slots: "flex" | int | list of ints (enumerated); tokens: 1 | "flex" (1..128)."""
    def dim(value, lo, hi, default):
        if value == "flex":
            return ct.RangeDim(lower_bound=lo, upper_bound=hi, default=default)
        return value
    enumerated = isinstance(batch, list) or isinstance(slots, list)
    if enumerated:
        if tokens != 1:
            raise ValueError("Enumerated shapes are built for the decode step (T=1) only.")
        bs = batch if isinstance(batch, list) else [batch]
        ss = slots if isinstance(slots, list) else [slots]
        if "flex" in bs or "flex" in ss:
            raise ValueError("Do not mix enumerated and flexible dimensions.")
        pairs = [(b, s) for b in bs for s in ss]
        if len(pairs) < 2:
            raise ValueError("Enumerated shapes need at least two (batch, slots) pairs; use fixed shapes.")
        # Mac 2026-11-15: iOS 18 requires every enumerated input to list the same number of shapes (the
        # i-th shape of each input forms configuration i), so ids/position repeat their shape per pair.
        cache = ct.EnumeratedShapes(shapes=[(LAYERS, b, HEADS, s, HEAD_DIM) for b, s in pairs],
                                    default=(LAYERS, bs[-1], HEADS, ss[0], HEAD_DIM))
        if len(bs) == 1:        # fixed batch: ids/position are plain shapes (EnumeratedShapes needs >= 2)
            return (bs[0], 1), (bs[0],), cache, cache
        ids = ct.EnumeratedShapes(shapes=[(b, 1) for b, _ in pairs], default=(bs[-1], 1))
        pos = ct.EnumeratedShapes(shapes=[(b,) for b, _ in pairs], default=(bs[-1],))
        return ids, pos, cache, cache
    b = dim(batch, 1, MAX_BATCH, MAX_BATCH)
    t = dim(tokens, 1, CONTEXT, 1)
    s = dim(slots, 1, CONTEXT, 32)
    return (b, t), (b,), (LAYERS, b, HEADS, s, HEAD_DIM), (LAYERS, b, HEADS, s, HEAD_DIM)


def describe(value, lo, hi):
    if value == "flex":
        return f"flexible {lo}..{hi} (RangeDim)", [lo, hi]
    if isinstance(value, list):
        return f"enumerated {value}", [min(value), max(value)]
    return f"fixed {value}", [value, value]


def convert(bundle, output, batch="flex", tokens=1, slots="flex", head="logits"):
    ct = coremltools()
    if Path(output).exists():
        raise ValueError("Conversion output exists.")
    if head not in HEADS_V2:
        raise ValueError(head)
    if head == "sequence" and (tokens == 1 or isinstance(batch, list) or isinstance(slots, list)):
        raise ValueError("The sequence head is built with a flexible T and no enumerated shapes.")
    lm = zh_v1.BundleLMZh(bundle)
    ids, pos, kc, vc = shape_spec(ct, batch, tokens, slots)
    inputs = [ct.TensorType(name="input_ids", shape=ids, dtype=np.int32),
              ct.TensorType(name="position", shape=pos, dtype=np.int32),
              ct.TensorType(name="key_cache", shape=kc, dtype=np.float32),
              ct.TensorType(name="value_cache", shape=vc, dtype=np.float32)]
    if head == "scored":
        inputs.append(ct.TensorType(name="allowed", shape=(ct.RangeDim(1, MAX_ALLOWED, 160),), dtype=np.int32))
        outputs = ["logsumexp", "allowed_logits", "new_key", "new_value"]
    elif head == "sequence":
        inputs.append(ct.TensorType(name="targets", shape=ids, dtype=np.int32))
        outputs = ["target_logprobs", "new_key", "new_value"]
    else:
        outputs = ["logits", "new_key", "new_value"]
    model = ct.convert(trace(lm.model, head, tokens), source="pytorch", convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT32, inputs=inputs,
        outputs=[ct.TensorType(name=n, dtype=np.float32) for n in outputs], skip_model_load=True)
    model.short_description = "vimeml-tiny-zh-v1 batched KV step v2 (read-only caches); FP32 compute."
    model.user_defined_metadata["bundle_manifest_sha256"] = lm.metadata["bundle_manifest_sha256"]
    batch_text, batch_range = describe(batch, 1, MAX_BATCH)
    slot_text, _ = describe(slots, 1, CONTEXT)
    head_io = ({"allowed": f"int32 [K], 1<=K<={MAX_ALLOWED}; one id list shared by every row",
                "logsumexp": "float32 [B]; over all 12,288 logits of the last new token",
                "allowed_logits": "float32 [B,K]; log P = allowed_logits - logsumexp"} if head == "scored" else
               {"targets": "int32 [B,T]; the token that follows input_ids[b,t] (right padding: any id)",
                "target_logprobs": "float32 [B,T]; log P(targets[b,t] | row b through input_ids[b,t]), normalized "
                                   "over all 12,288 logits; ignore padded positions"} if head == "sequence" else
               {"logits": f"float32 [B,{VOCAB}] for the last new token; no softmax"})
    zh_v1.finish(output, model, FORMAT, {"kind": "fp32", "minimum_ios": 18, "bundle": lm.metadata,
        "batch": batch_text, "batch_range": batch_range, "tokens": "1 (decode step)" if tokens == 1 else
        f"flexible 1..{CONTEXT} (RangeDim)", "head": head,
        "cache_shape": [LAYERS, "B", HEADS, "S", HEAD_DIM], "cache_slots": f"S {slot_text}",
        "cache_dtype": "float32", "compute_units": "CPU_ONLY",
        "interface": {"input_ids": "int32 [B,T]", "position": "int32 [B]; row cache length; position[b]+T <= 128",
                      "key_cache/value_cache": "float32 (9,B,6,S,64); S >= max(position), S >= 1; "
                                               "row b slots >= position[b] are masked but must be finite (zero new buffers): NaN/Inf there reaches the outputs", **head_io,
                      "new_key/new_value": "float32 (9,B,6,T,64); the caller writes them into slots "
                                           "position[b].. of row b"},
        "rows": "independent; padding rows (token 0, position 0) do not affect others",
        **({"scoring": "one call per batch of candidates: input = [last shared token] + candidate[:-1], targets = "
                       "candidate, position = shared length - 1, every row reading the shared cache; sum "
                       "target_logprobs over the candidate's tokens (zh_v1_batch2.score_continuations)"}
           if head == "sequence" else {}),
        "prompt": "B=1, T=prompt length, position 0, S=1 (any finite values, e.g. zeros)" if tokens != 1 else
                  "B=1 one token per call, or the batch-1 chunk package",
        "graph": f"vimeml.deployment.zh_v1_batch2.KVBatchGraphZhV2 (read-only caches, {head} head)",
        "precision_policy": "FP32 parameters and compute.", "policy": "Caller owns all cache buffers; inputs are never modified.",
        "conversion_source_sha256": file_sha(Path(__file__))})


def compress(source, output, alignment):
    """INT8 block32 (deployed recipe) of an aligned FP32 v2 package."""
    ct = coremltools()
    source, output, alignment = map(Path, (source, output, alignment))
    if output.exists():
        raise ValueError("Compression output exists.")
    original = zh_v1.verify_package(source, FORMAT)
    gate = read_json(alignment)
    if (original["kind"] != "fp32" or gate.get("format") != ALIGNMENT_FORMAT or not gate.get("passed")
            or gate.get("coreml_manifest_sha256") != file_sha(source / "manifest.json")):
        raise ValueError("First pass alignment for this exact FP32 v2 package.")
    from coremltools.optimize import coreml as opt
    model = ct.models.MLModel(str(source / "model.mlpackage"), skip_model_load=True)
    spec = zh_v1.RECIPES[zh_v1.DEPLOYED_RECIPE]
    config = opt.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_block",
                                         block_size=spec["block_size"], weight_threshold=zh_v1.WEIGHT_THRESHOLD)
    optimization, selection = v2.parameter_selection(opt, model, config)
    print(f"selected {len(selection['selected_constants'])} parameter matrices; "
          f"excluded {len(selection['excluded_constants'])} structural constants.", flush=True)
    result = opt.linear_quantize_weights(model, config=optimization)
    if not any(name.startswith("constexpr_") for name in inspect_spec(result)["operations"]):
        raise ValueError("No compressed weights found.")
    carried = {k: original[k] for k in ("minimum_ios", "bundle", "batch", "batch_range", "tokens", "head", "cache_shape",
                                        "cache_slots", "cache_dtype", "compute_units", "interface", "rows", "prompt",
                                        "graph", "policy")}
    carried.update({k: original[k] for k in ("scoring",) if k in original})
    zh_v1.finish(output, result, FORMAT, {**carried, "kind": "int8-b32_fp32_compute", "recipe": zh_v1.DEPLOYED_RECIPE,
        "source_manifest_sha256": file_sha(source / "manifest.json"), "fp32_alignment_sha256": file_sha(alignment),
        "compression": {**spec, "weight_threshold": zh_v1.WEIGHT_THRESHOLD, "selection": selection},
        "precision_policy": "INT8 block32 parameter matrices, FP32 compute; norms unchanged; no activation quantization.",
        "conversion_source_sha256": file_sha(Path(__file__))})


def shapes_of(manifest):
    """(batch sizes, slot counts) a package accepts, for alignment and measurement."""
    lo, hi = manifest["batch_range"]
    batch = manifest["batch"]
    if batch.startswith("enumerated"):
        bs = json.loads(batch.split(" ", 1)[1])
    else:
        bs = None if lo != hi else [hi]
    slots = manifest["cache_slots"]
    if "enumerated" in slots:
        ss = json.loads(slots.split("enumerated ", 1)[1])
    elif "fixed" in slots:      # Mac 2026-11-15: fixed-S fallback packages ("S fixed 32") accept only that S
        ss = [int(slots.rsplit(" ", 1)[1])]
    else:
        ss = None
    return bs, ss


def coreml_step(model, manifest, allowed=None):
    """v1.check_graph-style step over a v2 package. Rows are padded up to the nearest accepted batch,
    slots up to the nearest accepted slot count. With the scored head, returns log-probs on `allowed`
    in place of logits (callers compare against log_softmax)."""
    bs, ss = shapes_of(manifest)

    def fit(n, options):
        if options is None:
            return n
        bigger = [o for o in options if o >= n]
        if not bigger:
            raise ValueError(f"{n} exceeds the enumerated shapes {options}")
        return min(bigger)

    def fn(tokens, position, k, v):
        tokens, position = torch.as_tensor(tokens), torch.as_tensor(position)
        rows, t = tokens.shape
        size, width = fit(rows, bs), fit(max(k.shape[3], 1), ss)
        ids = np.zeros((size, t), np.int32)
        pos = np.zeros(size, np.int32)
        kk = np.zeros((LAYERS, size, HEADS, width, HEAD_DIM), np.float32)
        vv = np.zeros_like(kk)
        ids[:rows] = tokens.numpy()
        pos[:rows] = position.numpy()
        kk[:, :rows, :, :k.shape[3]] = np.asarray(k)
        vv[:, :rows, :, :k.shape[3]] = np.asarray(v)
        feed = {"input_ids": ids, "position": pos, "key_cache": kk, "value_cache": vv}
        if manifest["head"] == "scored":
            feed["allowed"] = np.asarray(allowed, np.int32)
            out = model.predict(feed)
            first = np.asarray(out["allowed_logits"][:rows], np.float32) - np.asarray(out["logsumexp"][:rows], np.float32)[:, None]
        else:
            out = model.predict(feed)
            first = np.asarray(out["logits"][:rows], np.float32)
        return first, np.asarray(out["new_key"][:, :rows], np.float32), np.asarray(out["new_value"][:, :rows], np.float32)
    return fn


def coreml_sequence_step(model):
    """fn(ids, targets, position, k, v) over a sequence-head package (flexible B, T and S)."""
    def fn(ids, targets, position, k, v):
        out = model.predict({"input_ids": np.asarray(ids, np.int32), "targets": np.asarray(targets, np.int32),
                             "position": np.asarray(position, np.int32), "key_cache": np.asarray(k, np.float32),
                             "value_cache": np.asarray(v, np.float32)})
        return (np.asarray(out["target_logprobs"], np.float32), np.asarray(out["new_key"], np.float32),
                np.asarray(out["new_value"], np.float32))
    return fn


SEQUENCE_CASES = ({"batch": 4}, {"batch": 1}, {"batch": 16, "seed": 2026})


class LogSoftmaxOn:
    """Reference wrapper: full forward -> log-softmax restricted to `allowed` (for the scored head)."""

    def __init__(self, reference, allowed):
        self.reference, self.allowed = reference, torch.as_tensor(np.asarray(allowed)).long()

    def __call__(self, ids):
        return torch.as_tensor(np.asarray(self.reference(ids))).float().log_softmax(-1)[..., self.allowed]


def align(package, bundle, output, stateless=None):
    """Gate a v2 package (decode cases always; prefill cases when T is flexible). Reference: PyTorch FP32
    from the bundle, or (`stateless`) the stateless Core ML package of the same precision."""
    ct = coremltools()
    package, output = Path(package), Path(output)
    if output.exists():
        raise ValueError("Alignment output exists.")
    manifest = zh_v1.verify_package(package, FORMAT)
    lm = zh_v1.BundleLMZh(bundle)
    if manifest["bundle"]["bundle_manifest_sha256"] != lm.metadata["bundle_manifest_sha256"]:
        raise ValueError("Package belongs to another bundle.")
    unit = ct.ComputeUnit.CPU_ONLY
    model = ct.models.MLModel(str(package / "model.mlpackage"), compute_units=unit)
    if stateless:
        ref_manifest = zh_v1.verify_package(stateless)
        reference = v1.coreml_full(ct.models.MLModel(str(Path(stateless) / "model.mlpackage"), compute_units=unit))
        ref_name = f"stateless Core ML {ref_manifest['kind']} ({Path(stateless).name})"
    else:
        reference = lambda ids: lm.model(ids).detach()
        ref_name = "PyTorch FP32 full forward (bundle)"
    allowed = None
    if manifest["head"] == "scored":
        allowed = np.random.default_rng(1118).choice(np.arange(8, VOCAB), 149, replace=False).astype(np.int32)
        reference = LogSoftmaxOn(reference, allowed)
    bs, ss = shapes_of(manifest)
    hi = manifest["batch_range"][1]
    checks = []
    if manifest["head"] == "sequence":
        fn = coreml_sequence_step(model)
        for case in SEQUENCE_CASES:
            stats = check_sequence(reference, fn, strict=False, **case)
            stats["kind"] = "sequence"
            stats["rows_outside_tolerance"] = stats["positions_outside_tolerance"]
            checks.append(stats)
            print(json.dumps(stats), flush=True)
        stats = check_continuations(reference, fn)
        stats.update(kind="continuations", max_abs=stats["max_abs_sum"], rows_outside_tolerance=0 if stats["passed"] else 1)
        checks.append(stats)
        print(json.dumps(stats), flush=True)
    for case in (() if manifest["head"] == "sequence" else v1.ALIGN_CASES):
        case = dict(case)
        if case["batch"] > hi or (ss and case.get("slots", 0) > max(ss)):
            continue
        stats = v1.check_graph(reference, step_view(coreml_step(model, manifest, allowed)), strict=False, **case)
        stats["kind"] = "decode"
        checks.append(stats)
        print(json.dumps(stats), flush=True)
    if manifest["tokens"].startswith("flexible") and manifest["head"] == "logits":
        for case in ({"batch": 4}, {"batch": 16, "tokens": 3}):
            if case["batch"] > hi:
                continue
            stats = check_prefill(reference, coreml_step(model, manifest), strict=False, **case)
            stats["kind"] = "prefill"
            checks.append(stats)
            print(json.dumps(stats), flush=True)
    passed = bool(checks) and all(c["rows_outside_tolerance"] == 0 for c in checks)
    write_json(output, {"format": ALIGNMENT_FORMAT, "passed": passed, "reference": ref_name,
        "coreml_manifest_sha256": file_sha(package / "manifest.json"), "package": package.name, "kind": manifest["kind"],
        "head": manifest["head"], "tolerance": {"atol": TOLERANCE, "rtol": TOLERANCE}, "checks": checks,
        "method": "decode: rows of 3..25 tokens starting at different steps, each step vs the reference's full forward; "
                  "prefill (flexible T): per-row prompt in one call, then a multi-token call for all rows at different "
                  "positions; scored head: log-probs on 149 fixed allowed ids vs the reference's log-softmax; sequence head: "
                  "per-position log-probs after a per-row prompt and after a padded multi-row continuation call, and "
                  "score_continuations sums (37 candidates, tolerance 1e-3 per sum)",
        "conversion_source_sha256": file_sha(Path(__file__))})
    print(f"alignment {'passed' if passed else 'FAILED'}: max {max(c['max_abs'] for c in checks):.3g}")
    return passed


def _pack(states, size, width, dtype=np.float32):
    k = np.zeros((LAYERS, size, HEADS, max(width, 1), HEAD_DIM), dtype)
    v = np.zeros_like(k)
    position = np.zeros(size, np.int32)
    for r, (sk, sv) in enumerate(states):
        n = sk.shape[2]
        k[:, r, :, :n] = np.asarray(sk)
        v[:, r, :, :n] = np.asarray(sv)
        position[r] = n
    return k, v, position


class TorchBatchBackendV2:
    """PyTorch reference of the v2 calls; scored=True uses the scored head (zh_search passes allowed ids)."""

    def __init__(self, model, scored=False, device="cpu"):
        self.model = model.to(device).float().eval()
        self.device = device
        self.scored = scored
        self.graph = KVBatchGraphZhV2(self.model, "scored" if scored else "logits").to(device).eval()
        self.calls = self.rows = 0

    @torch.inference_mode()
    def _run(self, ids, position, k, v, allowed):
        t = lambda x, d=torch.float32: torch.as_tensor(np.asarray(x), dtype=d, device=self.device)
        args = (t(ids, torch.int32), t(position, torch.int32), t(k), t(v))
        if self.scored:
            lse, picked, nk, nv = self.graph(*args, t(allowed, torch.int32))
            return (picked - lse[:, None]).float().cpu(), nk.cpu().numpy(), nv.cpu().numpy()
        logits, nk, nv = self.graph(*args)
        return logits.float().cpu(), nk.cpu().numpy(), nv.cpu().numpy()

    def prefill(self, ids, allowed=None):
        k = np.zeros((LAYERS, 1, HEADS, 1, HEAD_DIM), np.float32)
        out, nk, nv = self._run([ids], [0], k, k, allowed)
        return out[0], (nk[:, 0], nv[:, 0])

    def decode(self, tokens, states, allowed=None):
        k, v, position = _pack(states, len(states), max(s[0].shape[2] for s in states))
        out, nk, nv = self._run(np.array(tokens)[:, None], position, k, v, allowed)
        self.calls += 1
        self.rows += len(states)
        new = [(np.concatenate([sk, nk[:, r]], 2), np.concatenate([sv, nv[:, r]], 2)) for r, (sk, sv) in enumerate(states)]
        return out, new


class CoreMLBatchBackendV2:
    """The search's two calls on a v2 package. Prompt: the package itself when T is flexible, else the
    batch-1 chunk package when given, else B=1 one token per call through the package."""

    def __init__(self, package, chunk=None, compute_units="CPU_ONLY"):
        ct = coremltools()
        unit = getattr(ct.ComputeUnit, compute_units)
        self.manifest = zh_v1.verify_package(package, FORMAT)
        self.model = ct.models.MLModel(str(Path(package) / "model.mlpackage"), compute_units=unit)
        self.scored = self.manifest["head"] == "scored"
        self.flexible_tokens = self.manifest["tokens"].startswith("flexible")
        self.chunk = None
        if chunk and not self.flexible_tokens:
            zh_v1.verify_package(chunk, zh_v1.CHUNK_FORMAT)
            self.chunk = ct.models.MLModel(str(Path(chunk) / "model.mlpackage"), compute_units=unit)
        self.bs, self.ss = shapes_of(self.manifest)
        self.calls = self.rows = self.prefill_calls = 0

    def _fit(self, n, options):
        if options is None:
            return n
        bigger = [o for o in options if o >= n]
        if not bigger:
            raise ValueError(f"{n} rows/slots exceed this package's shapes {options}")
        return min(bigger)

    def _call(self, ids, position, k, v, allowed, rows):
        feed = {"input_ids": ids, "position": position, "key_cache": k, "value_cache": v}
        if self.scored:
            feed["allowed"] = np.asarray(allowed, np.int32)
        out = self.model.predict(feed)
        if self.scored:
            first = np.asarray(out["allowed_logits"][:rows], np.float32) - np.asarray(out["logsumexp"][:rows], np.float32)[:, None]
        else:
            first = np.asarray(out["logits"][:rows], np.float32)
        return torch.from_numpy(first), np.asarray(out["new_key"][:, :rows], np.float32), np.asarray(out["new_value"][:, :rows], np.float32)

    def prefill(self, ids, allowed=None):
        if self.chunk is not None:
            if self.scored:
                raise ValueError("Use a flexible-T package or token-by-token prefill with the scored head.")
            zeros = np.zeros((LAYERS, 1, HEADS, CONTEXT, HEAD_DIM), np.float32)
            out = self.chunk.predict({"input_ids": np.array([ids], np.int32), "position": np.array([0], np.int32),
                                      "key_cache": zeros, "value_cache": zeros})
            n = len(ids)
            self.prefill_calls += 1
            return (torch.from_numpy(np.array(out["logits"][0, -1], np.float32)),
                    (np.array(out["new_key_cache"][:, 0, :, :n], np.float32), np.array(out["new_value_cache"][:, 0, :, :n], np.float32)))
        size = self._fit(1, self.bs)
        if self.flexible_tokens:
            k, v, position = _pack([], size, self._fit(1, self.ss))
            tokens = np.zeros((size, len(ids)), np.int32)
            tokens[0] = ids
            self.prefill_calls += 1
            out, nk, nv = self._call(tokens, position, k, v, allowed, 1)
            return out[0], (nk[:, 0], nv[:, 0])
        state = (np.zeros((LAYERS, HEADS, 0, HEAD_DIM), np.float32),) * 2
        for i, tid in enumerate(ids):                 # one token per call; only the last call's scores are used
            k, v, position = _pack([state], size, self._fit(max(i, 1), self.ss))
            tokens = np.zeros((size, 1), np.int32)
            tokens[0, 0] = tid
            self.prefill_calls += 1
            last = i == len(ids) - 1
            out, nk, nv = self._call(tokens, position, k, v, allowed if (last or not self.scored) else [0], 1)
            state = (np.concatenate([state[0], nk[:, 0]], 2), np.concatenate([state[1], nv[:, 0]], 2))
        return out[0], state

    def decode(self, tokens, states, allowed=None):
        rows = len(states)
        size = self._fit(rows, self.bs)
        width = self._fit(max(max(s[0].shape[2] for s in states), 1), self.ss)
        k, v, position = _pack(states, size, width)
        ids = np.zeros((size, 1), np.int32)
        ids[:rows, 0] = tokens
        out, nk, nv = self._call(ids, position, k, v, allowed, rows)
        self.calls += 1
        self.rows += rows
        new = [(np.concatenate([sk, nk[:, r]], 2), np.concatenate([sv, nv[:, r]], 2)) for r, (sk, sv) in enumerate(states)]
        return out, new
