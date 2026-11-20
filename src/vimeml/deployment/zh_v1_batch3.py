"""vimeml-tiny-zh-v1 batched KV step, version 3: shared prefix, lean sentence scoring, one package.

Changes against `zh_v1_batch2` (v2), after Vime's on-device memory work (2026-11-17/18):

1. **Shared prefix.** All rows of a call read one prefix cache `prefix_key/prefix_value` (9,1,6,P,64), of
   which `prefix_length` slots are valid. Rows keep only their own tokens after the prefix in
   `key_cache/value_cache` (9,B,6,S,64). Attention reads the prefix once for all rows (a broadcast
   matmul) and combines prefix, row cache and new tokens under one softmax normalizer. v2 needed the
   prompt copied into every row: 16 rows of a 64-token prompt are 28 MiB of FP32 cache input per call;
   here the prompt is 1.7 MiB, once.
2. **`score` function** (sentence scoring): no row cache and no new keys/values; rows start right after
   the prefix and only `target_logprobs` [B,T] comes back. With `vocab_chunks` > 1 the logsumexp over the
   12,288 logits runs in chunks with a running max and sum, and target logits come from the tied head
   rows, so the full [B,T,12288] logits need not exist at once. Whether Core ML frees each chunk before
   the next, and what the sliced head constants cost, is measured on the Mac (`vocab_chunks` 1 and 4).
3. **One multifunction package**: `logits` (full logits of the last new token; next-word prediction),
   `scored` (logsumexp + allowed logits; beam search and prompt prefill) and `score`. Identical weight
   constants are stored once.

Interface (Core ML names). B <= 16; P, S and T are flexible 1..128; positions are absolute and must stay
below 128:
  input_ids     int32 [B,T]
  position      int32 [B]            logits/scored: row b's cached own tokens; token t of row b sits at
                                     prefix_length + position[b] + t
  key_cache     float32 (9,B,6,S,64) logits/scored: row b's own slots; slots >= position[b] are masked
  value_cache   float32 (9,B,6,S,64)   but must be finite (zero new buffers)
  prefix_length int32 [1]            valid prefix slots; 0 = no prefix (pass P=1 of zeros)
  prefix_key    float32 (9,1,6,P,64) shared by every row; slots >= prefix_length masked, finite
  prefix_value  float32 (9,1,6,P,64)
  allowed       int32 [K]            scored only; one id list shared by every row
  targets       int32 [B,T]          score only; the token that follows input_ids[b,t]
  -> logits float32 [B,12288]                              logits: last new token, no softmax
  -> logsumexp float32 [B], allowed_logits float32 [B,K]   scored: log P = allowed_logits - logsumexp
  -> new_key, new_value float32 (9,B,6,T,64)               logits/scored: keys/values of the new tokens
  -> target_logprobs float32 [B,T]                         score: token t of row b at prefix_length + t

Typical calls:
  prompt      scored (or logits), B=1, T=prompt length, position 0, S=1 zeros, prefix_length 0, P=1 zeros;
              its new_key/new_value become prefix_key/prefix_value with prefix_length = prompt length
  beam step   scored, B rows, T=1, position = each row's own token count, prefix = the prompt
  sentences   score, B candidates, input = [last prefix token] + c[:-1], targets = c, prefix = BOS +
              context + the shared part minus its last token (see score_continuations)
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vimeml.deployment import v2, zh_v1
from vimeml.deployment import zh_v1_batch as v1
from vimeml.deployment.bundle import read_json, write_json
from vimeml.deployment.coreml import coremltools, inspect_spec
from vimeml.training.data import file_sha

FORMAT = "vimeml_coreml_zh_v1_kv_batch_v3"
MERGED_FORMAT = "vimeml_coreml_zh_v1_kv_batch_v3_merged"
ALIGNMENT_FORMAT = "vimeml_kv_batch_alignment_zh_v1_v3"
FUNCTIONS = ("logits", "scored", "score")
VOCAB, LAYERS, CONTEXT, HEADS, HEAD_DIM, MAX_BATCH = v1.VOCAB, v1.LAYERS, v1.CONTEXT, v1.HEADS, v1.HEAD_DIM, v1.MAX_BATCH
WIDTH = HEADS * HEAD_DIM
TOLERANCE, MASKED = v1.TOLERANCE, v1.MASKED
MAX_ALLOWED = 1024
SCORE_ROWS = 16
SUM_TOLERANCE = 1e-3
STEP_INPUTS = ("input_ids", "position", "key_cache", "value_cache", "prefix_length", "prefix_key", "prefix_value")
SCORE_INPUTS = ("input_ids", "targets", "prefix_length", "prefix_key", "prefix_value")


def _layers(model, input_ids, start, prefix, private):
    """start: [B] or [1] int32, absolute position of the first new token. prefix: (length [1], key, value)
    with key/value (9,1,6,P,64). private: None or (position [B], key, value) with key/value (9,B,6,S,64).
    Returns the last hidden states [B,T,384] and the new tokens' keys and values (9,B,6,T,64)."""
    batch, tokens = input_ids.shape
    steps = torch.arange(tokens, dtype=torch.int32, device=input_ids.device)
    hidden = model.token_embedding(input_ids) + model.position_embedding(start[:, None] + steps[None, :])
    length, prefix_key, prefix_value = prefix
    p_index = torch.arange(prefix_key.shape[3], dtype=torch.int32, device=input_ids.device)
    prefix_mask = (p_index >= length).to(hidden.dtype)[None, None, None, :] * MASKED             # [1,1,1,P]
    causal = (steps[None, :] > steps[:, None]).to(hidden.dtype)[None, None] * MASKED             # [1,1,T,T]
    if private is not None:
        position, key_cache, value_cache = private
        s_index = torch.arange(key_cache.shape[3], dtype=torch.int32, device=input_ids.device)
        cache_mask = (s_index[None, :] >= position[:, None]).to(hidden.dtype)[:, None, None, :] * MASKED  # [B,1,1,S]
    scale = 1.0 / math.sqrt(HEAD_DIM)
    keys, values = [], []
    for layer, block in enumerate(model.blocks):
        qkv = block.attention.qkv(block.attention_norm(hidden)).reshape(batch, tokens, 3, HEADS, HEAD_DIM)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)                                # [B,6,T,64]
        parts = [(torch.matmul(query, prefix_key[layer].transpose(-2, -1)) * scale + prefix_mask, prefix_value[layer]),
                 (torch.matmul(query, key.transpose(-2, -1)) * scale + causal, value)]
        if private is not None:
            parts.append((torch.matmul(query, key_cache[layer].transpose(-2, -1)) * scale + cache_mask, value_cache[layer]))
        top = parts[0][0].amax(-1, keepdim=True)
        for scores, _ in parts[1:]:
            top = torch.maximum(top, scores.amax(-1, keepdim=True))
        total, attended = None, None
        for scores, vals in parts:
            weights = torch.exp(scores - top)
            part_sum, part_out = weights.sum(-1, keepdim=True), torch.matmul(weights, vals)
            total = part_sum if total is None else total + part_sum
            attended = part_out if attended is None else attended + part_out
        attended = attended / total
        hidden = hidden + block.attention.projection(attended.transpose(1, 2).reshape(batch, tokens, WIDTH))
        hidden = hidden + block.ffn(block.ffn_norm(hidden))
        keys.append(key)
        values.append(value)
    return hidden, torch.stack(keys), torch.stack(values)


class StepGraphV3(nn.Module):
    """`logits` or `scored` function (see the module docstring)."""

    def __init__(self, model, head="scored"):
        super().__init__()
        if head not in ("logits", "scored"):
            raise ValueError(head)
        self.model, self.head = model.eval(), head

    def forward(self, input_ids, position, key_cache, value_cache, prefix_length, prefix_key, prefix_value, allowed=None):
        hidden, keys, values = _layers(self.model, input_ids, prefix_length + position,
                                       (prefix_length, prefix_key, prefix_value), (position, key_cache, value_cache))
        logits = self.model.lm_head(self.model.final_norm(hidden[:, -1]))                       # [B,V]
        if self.head == "scored":
            return torch.logsumexp(logits, -1), torch.index_select(logits, 1, allowed), keys, values
        return logits, keys, values


class ScoreGraphV3(nn.Module):
    """`score` function: target log-probs after a shared prefix; no row cache, no new keys/values."""

    def __init__(self, model, vocab_chunks=1):
        super().__init__()
        if vocab_chunks < 1 or VOCAB % vocab_chunks:
            raise ValueError(f"vocab_chunks must divide {VOCAB}")
        self.model, self.chunks = model.eval(), vocab_chunks

    def forward(self, input_ids, targets, prefix_length, prefix_key, prefix_value):
        hidden, _, _ = _layers(self.model, input_ids, prefix_length, (prefix_length, prefix_key, prefix_value), None)
        h = self.model.final_norm(hidden)                                                          # [B,T,384]
        if self.chunks == 1:
            logits = self.model.lm_head(h)                                                         # [B,T,V]
            picked = torch.gather(logits, 2, targets.long().unsqueeze(-1)).squeeze(-1)
            return picked - torch.logsumexp(logits, -1)
        weight = self.model.lm_head.weight                                                         # tied, no bias
        picked = (h * F.embedding(targets, weight)).sum(-1)                                        # [B,T]
        size = VOCAB // self.chunks
        top = total = None
        for c in range(self.chunks):
            part = torch.matmul(h, weight[c * size:(c + 1) * size].t())                           # [B,T,V/c]
            m = part.amax(-1)
            s = torch.exp(part - m[..., None]).sum(-1)
            if top is None:
                top, total = m, s
            else:
                new = torch.maximum(top, m)
                total = total * torch.exp(top - new) + s * torch.exp(m - new)
                top = new
        return picked - (top + torch.log(total))


def graph(model, function, vocab_chunks=1):
    return ScoreGraphV3(model, vocab_chunks) if function == "score" else StepGraphV3(model, function)


# ---- feeds and runners ---------------------------------------------------------------------------------

def empty_cache(batch=1, fill=0.0):
    return np.full((LAYERS, batch, HEADS, 1, HEAD_DIM), fill, np.float32)


def no_prefix(fill=0.0):
    return {"prefix_length": np.zeros(1, np.int32), "prefix_key": empty_cache(1, fill), "prefix_value": empty_cache(1, -fill)}


def as_prefix(key, value, length=None, slots=None, fill=0.0):
    """Prefix inputs from a prefill call's new_key/new_value (9,1,6,n,64), padded to `slots` with `fill`."""
    n = key.shape[3]
    width = max(slots or n, n, 1)
    k = np.full((LAYERS, 1, HEADS, width, HEAD_DIM), fill, np.float32)
    v = np.full_like(k, -fill)
    k[:, :, :, :n], v[:, :, :, :n] = key, value
    return {"prefix_length": np.array([n if length is None else length], np.int32), "prefix_key": k, "prefix_value": v}


def pack(states, width=None, fill=0.0):
    """Row caches from per-row (key, value) of shape (9,6,r,64): arrays (9,B,6,W,64) and position [B]."""
    width = max(width or 0, max((s[0].shape[2] for s in states), default=0), 1)
    k = np.full((LAYERS, len(states), HEADS, width, HEAD_DIM), fill, np.float32)
    v = np.full_like(k, -fill)
    position = np.zeros(len(states), np.int32)
    for r, (sk, sv) in enumerate(states):
        n = sk.shape[2]
        k[:, r, :, :n], v[:, r, :, :n] = sk, sv
        position[r] = n
    return k, v, position


class TorchRunner:
    """runner(function, feed) -> outputs, with Core ML names, over PyTorch graphs (eager or traced)."""

    def __init__(self, model=None, vocab_chunks=1, graphs=None):
        self.graphs = graphs or {f: graph(model, f, vocab_chunks) for f in FUNCTIONS}

    @torch.inference_mode()
    def __call__(self, function, feed):
        t = lambda name: torch.as_tensor(np.ascontiguousarray(feed[name]))
        g = self.graphs[function]
        if function == "score":
            return {"target_logprobs": g(*(t(n) for n in SCORE_INPUTS)).float().numpy()}
        args = [t(n) for n in STEP_INPUTS]
        if function == "scored":
            lse, picked, k, v = g(*args, t("allowed"))
            return {"logsumexp": lse.numpy(), "allowed_logits": picked.numpy(), "new_key": k.numpy(), "new_value": v.numpy()}
        logits, k, v = g(*args)
        return {"logits": logits.numpy(), "new_key": k.numpy(), "new_value": v.numpy()}


class CoreMLRunner:
    """The same calls over a merged v3 package (one MLModel per function) or single-function packages."""

    def __init__(self, path, compute_units="CPU_ONLY", functions=FUNCTIONS):
        ct = coremltools()
        unit = getattr(ct.ComputeUnit, compute_units)
        path = Path(path)
        self.manifest = read_json(path / "manifest.json")
        if self.manifest.get("format") == MERGED_FORMAT:
            verify_merged(path)
            self.models = {f: ct.models.MLModel(str(path / "model.mlpackage"), compute_units=unit, function_name=f)
                           for f in functions if f in self.manifest["functions"]}
        else:
            manifest = zh_v1.verify_package(path, FORMAT)
            self.models = {manifest["function"]: ct.models.MLModel(str(path / "model.mlpackage"), compute_units=unit)}

    def __call__(self, function, feed):
        out = self.models[function].predict({k: np.ascontiguousarray(v) for k, v in feed.items()})
        return {k: np.asarray(v, np.float32) for k, v in out.items()}


def first_scores(function, out):
    """Per-row scores of the last new token: logits, or log-probs on `allowed` for the scored function."""
    if function == "scored":
        return out["allowed_logits"] - out["logsumexp"][:, None]
    return out["logits"]


def prompt_call(runner, function, ids, allowed=None, fill=0.0):
    """B=1 prompt prefill without a prefix. Returns (outputs, prefix inputs for the following calls)."""
    feed = {"input_ids": np.array([ids], np.int32), "position": np.zeros(1, np.int32),
            "key_cache": empty_cache(1, fill), "value_cache": empty_cache(1, -fill), **no_prefix(fill)}
    if function == "scored":
        feed["allowed"] = np.asarray([0] if allowed is None else allowed, np.int32)
    out = runner(function, feed)
    return out, as_prefix(out["new_key"], out["new_value"])


def score_continuations(runner, prefix, continuations, rows=SCORE_ROWS, prefill="scored"):
    """Reference algorithm for sentence scoring. prefix: BOS + context + the part every candidate shares
    (at least one token). continuations: each candidate's remaining ids (non-empty). One prompt call over
    prefix[:-1] (skipped for a one-token prefix), then one `score` call per `rows` candidates: input =
    [prefix[-1]] + c[:-1], targets = c, right-padded with 0. Returns (sum of log P per candidate, calls)."""
    prefix = [int(t) for t in prefix]
    if not prefix or any(not c for c in continuations):
        raise ValueError("Need a non-empty prefix and non-empty continuations.")
    if len(prefix) - 1 + max(len(c) for c in continuations) > CONTEXT:
        raise ValueError(f"prefix + continuation exceed {CONTEXT} tokens")
    calls, shared = 0, no_prefix()
    if len(prefix) > 1:
        _, shared = prompt_call(runner, prefill, prefix[:-1])
        calls += 1
    scores = []
    for start in range(0, len(continuations), rows):
        part = continuations[start:start + rows]
        t = max(len(c) for c in part)
        ids = np.zeros((len(part), t), np.int32)
        targets = np.zeros_like(ids)
        for r, c in enumerate(part):
            ids[r, :len(c)] = [prefix[-1]] + list(c[:-1])
            targets[r, :len(c)] = c
        lp = np.asarray(runner("score", {"input_ids": ids, "targets": targets, **shared})["target_logprobs"], np.float64)
        calls += 1
        scores += [float(lp[r, :len(c)].sum()) for r, c in enumerate(part)]
    return scores, calls


# ---- checks against a full forward ------------------------------------------------------------------------

class torch_reference:
    """Full forward of a PyTorch model: ids [1,n] -> logits [1,n,V]."""

    def __init__(self, model):
        self.model = model

    @torch.inference_mode()
    def __call__(self, ids):
        return self.model(torch.as_tensor(np.asarray(ids)).long()).float().numpy()


def _full(reference, ids):
    return np.asarray(reference(np.asarray([ids], np.int32)), np.float32)[0]                     # [n,V]


def _log_softmax(x):
    x = np.asarray(x, np.float64)
    m = x.max(-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))


class Judge:
    def __init__(self, strict, tolerance=TOLERANCE):
        self.strict, self.tolerance = strict, tolerance
        self.worst, self.outside, self.rows = 0.0, 0, 0

    def __call__(self, got, want):
        got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
        error = float(np.abs(got - want).max())
        self.worst = max(self.worst, error)
        self.rows += 1
        if not np.allclose(got, want, atol=self.tolerance, rtol=self.tolerance):
            self.outside += 1
            if self.strict:
                raise AssertionError(f"differs by {error}")


def check_step(reference, runner, function, batch=16, tokens=1, prefix=11, seed=1118, strict=True, tolerance=TOLERANCE):
    """Three kinds of calls, each row's last-token scores vs the reference's full forward:
    1. the prompt (B=1, no prefix, junk in the masked P=1 and S=1 slots);
    2. each row's own first tokens (B=1 per row, after the shared prefix padded with junk slots);
    3. all rows together, `tokens` new tokens each, from different own positions (0..6), with junk in
       every masked slot of the prefix and the row caches."""
    rng = np.random.default_rng(seed)
    own = [(3 * b) % 7 for b in range(batch)]
    ids = rng.integers(8, VOCAB, (batch, prefix + max(own) + tokens)).astype(np.int32)
    ids[:, :prefix] = ids[0, :prefix]
    allowed = rng.choice(np.arange(8, VOCAB), 149, replace=False).astype(np.int32) if function == "scored" else None
    extra = {"allowed": allowed} if allowed is not None else {}
    judge = Judge(strict, tolerance)

    def want(seq):
        last = _full(reference, seq)[-1]
        return _log_softmax(last)[allowed] if allowed is not None else last

    out, _ = prompt_call(runner, function, ids[0, :prefix].tolist(), allowed, fill=30.0)
    judge(first_scores(function, out)[0], want(ids[0, :prefix]))
    shared = as_prefix(out["new_key"], out["new_value"], slots=prefix + 2, fill=30.0)
    states = []
    for b in range(batch):
        n = own[b]
        if not n:
            states.append((np.zeros((LAYERS, HEADS, 0, HEAD_DIM), np.float32),) * 2)
            continue
        o = runner(function, {"input_ids": ids[b:b + 1, prefix:prefix + n], "position": np.zeros(1, np.int32),
                              "key_cache": empty_cache(1, 30.0), "value_cache": empty_cache(1, -30.0), **shared, **extra})
        judge(first_scores(function, o)[0], want(ids[b, :prefix + n]))
        states.append((o["new_key"][:, 0], o["new_value"][:, 0]))
    k, v, position = pack(states, max(own) + 2, fill=30.0)
    new = np.stack([ids[b, prefix + own[b]:prefix + own[b] + tokens] for b in range(batch)])
    o = runner(function, {"input_ids": new, "position": position, "key_cache": k, "value_cache": v, **shared, **extra})
    got = first_scores(function, o)
    for b in range(batch):
        judge(got[b], want(ids[b, :prefix + own[b] + tokens]))
    return {"kind": function, "batch": batch, "tokens": tokens, "prefix": prefix, "rows": judge.rows,
            "max_abs": judge.worst, "rows_outside_tolerance": judge.outside}


def check_score(reference, runner, seed=1120, count=37, prefix=9, prefill="scored", sum_tolerance=SUM_TOLERANCE):
    """score_continuations vs the full forward: one shared prefix, `count` continuations of 1..12 tokens
    (three score calls of up to 16 rows), plus a one-token prefix (no prompt call)."""
    rng = np.random.default_rng(seed)
    worst, total = 0.0, 0
    for n in (prefix, 1):
        head = rng.integers(8, VOCAB, n).tolist()
        conts = [rng.integers(8, VOCAB, 1 + i % 12).tolist() for i in range(count)]
        got, _ = score_continuations(runner, head, conts, prefill=prefill)
        for c, g in zip(conts, got):
            seq = head + c
            lp = _log_softmax(_full(reference, seq))
            want = float(sum(lp[len(head) - 1 + j, c[j]] for j in range(len(c))))
            worst = max(worst, abs(g - want))
            total += 1
    return {"kind": "score", "continuations": total, "max_abs": worst, "tolerance_sum": sum_tolerance,
            "rows_outside_tolerance": 0 if worst <= sum_tolerance else 1}


CASES = ({"batch": 16}, {"batch": 1}, {"batch": 5, "tokens": 3}, {"batch": 16, "seed": 2026, "prefix": 64})


def run_checks(reference, runner, functions=FUNCTIONS, strict=True):
    checks = []
    for function in functions:
        if function == "score":
            if "scored" in functions or "logits" in functions:
                checks.append(check_score(reference, runner, prefill="scored" if "scored" in functions else "logits"))
            continue
        for case in CASES:
            checks.append(check_step(reference, runner, function, strict=strict, **case))
    return checks


# ---- tracing and conversion -------------------------------------------------------------------------------

def example_inputs(function, batch=2, tokens=3, slots=4, prefix=5):
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(8, VOCAB, (batch, tokens), generator=g, dtype=torch.int32)
    pk = torch.randn(LAYERS, 1, HEADS, prefix, HEAD_DIM, generator=g)
    pv = torch.randn(LAYERS, 1, HEADS, prefix, HEAD_DIM, generator=g)
    plen = torch.tensor([prefix - 1], dtype=torch.int32)
    if function == "score":
        return ids, torch.randint(8, VOCAB, (batch, tokens), generator=g, dtype=torch.int32), plen, pk, pv
    position = torch.tensor([b % slots for b in range(batch)], dtype=torch.int32)
    k = torch.randn(LAYERS, batch, HEADS, slots, HEAD_DIM, generator=g)
    v = torch.randn(LAYERS, batch, HEADS, slots, HEAD_DIM, generator=g)
    ex = (ids, position, k, v, plen, pk, pv)
    return ex + (torch.tensor([5, 6, 7], dtype=torch.int32),) if function == "scored" else ex


@torch.inference_mode()
def trace(model, function, vocab_chunks=1):
    """Trace one function and confirm on other shapes (B, T, S, P all different) that no size was baked in."""
    g = graph(model, function, vocab_chunks).eval()
    traced = torch.jit.trace(g, example_inputs(function))
    for shape in ({"batch": 3, "tokens": 1, "slots": 7, "prefix": 2}, {"batch": 1, "tokens": 6, "slots": 1, "prefix": 9}):
        ex = example_inputs(function, **shape)
        for a, b in zip(_tuple(traced(*ex)), _tuple(g(*ex))):
            torch.testing.assert_close(a, b, atol=TOLERANCE, rtol=TOLERANCE)
    return traced


def _tuple(x):
    return x if isinstance(x, tuple) else (x,)


def convert(bundle, output, function, vocab_chunks=1):
    ct = coremltools()
    output = Path(output)
    if output.exists():
        raise ValueError("Conversion output exists.")
    if function not in FUNCTIONS:
        raise ValueError(function)
    if function != "score" and vocab_chunks != 1:
        raise ValueError("vocab_chunks applies to the score function only.")
    lm = zh_v1.BundleLMZh(bundle)
    b = ct.RangeDim(lower_bound=1, upper_bound=MAX_BATCH, default=MAX_BATCH)
    t = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT, default=8 if function == "score" else 1)
    s = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT, default=8)
    p = ct.RangeDim(lower_bound=1, upper_bound=CONTEXT, default=32)
    tensor = lambda name, shape, dtype=np.float32: ct.TensorType(name=name, shape=shape, dtype=dtype)
    prefix = [tensor("prefix_length", (1,), np.int32), tensor("prefix_key", (LAYERS, 1, HEADS, p, HEAD_DIM)),
              tensor("prefix_value", (LAYERS, 1, HEADS, p, HEAD_DIM))]
    if function == "score":
        inputs = [tensor("input_ids", (b, t), np.int32), tensor("targets", (b, t), np.int32)] + prefix
        outputs = ["target_logprobs"]
    else:
        inputs = [tensor("input_ids", (b, t), np.int32), tensor("position", (b,), np.int32),
                  tensor("key_cache", (LAYERS, b, HEADS, s, HEAD_DIM)), tensor("value_cache", (LAYERS, b, HEADS, s, HEAD_DIM))] + prefix
        if function == "scored":
            inputs.append(ct.TensorType(name="allowed", shape=(ct.RangeDim(1, MAX_ALLOWED, 160),), dtype=np.int32))
            outputs = ["logsumexp", "allowed_logits", "new_key", "new_value"]
        else:
            outputs = ["logits", "new_key", "new_value"]
    model = ct.convert(trace(lm.model, function, vocab_chunks), source="pytorch", convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT32, inputs=inputs,
        outputs=[ct.TensorType(name=n, dtype=np.float32) for n in outputs], skip_model_load=True)
    model.short_description = f"vimeml-tiny-zh-v1 batched KV step v3 ({function}); shared prefix; FP32 compute."
    model.user_defined_metadata["bundle_manifest_sha256"] = lm.metadata["bundle_manifest_sha256"]
    zh_v1.finish(output, model, FORMAT, {"kind": "fp32", "minimum_ios": 18, "bundle": lm.metadata, "function": function,
        "vocab_chunks": vocab_chunks, "batch_range": [1, MAX_BATCH], "compute_units": "CPU_ONLY",
        "interface": interface(function), "graph": f"vimeml.deployment.zh_v1_batch3.{type(graph(lm.model, function, vocab_chunks)).__name__}",
        "precision_policy": "FP32 parameters and compute.", "policy": "Caller owns all buffers; inputs are never modified.",
        "conversion_source_sha256": file_sha(Path(__file__))})


def interface(function):
    prefix = {"prefix_length": "int32 [1]; valid prefix slots, 0 = no prefix (P=1 of zeros)",
              "prefix_key/prefix_value": "float32 (9,1,6,P,64), 1<=P<=128; shared by every row; slots >= prefix_length "
                                         "are masked but must be finite"}
    if function == "score":
        return {"input_ids": "int32 [B,T]; token t of row b sits at prefix_length + t", "targets": "int32 [B,T]; the "
                "token after input_ids[b,t]; right padding with any id", **prefix,
                "target_logprobs": "float32 [B,T]; log P(targets[b,t] | prefix + row b through t), normalized over all "
                                   "12,288 logits; ignore padded positions"}
    head = ({"allowed": f"int32 [K], 1<=K<={MAX_ALLOWED}; one id list for every row",
             "logsumexp": "float32 [B]; over all 12,288 logits of the last new token",
             "allowed_logits": "float32 [B,K]; log P = allowed_logits - logsumexp"} if function == "scored" else
            {"logits": f"float32 [B,{VOCAB}]; last new token, no softmax"})
    return {"input_ids": "int32 [B,T]", "position": "int32 [B]; row b's own cached tokens; token t of row b sits at "
            "prefix_length + position[b] + t (< 128)", "key_cache/value_cache": "float32 (9,B,6,S,64), 1<=S<=128; row b's "
            "own slots, slots >= position[b] masked but finite", **prefix, **head,
            "new_key/new_value": "float32 (9,B,6,T,64); the caller appends them to row b's own cache (or, for a prompt "
                                 "call, uses them as the prefix)"}


def compress(source, output, alignment):
    """INT8 block32 (deployed recipe) of a single-function FP32 v3 package whose FP32 merged package passed."""
    ct = coremltools()
    source, output, alignment = map(Path, (source, output, alignment))
    if output.exists():
        raise ValueError("Compression output exists.")
    original = zh_v1.verify_package(source, FORMAT)
    gate = read_json(alignment)
    sources = gate.get("source_manifests", {})
    if (original["kind"] != "fp32" or gate.get("format") != ALIGNMENT_FORMAT or not gate.get("passed")
            or sources.get(original["function"]) != file_sha(source / "manifest.json")):
        raise ValueError("First pass alignment of a merged FP32 package built from this exact package.")
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
    carried = {k: original[k] for k in ("minimum_ios", "bundle", "function", "vocab_chunks", "batch_range",
                                        "compute_units", "interface", "graph", "policy")}
    zh_v1.finish(output, result, FORMAT, {**carried, "kind": "int8-b32_fp32_compute", "recipe": zh_v1.DEPLOYED_RECIPE,
        "source_manifest_sha256": file_sha(source / "manifest.json"), "fp32_alignment_sha256": file_sha(alignment),
        "compression": {**spec, "weight_threshold": zh_v1.WEIGHT_THRESHOLD, "selection": selection},
        "precision_policy": "INT8 block32 parameter matrices, FP32 compute; norms unchanged; no activation quantization.",
        "conversion_source_sha256": file_sha(Path(__file__))})


def merge(packages, output):
    """One multifunction package from single-function v3 packages of one bundle and one precision.
    packages: {function: path}. Identical weight constants are stored once."""
    ct = coremltools()
    output = Path(output)
    if output.exists():
        raise ValueError("Merge output exists.")
    manifests = {f: zh_v1.verify_package(Path(p), FORMAT) for f, p in packages.items()}
    for f, m in manifests.items():
        if m["function"] != f:
            raise ValueError(f"{packages[f]} holds the {m['function']} function, not {f}.")
    first = next(iter(manifests.values()))
    if any(m["bundle"]["bundle_manifest_sha256"] != first["bundle"]["bundle_manifest_sha256"] or m["kind"] != first["kind"]
           for m in manifests.values()):
        raise ValueError("Packages differ in bundle or precision.")
    output.mkdir(parents=True)
    desc = ct.utils.MultiFunctionDescriptor()
    for f, p in packages.items():
        desc.add_function(str(Path(p) / "model.mlpackage"), src_function_name="main", target_function_name=f)
    desc.default_function_name = "scored" if "scored" in packages else next(iter(packages))
    ct.utils.save_multifunction(desc, str(output / "model.mlpackage"))
    files = v2.inventory(output / "model.mlpackage")
    separate = sum(m["package_bytes"] for m in manifests.values())
    write_json(output / "manifest.json", {"format": MERGED_FORMAT, "status": "complete",
        "architecture": zh_v1.ARCHITECTURE, "release_name": zh_v1.RELEASE, "kind": first["kind"],
        "bundle": first["bundle"], "minimum_ios": 18, "compute_units": "CPU_ONLY", "batch_range": [1, MAX_BATCH],
        "functions": {f: {"source_manifest_sha256": file_sha(Path(packages[f]) / "manifest.json"),
                          "vocab_chunks": m["vocab_chunks"], "interface": m["interface"]} for f, m in manifests.items()},
        **({"recipe": first["recipe"]} if "recipe" in first else {}),
        "default_function": desc.default_function_name,
        "loading": "MLModelConfiguration.functionName = \"logits\" | \"scored\" | \"score\" (iOS 18); one MLModel per function",
        "package_files": files, "package_bytes": sum(e["bytes"] for e in files.values()), "separate_package_bytes": separate,
        "conversion_source_sha256": file_sha(Path(__file__))})


def verify_merged(path):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if (manifest.get("format") != MERGED_FORMAT or manifest.get("status") != "complete"
            or manifest.get("architecture") != zh_v1.ARCHITECTURE):
        raise ValueError("Incomplete/unsupported merged v3 package.")
    if v2.inventory(path / "model.mlpackage") != manifest["package_files"]:
        raise ValueError("Package file inventory changed.")
    return manifest


def align(package, bundle, output, stateless=None):
    """Gate a merged v3 package, every function. Reference: PyTorch FP32 from the bundle, or (`stateless`)
    the stateless Core ML package of the same precision (Vime's INT8 gate)."""
    ct = coremltools()
    package, output = Path(package), Path(output)
    if output.exists():
        raise ValueError("Alignment output exists.")
    manifest = verify_merged(package)
    lm = zh_v1.BundleLMZh(bundle)
    if manifest["bundle"]["bundle_manifest_sha256"] != lm.metadata["bundle_manifest_sha256"]:
        raise ValueError("Package belongs to another bundle.")
    if stateless:
        ref_manifest = zh_v1.verify_package(stateless)
        model = ct.models.MLModel(str(Path(stateless) / "model.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
        reference = lambda ids: np.asarray(model.predict({"input_ids": np.asarray(ids, np.int32)})["logits"], np.float32)
        ref_name = f"stateless Core ML {ref_manifest['kind']} ({Path(stateless).name})"
    else:
        reference = torch_reference(lm.model.float().eval())
        ref_name = "PyTorch FP32 full forward (bundle)"
    runner = CoreMLRunner(package)
    checks = run_checks(reference, runner, tuple(manifest["functions"]), strict=False)
    for c in checks:
        print(json.dumps(c), flush=True)
    passed = bool(checks) and all(c["rows_outside_tolerance"] == 0 for c in checks)
    write_json(output, {"format": ALIGNMENT_FORMAT, "passed": passed, "reference": ref_name,
        "coreml_manifest_sha256": file_sha(package / "manifest.json"), "package": package.name, "kind": manifest["kind"],
        "source_manifests": {f: e["source_manifest_sha256"] for f, e in manifest["functions"].items()},
        "tolerance": {"atol": TOLERANCE, "rtol": TOLERANCE, "per_sum": SUM_TOLERANCE}, "checks": checks,
        "method": "logits/scored: prompt call, per-row own-token calls after a shared prefix, then all rows together from "
                  "different own positions, with finite junk in every masked slot; each row's last-token logits (or "
                  "log-probs on 149 allowed ids) vs the reference's full forward. score: score_continuations sums over 74 "
                  "continuations (prefix of 9 tokens and of 1 token) vs the reference.",
        "conversion_source_sha256": file_sha(Path(__file__))})
    print(f"alignment {'passed' if passed else 'FAILED'}: max {max(c['max_abs'] for c in checks):.3g}")
    return passed


# ---- search backend ---------------------------------------------------------------------------------------

class BatchBackendV3:
    """zh_search backend over a runner. The prompt becomes the shared prefix; each row's state holds only
    the tokens it added after the prompt, so a bucket never copies the prompt into its rows."""

    def __init__(self, runner, scored=True):
        self.runner, self.scored = runner, scored
        self.function = "scored" if scored else "logits"
        self.prefix = no_prefix()
        self.calls = self.rows = self.prefill_calls = 0

    def prefill(self, ids, allowed=None):
        out, self.prefix = prompt_call(self.runner, self.function, list(ids), allowed)
        self.prefill_calls += 1
        empty = np.zeros((LAYERS, HEADS, 0, HEAD_DIM), np.float32)
        return torch.from_numpy(np.array(first_scores(self.function, out)[0])), (empty, empty)

    def decode(self, tokens, states, allowed=None):
        k, v, position = pack(states)
        feed = {"input_ids": np.asarray(tokens, np.int32)[:, None], "position": position, "key_cache": k,
                "value_cache": v, **self.prefix}
        if self.scored:
            feed["allowed"] = np.asarray(allowed, np.int32)
        out = self.runner(self.function, feed)
        self.calls += 1
        self.rows += len(states)
        new = [(np.concatenate([sk, out["new_key"][:, r]], 2), np.concatenate([sv, out["new_value"][:, r]], 2))
               for r, (sk, sv) in enumerate(states)]
        return torch.from_numpy(np.array(first_scores(self.function, out))), new
