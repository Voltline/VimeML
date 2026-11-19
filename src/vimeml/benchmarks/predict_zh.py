"""Decoder, simulation and metrics of the Chinese prediction benchmark ime-predict-zh-v1.

Protocol: docs/zh-CN/benchmarks/predict-zh-plan.md; implementation details fixed before any
score: docs/zh-CN/benchmarks/predict-ime-zh.md (解码与模拟的实现细节).

Every language model goes through the same decoder; only the adapter differs (how turns are
joined into token ids, and next-token log-probabilities):

- context: the adapter joins earlier turns and the typed prefix in its own training format
  and drops the oldest whole turns that do not fit, keeping RESERVE tokens for decoding;
- token healing: when the prefix ends in Chinese and its last token is pure Chinese, that
  token is taken back and every hypothesis must start by re-spelling it;
- beam search, width 16, one token per step, only tokens whose text is entirely Chinese
  characters, at most 6 new characters; every hypothesis ending on a token boundary with
  2-6 new characters is a candidate, scored by the sum of its token log-probabilities
  (the best path when several spell the same string);
- the top 5 candidates by score that are not prefixes of one another;
- track C keeps only hypotheses compatible with the typed keys (rime-ice single-character
  readings).
"""
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch

HAN = re.compile("[〇㐀-䶿一-鿿豈-﫿]")
HAN_RUN = re.compile("[〇㐀-䶿一-鿿豈-﫿]+")
BEAM = 16
MAX_CHARS = 6
MIN_CHARS = 2
TOP = 5
SHOW = 3
RESERVE = 12
NEG = float("-inf")


def all_han(text):
    return bool(text) and all(HAN.fullmatch(c) for c in text)


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_readings(dict_dir):
    """Character -> set of syllables (ü as v) from the rime-ice single-character tables."""
    readings = {}
    for name in ("8105.dict.yaml", "41448.dict.yaml"):
        body = False
        with open(Path(dict_dir) / name, encoding="utf-8") as stream:
            for line in stream:
                line = line.rstrip("\n")
                if line == "...":
                    body = True
                    continue
                if not body or not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2 and len(parts[0]) == 1 and parts[1].isalpha():
                    readings.setdefault(parts[0], set()).add(parts[1])
    return readings


@dataclass(frozen=True)
class Constraint:
    """kind 'syllable': keys is the full first syllable; kind 'initials': one letter per character."""
    kind: str
    keys: str

    @property
    def max_chars(self):
        return len(self.keys) if self.kind == "initials" else MAX_CHARS

    def candidate_length_ok(self, n):
        return n == len(self.keys) if self.kind == "initials" else MIN_CHARS <= n <= MAX_CHARS

    def chars_ok(self, readings, chars, done):
        """Can `chars` follow `done` already generated characters?"""
        if self.kind == "syllable":
            return done > 0 or not chars or self.keys in readings.get(chars[0], ())
        if done + len(chars) > len(self.keys):
            return False
        return all(any(r.startswith(self.keys[done + i]) for r in readings.get(c, ())) for i, c in enumerate(chars))


class Vocabulary:
    """Per-token facts the decoder needs, built once per adapter."""

    def __init__(self, texts, device):
        self.texts = texts
        self.device = device
        han = [t is not None and all_han(t) for t in texts]
        self.han = torch.tensor(han, device=device)
        self.chars = torch.tensor([len(t) if h else 0 for t, h in zip(texts, han)], device=device)
        self.han_ids = [i for i, h in enumerate(han) if h]
        self.by_first = {}
        for i in self.han_ids:
            self.by_first.setdefault(texts[i][0], []).append(i)
        self._length = {}
        self._constraint = {}
        self._allowed = {}

    def length_mask(self, room):
        if room not in self._length:
            self._length[room] = self.han & (self.chars <= room)
        return self._length[room]

    def constraint_mask(self, constraint, readings, done):
        key = (constraint.kind, constraint.keys[done:] if constraint.kind == "initials" else constraint.keys,
               done if constraint.kind == "syllable" else None)
        if key not in self._constraint:
            if constraint.kind == "syllable" and done > 0:
                mask = self.han
            else:
                ok = [i for i in self.han_ids if constraint.chars_ok(readings, self.texts[i], done)]
                mask = torch.zeros_like(self.han)
                mask[ok] = True
            self._constraint[key] = mask
        return self._constraint[key]

    def allowed_mask(self, length, constraint):
        """length & constraint, cached so that equal masks stay one object."""
        key = (id(length), id(constraint))
        if key not in self._allowed:
            self._allowed[key] = length & constraint
        return self._allowed[key]


def healing(lm, ids, prefix):
    """Take back the last context token when it is a pure-Chinese spelling of the prefix's end."""
    if prefix and HAN.fullmatch(prefix[-1]) and len(ids) > 1:
        text = lm.vocabulary.texts[ids[-1]]
        if text is not None and all_han(text) and prefix.endswith(text):
            return ids[:-1], text
    return ids, ""


def select(scored, top=TOP):
    """Highest scores first; skip a candidate that is a prefix of a chosen one or has one as prefix."""
    chosen = []
    for text, score in sorted(scored.items(), key=lambda x: (-x[1], x[0])):
        if any(text.startswith(c) or c.startswith(text) for c, _ in chosen):
            continue
        chosen.append((text, score))
        if len(chosen) == top:
            break
    return chosen


class _Search:
    """One predict() call as a step machine, so that many can share each forward pass.

    requests() gives, per beam state, its token sequence and what the step needs from the
    next-token distribution: either the tokens that re-spell the healed text (pending) or
    the mask of allowed tokens. advance() takes, per state, the (logprob, token) pairs in
    the decoder's order and keeps the best `beam` expansions.
    """

    def __init__(self, lm, turns, prefix, constraint, readings, beam, top):
        self.vocabulary, self.constraint, self.readings, self.beam, self.top = (
            lm.vocabulary, constraint, readings, beam, top)
        self.ids, self.required = healing(lm, lm.context(turns, prefix), prefix)
        self.max_chars = constraint.max_chars if constraint else MAX_CHARS
        if len(self.ids) + len(self.required) + self.max_chars > lm.context_length:
            raise ValueError("Context leaves no room for decoding")
        self.candidates = {}
        self.states = [((), "", 0.0)]

    def requests(self):
        vocabulary, constraint, required = self.vocabulary, self.constraint, self.required
        out = []
        for suffix, text, _ in self.states:
            sequence = self.ids + list(suffix)
            pending = required[len(text):]
            if pending:
                tokens = []
                for token in vocabulary.by_first.get(pending[0], ()):
                    piece = vocabulary.texts[token]
                    if piece.startswith(pending):
                        extra = piece[len(pending):]
                        if len(extra) > self.max_chars or (
                                constraint and not constraint.chars_ok(self.readings, extra, 0)):
                            continue
                    elif not pending.startswith(piece):
                        continue
                    tokens.append(token)
                out.append((sequence, tokens, None))
                continue
            done = len(text) - len(required)
            mask = vocabulary.length_mask(self.max_chars - done)
            if constraint:
                mask = vocabulary.allowed_mask(mask, vocabulary.constraint_mask(constraint, self.readings, done))
            out.append((sequence, None, mask))
        return out

    def advance(self, answers):
        texts, required, constraint = self.vocabulary.texts, self.required, self.constraint
        expansions = []
        for (suffix, text, score), pairs in zip(self.states, answers):
            for value, token in pairs:
                expansions.append((score + value, suffix + (token,), text + texts[token]))
        expansions.sort(key=lambda x: (-x[0], x[2]))
        self.states = []
        for score, suffix, text in expansions[:self.beam]:
            if len(text) < len(required):
                self.states.append((suffix, text, score))
                continue
            new = text[len(required):]
            ok = constraint.candidate_length_ok(len(new)) if constraint else MIN_CHARS <= len(new) <= MAX_CHARS
            if ok and score > self.candidates.get(new, NEG):
                self.candidates[new] = score
            if len(new) < self.max_chars:
                self.states.append((suffix, text, score))

    def result(self):
        return select(self.candidates, self.top)


def _answers_rowwise(logprobs, requests, beam):
    """One state at a time on the adapter's (CPU) distribution."""
    answers = []
    for row, (_, tokens, mask) in enumerate(requests):
        if tokens is not None:
            answers.append([(float(logprobs[row, token]), token) for token in tokens])
            continue
        allowed = int(mask.sum())
        if not allowed:
            answers.append([])
            continue
        values, ids = logprobs[row].masked_fill(~mask, NEG).topk(min(beam, allowed))
        answers.append(list(zip(values.tolist(), ids.tolist())))
    return answers


def _answers_batched(logprobs, requests, beam):
    """All states at once on the device; only the top `beam` per row leave it."""
    answers = [[] for _ in requests]
    free = [i for i, r in enumerate(requests) if r[2] is not None]
    if free:
        slots, unique, index = {}, [], []
        for i in free:
            mask = requests[i][2]  # masks are cached by the vocabulary, so equal masks are one object
            if id(mask) not in slots:
                slots[id(mask)] = len(unique)
                unique.append(mask)
            index.append(slots[id(mask)])
        device = logprobs.device
        stack = torch.stack(unique)
        allowed = stack.sum(1).tolist()
        chosen = stack.to(device)[torch.tensor(index, device=device)]
        values, ids = logprobs[torch.tensor(free, device=device)].masked_fill(~chosen, NEG).topk(
            min(beam, logprobs.shape[1]))
        values, ids = values.tolist(), ids.tolist()
        for j, i in enumerate(free):
            n = min(beam, allowed[index[j]])
            answers[i] = list(zip(values[j][:n], ids[j][:n]))
    pend = [(i, t) for i, r in enumerate(requests) if r[1] is not None for t in r[1]]
    if pend:
        rows = torch.tensor([i for i, _ in pend], device=logprobs.device)
        cols = torch.tensor([t for _, t in pend], device=logprobs.device)
        for (i, t), value in zip(pend, logprobs[rows, cols].tolist()):
            answers[i].append((value, t))
    return answers


GROUP = 1024  # searches decoded in lockstep


def predict_many(lm, jobs, readings=None, beam=BEAM, top=TOP):
    """predict() for each (turns, prefix, constraint) job, in order.

    An adapter with prefill/extend (the Vime models) decodes up to GROUP jobs of similar context
    length in lockstep, each context computed once; the others decode the jobs one at a time.
    """
    searches = [_Search(lm, turns, prefix, constraint, readings, beam, top) for turns, prefix, constraint in jobs]
    if not hasattr(lm, "prefill"):
        for search in searches:
            while search.states:
                requests = search.requests()
                search.advance(_answers_rowwise(lm.logprobs([r[0] for r in requests]), requests, beam))
        return [s.result() for s in searches]
    order = sorted(range(len(searches)), key=lambda i: len(searches[i].ids))
    start = 0
    while start < len(order):  # groups of similar context length whose caches fit together
        end = min(len(order), start + GROUP)
        end = min(end, start + lm.cache_jobs(len(searches[order[end - 1]].ids)))
        _decode_group(lm, [searches[i] for i in order[start:end]], beam)
        start = end
    return [s.result() for s in searches]


def _decode_group(lm, group, beam):
    """Every search's beam steps in lockstep: the contexts once (prefill), then each step runs
    only the suffixes (every state of every search has the same number of new tokens)."""
    cache, first = lm.prefill([s.ids for s in group])
    requests = [s.requests() for s in group]  # one state each, empty suffix
    answers = _answers_batched(first, [r[0] for r in requests], beam)
    for search, answer in zip(group, answers):
        search.advance([answer])
    while True:
        alive = [i for i, s in enumerate(group) if s.states]
        if not alive:
            return
        per_search = [group[i].requests() for i in alive]
        size = len(per_search[0][0][0]) - len(group[alive[0]].ids)
        width = max(len(r) for r in per_search)
        tokens = torch.zeros((len(alive), width, size), dtype=torch.long)
        rows = []
        for a, (i, requests) in enumerate(zip(alive, per_search)):
            for b, (sequence, _, _) in enumerate(requests):
                suffix = sequence[len(group[i].ids):]
                if len(suffix) != size:
                    raise AssertionError("searches out of step")
                tokens[a, b] = torch.tensor(suffix)
                rows.append(a * width + b)
        logprobs = lm.extend(cache, alive, tokens.to(lm.device)).reshape(len(alive) * width, -1)
        flat = [r for requests in per_search for r in requests]
        answers = _answers_batched(logprobs[torch.tensor(rows, device=logprobs.device)], flat, beam)
        at = 0
        for i, requests in zip(alive, per_search):
            group[i].advance(answers[at:at + len(requests)])
            at += len(requests)


def predict(lm, turns, prefix, constraint=None, readings=None, beam=BEAM, top=TOP):
    """Top candidates [(text, score)] after `prefix`, with earlier `turns` as context."""
    return predict_many(lm, [(turns, prefix, constraint)], readings, beam, top)[0]


class VimeLM:
    """vimeml-tiny-zh models: each turn is its own BOS ... EOS sequence, as in training."""

    def __init__(self, model, processor, device, special):
        self.model, self.processor, self.device, self.special = model, processor, device, special
        self.context_length = model.config.context_length
        texts = []
        for i in range(processor.get_piece_size()):
            plain = not (processor.is_control(i) or processor.is_unknown(i) or processor.is_byte(i))
            texts.append(processor.id_to_piece(i) if plain else None)
        self.vocabulary = Vocabulary(texts, "cpu")

    def context(self, turns, prefix):
        bos, eos = self.special["bos"], self.special["eos"]
        limit = self.context_length - RESERVE
        tail = self.processor.encode(prefix) if prefix else []
        ids = [bos, *tail[-(limit - 1):]] if tail else [bos]
        for turn in reversed(turns):
            if not turn:
                continue
            piece = [bos, *self.processor.encode(turn), eos]
            if len(ids) + len(piece) > limit:
                break
            ids = piece + ids
        return ids

    @torch.inference_mode()
    def logprobs(self, sequences):
        inputs = torch.tensor(sequences, dtype=torch.long, device=self.device)
        return self.model(inputs)[:, -1].float().log_softmax(-1).cpu()

    # Batched decoding with a context cache (used by predict_many). Same blocks as the model's
    # forward, written out so that keys and values can be kept: qkv, attention, projection, ffn.

    def cache_jobs(self, width):
        """Contexts of `width` tokens whose cache fits in about 1 GiB."""
        config = self.model.config
        return max(1, (1 << 30) // (8 * config.n_layers * config.d_model * width))

    def _embed(self, tokens, positions):
        m = self.model
        if hasattr(m, "layer_qkv"):  # zh-800m arch2: RoPE inside attention, no position table
            return m.embed(tokens, positions)
        return m.token_embedding(tokens) + m.position_embedding(positions)

    def _qkv(self, block, hidden, positions):
        """query [batch, heads, size, head_dim]; key and value with the model's KV heads (GQA: fewer)."""
        if hasattr(self.model, "layer_qkv"):
            return self.model.layer_qkv(block, hidden, positions)
        heads = self.model.config.n_heads
        batch, size, width = hidden.shape
        qkv = block.attention.qkv(block.attention_norm(hidden)).reshape(batch, size, 3, heads, width // heads)
        return qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [batch, heads, size, head_dim]

    def _block(self, block, hidden, attended):
        batch, size, width = hidden.shape
        hidden = hidden + block.attention.projection(attended.transpose(1, 2).reshape(batch, size, width))
        return hidden + block.ffn(block.ffn_norm(hidden))

    def _head(self, hidden):
        return self.model.lm_head(self.model.final_norm(hidden)).float().log_softmax(-1)

    @torch.inference_mode()
    def prefill(self, contexts):
        """One causal pass over right-padded contexts: (cache, next-token logprobs [P, V])."""
        m, device = self.model, self.device
        n = torch.tensor([len(c) for c in contexts], device=device)
        width = int(n.max())
        tokens = torch.tensor([c + [0] * (width - len(c)) for c in contexts], device=device)
        positions = torch.arange(width, device=device)
        mask = ((positions[:, None] >= positions[None, :])[None] & (positions[None, :] < n[:, None])[:, None])[:, None]
        hidden = self._embed(tokens, positions)
        keys, values = [], []
        for block in m.blocks:
            query, key, value = self._qkv(block, hidden, positions)
            attended = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask,
                                                                        enable_gqa=True)
            hidden = self._block(block, hidden, attended)
            keys.append(key)
            values.append(value)
        last = hidden[torch.arange(len(contexts), device=device), n - 1]
        return {"keys": torch.stack(keys), "values": torch.stack(values), "n": n}, self._head(last)

    QUERY_TOKENS = 1 << 15

    @torch.inference_mode()
    def extend(self, cache, alive, tokens):
        """Logprobs [A, B, V] after each beam's suffix. tokens [A, B, s]: B suffixes of s tokens for
        context alive[a]; a suffix attends to its context and causally to itself only."""
        m, device = self.model, self.device
        count, beams, size = tokens.shape
        step = max(1, self.QUERY_TOKENS // (beams * size))
        out = []
        for start in range(0, count, step):
            rows = torch.tensor(alive[start:start + step], device=device)
            part = tokens[start:start + step]
            batch = part.shape[0]
            keys, values, n = cache["keys"][:, rows], cache["values"][:, rows], cache["n"][rows]
            width = keys.shape[3]
            offsets = torch.arange(size, device=device)
            where = (n[:, None] + offsets).repeat(1, beams)                                 # [batch, B*s]
            hidden = self._embed(part.reshape(batch, beams * size), where)
            beam = torch.arange(beams, device=device).repeat_interleave(size)
            at = offsets.repeat(beams)
            own = (beam[:, None] == beam[None, :]) & (at[:, None] >= at[None, :])          # [B*s, B*s]
            seen = torch.arange(width, device=device)[None, :] < n[:, None]                # [batch, W]
            mask = torch.cat([seen[:, None, :].expand(batch, beams * size, width),
                              own[None].expand(batch, -1, -1)], 2)[:, None]
            for layer, block in enumerate(m.blocks):
                query, key, value = self._qkv(block, hidden, where)
                attended = torch.nn.functional.scaled_dot_product_attention(
                    query, torch.cat([keys[layer], key], 2), torch.cat([values[layer], value], 2), attn_mask=mask,
                    enable_gqa=True)
                hidden = self._block(block, hidden, attended)
            out.append(self._head(hidden.reshape(batch, beams, size, -1)[:, :, -1]))
        return torch.cat(out)


# Track B ------------------------------------------------------------------------------------

def keystrokes(message, pinyin):
    """Per-character full-pinyin key count: letters of the reading for Chinese, 1 otherwise."""
    costs = [1] * len(message)
    for run in HAN_RUN.finditer(message):
        readings = pinyin(run.group())
        if len(readings) != len(run.group()):
            raise ValueError("Reading count differs from character count")
        for k, reading in enumerate(readings):
            costs[run.start() + k] = len(reading)
    return costs


def boundaries(message, cut):
    out, position = [], 0
    for word in cut(message):
        position += len(word)
        out.append(position)
    if position != len(message):
        raise ValueError("Segmentation does not cover the message")
    return out


def simulate(message, costs, ends, predictions, threshold, show=SHOW):
    """Type `message`; after every commit (not before the first one) up to `show` predictions
    with score >= threshold are offered, and the longest one that is a prefix of the rest is
    taken (one key). Otherwise the next segment (to the next word end) is typed in full pinyin.
    predictions(p) -> [(text, score)]."""
    position = keys = shows = clicks = 0
    while position < len(message):
        offered = []
        if position:
            offered = [text for text, score in predictions(position)[:show] if score >= threshold]
        shows += bool(offered)
        hits = [text for text in offered if message.startswith(text, position)]
        if hits:
            position += len(max(hits, key=len))
            keys += 1
            clicks += 1
            continue
        end = next(e for e in ends if e > position)
        keys += sum(costs[position:end])
        position = end
    return {"keys": keys, "baseline_keys": sum(costs), "shows": shows, "clicks": clicks}


# Metrics ------------------------------------------------------------------------------------

def rank_of_hit(candidates, continuation):
    for rank, (text, _) in enumerate(candidates, 1):
        if continuation.startswith(text):
            return rank, len(text)
    return None, 0


def rank_exact(candidates, target):
    for rank, (text, _) in enumerate(candidates, 1):
        if text == target:
            return rank
    return None


def rates(rows, key, ks=(1, 3, 5)):
    """rows: dicts with 'source' and key -> rank or None. Micro and macro over sources."""
    sources = sorted({r["source"] for r in rows})
    out = {}
    for k in ks:
        per = {s: sum(1 for r in rows if r["source"] == s and r[key] and r[key] <= k) /
               max(1, sum(1 for r in rows if r["source"] == s)) for s in sources}
        hits = sum(1 for r in rows if r[key] and r[key] <= k)
        out[f"top{k}"] = {"micro": hits / max(1, len(rows)), "hits": hits, "items": len(rows),
                          "macro": sum(per.values()) / max(1, len(per)), "by_source": per}
    return out


def coverage_curve(rows, points=(0.2, 0.5)):
    """Track D: Top-1 hit rate among the most confident fraction of track A items."""
    ordered = sorted(rows, key=lambda r: (-(r["confidence"] if r["confidence"] is not None else NEG), r["id"]))
    out = {}
    for c in points:
        n = max(1, round(c * len(ordered)))
        out[f"{int(c * 100)}%"] = {"items": n, "top1": sum(1 for r in ordered[:n] if r["rank"] == 1) / n}
    return out


def finite(x):
    return x if x is not None and math.isfinite(x) else None
