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


def predict(lm, turns, prefix, constraint=None, readings=None, beam=BEAM, top=TOP):
    """Top candidates [(text, score)] after `prefix`, with earlier `turns` as context."""
    vocabulary = lm.vocabulary
    ids, required = healing(lm, lm.context(turns, prefix), prefix)
    max_chars = constraint.max_chars if constraint else MAX_CHARS
    if len(ids) + len(required) + max_chars > lm.context_length:
        raise ValueError("Context leaves no room for decoding")
    candidates = {}
    states = [((), "", 0.0)]
    while states:
        logprobs = lm.logprobs([ids + list(suffix) for suffix, _, _ in states])
        expansions = []
        for row, (suffix, text, score) in enumerate(states):
            pending = required[len(text):]
            if pending:
                for token in vocabulary.by_first.get(pending[0], ()):
                    piece = vocabulary.texts[token]
                    if piece.startswith(pending):
                        extra = piece[len(pending):]
                        if len(extra) > max_chars or (constraint and not constraint.chars_ok(readings, extra, 0)):
                            continue
                    elif not pending.startswith(piece):
                        continue
                    expansions.append((score + float(logprobs[row, token]), suffix + (token,), text + piece))
                continue
            done = len(text) - len(required)
            mask = vocabulary.length_mask(max_chars - done)
            if constraint:
                mask = mask & vocabulary.constraint_mask(constraint, readings, done)
            allowed = int(mask.sum())
            if not allowed:
                continue
            values, tokens = logprobs[row].masked_fill(~mask, NEG).topk(min(beam, allowed))
            for value, token in zip(values.tolist(), tokens.tolist()):
                expansions.append((score + value, suffix + (token,), text + vocabulary.texts[token]))
        expansions.sort(key=lambda x: (-x[0], x[2]))
        states = []
        for score, suffix, text in expansions[:beam]:
            if len(text) < len(required):
                states.append((suffix, text, score))
                continue
            new = text[len(required):]
            ok = constraint.candidate_length_ok(len(new)) if constraint else MIN_CHARS <= len(new) <= MAX_CHARS
            if ok and score > candidates.get(new, NEG):
                candidates[new] = score
            if len(new) < max_chars:
                states.append((suffix, text, score))
    return select(candidates, top)


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
