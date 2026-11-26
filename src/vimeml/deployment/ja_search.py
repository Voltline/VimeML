"""Kana-to-text beam search for the ja-gen models, bucketed by consumed reading position.

Task-mode prompt:
    [BOS, CTX] + encode(context[-64:]) + [READ] + encode(reading) + [OUT]
Left context tokens are trimmed from the left if needed so the prompt leaves room for generation
within the model's context window (LIMIT = 128).

Two search modes share the same batched KV backend contract (`prefill(ids) -> (logits, state)`,
`decode(tokens, states) -> (logits, new_states)`, as provided by `zh_torch_backend.ModelBatchBackend`):

- `constrained` (default): hypotheses compete only with others that consumed the same number of
  hiragana characters from `reading`. Positions `1..len(reading)` are expanded in order, each
  bucket deduplicated by token ids, cut to `beam` (16), and forwarded in one batched decode call;
  at `len(reading)` the score adds `log P(EOS)`.
- `free`: unconstrained step-by-step beam search in task mode (useful as a diagnostic comparison
  or an explicit fallback when an input reading is unreachable under the piece-reading index).
"""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re

import numpy as np

from vimeml.deployment.zh_search import BOS, CTX, EOS, LIMIT, OUT, READ, log_softmax

CONTEXT_CHARS = 64
KANA_PUNCT = re.compile(r"^[ぁ-ゖァ-ヺー、。！？「」『』（）・…〜～]+$")
HIRAGANA_READING = re.compile(r"^[ぁ-ゖー、。！？「」『』（）・…〜～]+$")
# Deviation D5 overrides (same rules as scripts/ja/corpus/kana_samples.py)
D5_OVERRIDES = {
    "私": ("わたし", "わたくし"),
    "日本": ("にほん", "にっぽん"),
    "明日": ("あした", "あす"),
    "言う": ("いう", "ゆう"),
    "何": ("なに", "なん"),
}


def to_hiragana(text):
    """Convert katakana characters to hiragana, keeping punctuation and long-vowel marks."""
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def prompt_ids(sp, reading, context="", limit=LIMIT, min_output_room=2):
    """Build task-mode prompt ids, trimming left context on the left if needed."""
    ctx = sp.encode(context[-CONTEXT_CHARS:]) if context else []
    read = sp.encode(reading)
    fixed = 4 + len(read)  # BOS, CTX, READ, read..., OUT
    max_ctx = max(0, limit - min_output_room - fixed)
    if len(ctx) > max_ctx:
        ctx = ctx[-max_ctx:] if max_ctx > 0 else []
    return [BOS, CTX] + ctx + [READ] + read + [OUT]


@dataclass
class SearchStats:
    calls: int = 0
    rows: int = 0
    buckets: list = field(default_factory=list)
    mode: str = "constrained"
    fallback: bool = False


class KanaIndex:
    """Hiragana reading -> set of allowed token ids in the SentencePiece vocabulary."""

    def __init__(self, by_reading, max_len=None):
        self.by_reading = {r: sorted(set(tids)) for r, tids in by_reading.items() if r and tids}
        self.max_len = max_len if max_len is not None else max((len(r) for r in self.by_reading), default=0)

    def reachable(self, reading):
        """ok[p]: reading[p:] can be covered by a sequence of pieces in this index."""
        n = len(reading)
        ok = [False] * (n + 1)
        ok[n] = True
        for p in range(n - 1, -1, -1):
            hi = min(n, p + self.max_len)
            for end in range(p + 1, hi + 1):
                if ok[end] and reading[p:end] in self.by_reading:
                    ok[p] = True
                    break
        return ok

    def save(self, path, metadata=None):
        payload = {"format": "vimeml_ja_kana_index_v1", "max_len": self.max_len,
                   "readings": len(self.by_reading), "by_reading": self.by_reading}
        if metadata:
            payload["metadata"] = metadata
        Path(path).write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "vimeml_ja_kana_index_v1":
            raise ValueError(f"Unexpected index format in {path}")
        return cls(data["by_reading"], max_len=data.get("max_len"))


def _dictionary_readings(piece, mecab, sudachi, modes):
    """Collect candidate hiragana readings for one vocabulary piece from UniDic-lite and Sudachi."""
    out = set()
    if KANA_PUNCT.fullmatch(piece):
        h = to_hiragana(piece)
        out.add(h)
        # Allow voiced-kana equivalents (ヂ/ヅ <-> じ/ず) when matching readings
        if "ぢ" in h or "づ" in h:
            out.add(h.replace("ぢ", "じ").replace("づ", "ず"))
        return out

    # UniDic-lite reading
    parts, ok = [], True
    for w in mecab(piece):
        s = w.surface
        if KANA_PUNCT.fullmatch(s):
            parts.append(to_hiragana(s))
        elif w.feature.kana:
            parts.append(to_hiragana(w.feature.kana))
        else:
            ok = False
            break
    if ok and parts:
        r = "".join(parts)
        if HIRAGANA_READING.fullmatch(r):
            out.add(r)

    # Sudachi readings across split modes A, B, C
    for mode in modes:
        parts, ok = [], True
        for w in sudachi.tokenize(piece, mode):
            s = w.surface()
            if KANA_PUNCT.fullmatch(s):
                parts.append(to_hiragana(s))
            else:
                rf = w.reading_form()
                if rf:
                    parts.append(to_hiragana(rf))
                else:
                    ok = False
                    break
        if ok and parts:
            r = "".join(parts)
            if HIRAGANA_READING.fullmatch(r):
                out.add(r)

    return out


def build_index(sp, reading_table_path=None, extra_piece_readings=None):
    """Build a KanaIndex over every usable piece of `sp` using UniDic-lite, Sudachi, D4 and D5."""
    import fugashi
    from sudachipy import Dictionary, SplitMode

    mecab = fugashi.Tagger()
    if "unidic_lite" not in mecab.dictionary_info[0]["filename"]:
        raise RuntimeError("fugashi did not load unidic-lite; uninstall the `unidic` package")
    sudachi = Dictionary().create()
    modes = (SplitMode.A, SplitMode.B, SplitMode.C)

    d4 = {}
    if reading_table_path and Path(reading_table_path).exists():
        for line in Path(reading_table_path).read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) >= 4:
                surface = cols[0]
                for col in cols[1:4]:
                    if col:
                        h = to_hiragana(col)
                        if HIRAGANA_READING.fullmatch(h):
                            d4.setdefault(surface, set()).add(h)

    by_reading = {}
    covered_pieces = 0
    for tid in range(sp.get_piece_size()):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_byte(tid):
            continue
        piece = sp.id_to_piece(tid)
        if not piece or piece in ("<CTX>", "<READ>", "<OUT>", "\n"):
            continue
        readings = _dictionary_readings(piece, mecab, sudachi, modes)
        if piece in d4:
            readings |= d4[piece]
        if piece in D5_OVERRIDES:
            readings |= set(D5_OVERRIDES[piece])
        if extra_piece_readings and piece in extra_piece_readings:
            readings |= {r for r in extra_piece_readings[piece] if HIRAGANA_READING.fullmatch(r)}
        if readings:
            covered_pieces += 1
            for r in readings:
                by_reading.setdefault(r, []).append(tid)

    return KanaIndex(by_reading), {"vocab_size": sp.get_piece_size(), "covered_pieces": covered_pieces}


def _best_bucket(chunks, beam):
    """Top `beam` distinct token sequences from bucket chunks in stable descending score order."""
    values = np.concatenate([c[3] for c in chunks]) if len(chunks) > 1 else chunks[0][3]
    owner = np.repeat(np.arange(len(chunks)), [len(c[2]) for c in chunks])
    start = np.cumsum([0] + [len(c[2]) for c in chunks])
    order = np.argsort(-values, kind="stable")
    kept, seen = [], set()
    for lo in range(0, len(order), 4 * beam):
        for i in order[lo:lo + 4 * beam].tolist():
            c = int(owner[i])
            toks, st, tids, _ = chunks[c]
            tid = int(tids[i - int(start[c])])
            child = toks + (tid,)
            if child not in seen:
                seen.add(child)
                kept.append((child, float(values[i]), st, tid))
                if len(kept) == beam:
                    return kept
    return kept


def search_constrained(backend, sp, index, reading, context="", beam=16, topk=10):
    """Kana-position-bucketed beam search in task mode, ending with log P(EOS)."""
    stats = SearchStats(mode="constrained")
    base = prompt_ids(sp, reading, context)
    ok = index.reachable(reading)
    if not ok[0] or len(base) >= LIMIT:
        return [], stats
    n = len(reading)
    room = LIMIT - len(base)
    groups = {}

    def moves_at(pos):
        g = groups.get(pos)
        if g is None:
            g = groups[pos] = []
            for end in range(pos + 1, min(n, pos + index.max_len) + 1):
                if ok[end]:
                    tids = index.by_reading.get(reading[pos:end])
                    if tids:
                        g.append((end, tids, np.array(tids, dtype=np.int64)))
        return g

    logits, state = backend.prefill(base)
    root_lp = log_softmax(logits)
    buckets, done = {}, {}

    def finish(toks, score):
        text = sp.decode(list(toks))
        if score > done.get(text, (-math.inf,))[0]:
            done[text] = (score, list(toks))

    def expand(toks, score, lp, st, pos):
        if len(toks) + 1 >= room:
            return
        for end, tids, arr in moves_at(pos):
            values = lp[arr].astype(np.float64) + score
            buckets.setdefault(end, []).append((toks, st, tids, values))

    if n == 0:
        finish((), float(root_lp[EOS]))
    expand((), 0.0, root_lp, state, 0)
    for pos in range(1, n + 1):
        chunks = buckets.pop(pos, None)
        if not chunks:
            continue
        kept = _best_bucket(chunks, beam)
        out, states = backend.decode([h[3] for h in kept], [h[2] for h in kept])
        lps = log_softmax(out)
        stats.calls += 1
        stats.rows += len(kept)
        stats.buckets.append(len(kept))
        for r, (toks, score, _, _) in enumerate(kept):
            if pos == n:
                finish(toks, score + float(lps[r][EOS]))
            else:
                expand(toks, score, lps[r], states[r], pos)

    ranked = sorted(((t, s, ids) for t, (s, ids) in done.items()), key=lambda x: -x[1])[:topk]
    return ranked, stats


def _disallowed_ids(sp):
    """Reserved / control / unknown / byte token ids not emitted directly in free beam search."""
    bad = {0, 1, 2, 4, 5, 6, 7}  # pad, unk, bos, CTX, READ, OUT, newline (EOS=3 allowed)
    for tid in range(sp.get_piece_size()):
        if tid != EOS and (sp.is_control(tid) or sp.is_unknown(tid)):
            bad.add(tid)
    return sorted(bad)


def search_free(backend, sp, reading, context="", beam=16, topk=10, max_tokens=None):
    """Unconstrained task-mode beam search (step-bucketed), ending at EOS."""
    stats = SearchStats(mode="free")
    base = prompt_ids(sp, reading, context)
    if len(base) >= LIMIT:
        return [], stats
    room = LIMIT - len(base)
    steps = min(room - 1, max_tokens if max_tokens is not None else max(8, len(reading) + 8))
    banned = np.array(_disallowed_ids(sp), dtype=np.int64)

    logits, state = backend.prefill(base)
    root_lp = log_softmax(logits)
    root_lp[banned] = -math.inf

    done = {}

    def finish(toks, score):
        text = sp.decode(list(toks))
        if text and score > done.get(text, (-math.inf,))[0]:
            done[text] = (score, list(toks))

    # First step from root
    order = np.argsort(-root_lp, kind="stable")[:beam + 1]
    active = []
    for tid in order.tolist():
        val = float(root_lp[tid])
        if not math.isfinite(val):
            continue
        if tid == EOS:
            continue
        active.append(((tid,), val, state, tid))
        if len(active) == beam:
            break

    for _ in range(1, steps):
        if not active:
            break
        out, states = backend.decode([h[3] for h in active], [h[2] for h in active])
        lps = log_softmax(out)
        lps[:, banned] = -math.inf
        stats.calls += 1
        stats.rows += len(active)
        stats.buckets.append(len(active))

        candidates = []
        for r, (toks, score, _, _) in enumerate(active):
            row_lp = lps[r]
            eos_val = float(row_lp[EOS])
            if math.isfinite(eos_val):
                finish(toks, score + eos_val)
            # Pick top-`beam` non-EOS continuations per parent
            row_no_eos = row_lp.copy()
            row_no_eos[EOS] = -math.inf
            top = np.argpartition(-row_no_eos, min(beam, len(row_no_eos) - 1))[:beam]
            top = top[np.argsort(-row_no_eos[top], kind="stable")]
            for tid in top.tolist():
                v = float(row_no_eos[tid])
                if math.isfinite(v):
                    candidates.append((toks + (tid,), score + v, states[r], tid))

        candidates.sort(key=lambda x: -x[1])
        next_active, seen = [], set()
        for cand in candidates:
            if cand[0] not in seen:
                seen.add(cand[0])
                next_active.append(cand)
                if len(next_active) == beam:
                    break
        active = next_active

    ranked = sorted(((t, s, ids) for t, (s, ids) in done.items()), key=lambda x: -x[1])[:topk]
    return ranked, stats


def search(backend, sp, index, reading, context="", beam=16, topk=10,
           search_mode="constrained", fallback_free=True):
    """Unified entry point for Japanese task-mode beam search."""
    if search_mode == "free":
        return search_free(backend, sp, reading, context, beam=beam, topk=topk)
    if search_mode != "constrained":
        raise ValueError(f"Unknown search_mode: {search_mode}")
    ranked, stats = search_constrained(backend, sp, index, reading, context, beam=beam, topk=topk)
    if not ranked and fallback_free:
        ranked, stats = search_free(backend, sp, reading, context, beam=beam, topk=topk)
        stats.fallback = True
    return ranked, stats
