"""Pinyin-constrained beam search for the zh models, beam kept per key position (the Vime contract).

Every output token must be Han characters typeable in the reading dictionary, and the output must
spell the raw keys exactly, each character typed as one of its readings in full or as its initial
(first letter, or zh/ch/sh). One token can have several spellings, so a hypothesis is
(tokens, key position, score).

Hypotheses compete only with others that consumed the same number of keys. Positions are expanded
in increasing order; at each position the bucket is deduplicated by token ids, cut to `beam`, and
forwarded in one batched one-token call (each row feeds its own last token). lm mode finishes when
the keys are used up; task mode adds log P(EOS) at the end.

A backend provides `prefill(ids) -> (logits [V], state)` and `decode(tokens, states) -> (logits [B,V],
new states)`, as `zh_v1_batch.TorchBatchBackend` and `CoreMLBatchBackend` do.
"""
import math
from dataclasses import dataclass, field
from itertools import product

BOS, EOS, CTX, READ, OUT = 2, 3, 4, 5, 6
LIMIT = 128
MAX_SPELLINGS = 4096


def char_spellings(readings):
    """{spelling: 0 if a full reading, 1 if an initial}."""
    out = {r: 0 for r in readings}
    for r in readings:
        out.setdefault(r[:2] if r[:2] in ("zh", "ch", "sh") else r[0], 1)
        out.setdefault(r[0], 1)
    return out


class Index:
    """spelling -> {token id: fewest characters typed as initials}."""

    def __init__(self, sp, table):
        chars = {}
        for c in table.characters():
            rs = [r[0] for r, _ in table.words[c] if len(r) == 1]
            if rs:
                chars[c] = char_spellings(rs)
        self.char_spellings = {s for v in chars.values() for s in v}
        self.by_spelling, self.max_len, self.skipped = {}, 0, 0
        for tid in range(sp.get_piece_size()):
            if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_byte(tid):
                continue
            piece = sp.id_to_piece(tid)
            if not piece or any(c not in chars for c in piece):
                continue
            options = [list(chars[c].items()) for c in piece]
            if math.prod(len(o) for o in options) > MAX_SPELLINGS:
                self.skipped += 1
                continue
            for parts in product(*options):
                s = "".join(p for p, _ in parts)
                n = sum(a for _, a in parts)
                slot = self.by_spelling.setdefault(s, {})
                slot[tid] = min(n, slot.get(tid, n))
                self.max_len = max(self.max_len, len(s))

    def reachable(self, keys):
        """ok[p]: keys[p:] can be typed as some characters."""
        n = len(keys)
        ok = [False] * (n + 1)
        ok[n] = True
        for p in range(n - 1, -1, -1):
            ok[p] = any(ok[e] for e in range(p + 1, min(n, p + 6) + 1) if keys[p:e] in self.char_spellings)
        return ok

    def moves(self, keys, pos, ok):
        """[(token id, new position, characters typed as initials)] from key position pos."""
        out = []
        for end in range(pos + 1, min(len(keys), pos + self.max_len) + 1):
            if ok[end]:
                out.extend((tid, end, n) for tid, n in self.by_spelling.get(keys[pos:end], {}).items())
        return out


@dataclass
class SearchStats:
    calls: int = 0          # batched decode calls (one per non-empty bucket that needs a forward)
    rows: int = 0           # hypotheses forwarded = batch-1 predicts an unbatched runtime would make
    buckets: list = field(default_factory=list)   # rows per decode call


def prompt_ids(sp, keys, context="", mode="lm"):
    ctx = sp.encode(context) if context else []
    if mode == "lm":
        return [BOS] + ctx
    return [BOS, CTX] + ctx + [READ] + sp.encode(keys) + [OUT]


def log_softmax(logits):
    return logits.float().log_softmax(-1)


def search(backend, sp, index, keys, context="", mode="lm", beam=16, topk=10):
    """[(text, score, token ids)] best first (distinct texts), and SearchStats."""
    stats = SearchStats()
    base = prompt_ids(sp, keys, context, mode)
    ok = index.reachable(keys)
    if not ok[0] or len(base) >= LIMIT:
        return [], stats
    n = len(keys)
    logits, state = backend.prefill(base)
    # bucket entry: (tokens, score, parent log-probs row or None, parent state, last token)
    root_lp = log_softmax(logits)
    buckets = {0: [((), 0.0, root_lp, state)]}
    done = {}

    def finish(toks, score):
        text = sp.decode(list(toks))
        if score > done.get(text, (-math.inf,))[0]:
            done[text] = (score, list(toks))

    def expand(toks, score, lp, st, pos):
        for tid, end, _ in index.moves(keys, pos, ok):
            buckets.setdefault(end, []).append((toks + (tid,), score + float(lp[tid]), None, (st, tid)))

    for pos in range(n + 1):
        hyps = buckets.pop(pos, None)
        if not hyps:
            continue
        hyps.sort(key=lambda h: -h[1])
        kept, seen = [], set()
        for h in hyps:
            if h[0] not in seen and len(base) + len(h[0]) < LIMIT:
                seen.add(h[0])
                kept.append(h)
            if len(kept) == beam:
                break
        if pos == n and mode == "lm":
            for toks, score, _, _ in kept:
                finish(toks, score)
            break
        ready = [h for h in kept if h[2] is not None]          # only the root (pos 0) has log-probs already
        pending = [h for h in kept if h[2] is None]
        if pending:
            out, states = backend.decode([h[3][1] for h in pending], [h[3][0] for h in pending])
            stats.calls += 1
            stats.rows += len(pending)
            stats.buckets.append(len(pending))
            lps = log_softmax(out)
            ready += [(h[0], h[1], lps[r], states[r]) for r, h in enumerate(pending)]
        for toks, score, lp, st in ready:
            if pos == n:
                finish(toks, score + float(lp[EOS]))
            else:
                expand(toks, score, lp, st, pos)
    ranked = sorted(((t, s, ids) for t, (s, ids) in done.items()), key=lambda x: -x[1])[:topk]
    return ranked, stats
