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

import numpy as np

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
    """Row log-probs as a host array: the search reads one entry per move, and reading device tensors
    entry by entry costs a device sync each (about 4 ms per hypothesis on CUDA)."""
    if hasattr(logits, "detach"):
        return logits.detach().float().log_softmax(-1).cpu().numpy()
    x = np.asarray(logits, dtype=np.float32)
    m = x.max(-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))


def search(backend, sp, index, keys, context="", mode="lm", beam=16, topk=10):
    """[(text, score, token ids)] best first (distinct texts), and SearchStats."""
    stats = SearchStats()
    base = prompt_ids(sp, keys, context, mode)
    ok = index.reachable(keys)
    if not ok[0] or len(base) >= LIMIT:
        return [], stats
    n = len(keys)
    # A backend with `scored = True` takes the bucket's allowed ids and returns log-probs on them only
    # (a dict id -> log-prob row); otherwise it returns full logits and the search normalizes.
    scored = getattr(backend, "scored", False)
    room = LIMIT - len(base)              # a hypothesis may hold fewer tokens than this
    groups = {}

    def moves_at(pos):
        """[(end, token ids, the same as an array)], in Index.moves order; cached per position."""
        g = groups.get(pos)
        if g is None:
            g = groups[pos] = []
            for end in range(pos + 1, min(n, pos + index.max_len) + 1):
                if ok[end]:
                    tids = list(index.by_spelling.get(keys[pos:end], ()))
                    if tids:
                        g.append((end, tids, np.array(tids, dtype=np.int64)))
        return g

    def allowed(pos):
        return sorted({t for _, tids, _ in moves_at(pos) for t in tids} | ({EOS} if pos == n else set()))

    if scored:
        ids0 = allowed(0)
        lp0, state = backend.prefill(base, ids0)
        root_lp = dict(zip(ids0, lp0.tolist()))
    else:
        logits, state = backend.prefill(base)
        root_lp = log_softmax(logits)
    # Children are kept per target position as chunks (parent tokens, parent state, token ids, scores),
    # appended in the order the per-hypothesis loop made them, and only turned into hypotheses when kept.
    buckets = {}
    done = {}

    def finish(toks, score):
        text = sp.decode(list(toks))
        if score > done.get(text, (-math.inf,))[0]:
            done[text] = (score, list(toks))

    def expand(toks, score, lp, st, pos):
        if len(toks) + 1 >= room:
            return
        for end, tids, arr in moves_at(pos):
            if isinstance(lp, dict):
                values = np.array([lp[t] for t in tids], dtype=np.float64) + score
            else:
                values = lp[arr].astype(np.float64) + score
            buckets.setdefault(end, []).append((toks, st, tids, values))

    def best(chunks):
        """The bucket's top `beam` distinct hypotheses (tokens, score, parent state, last token): the
        stable descending order of the old list sort, read lazily."""
        values = np.concatenate([c[3] for c in chunks]) if len(chunks) > 1 else chunks[0][3]
        owner = np.repeat(np.arange(len(chunks)), [len(c[2]) for c in chunks])
        start = np.cumsum([0] + [len(c[2]) for c in chunks])
        order = np.argsort(-values, kind="stable")
        kept, seen = [], set()
        for lo in range(0, len(order), 4 * beam):
            for i in order[lo:lo + 4 * beam].tolist():
                c = int(owner[i])
                toks, st, tids, _ = chunks[c]
                tid = tids[i - int(start[c])]
                child = toks + (tid,)
                if child not in seen:
                    seen.add(child)
                    kept.append((child, float(values[i]), st, tid))
                    if len(kept) == beam:
                        return kept
        return kept

    if n == 0:
        finish((), 0.0 if mode == "lm" else float(root_lp[EOS]))
    expand((), 0.0, root_lp, state, 0)
    for pos in range(1, n + 1):
        chunks = buckets.pop(pos, None)
        if not chunks:
            continue
        kept = best(chunks)
        if pos == n and mode == "lm":
            for toks, score, _, _ in kept:
                finish(toks, score)
            break
        if scored:
            ids = allowed(pos)
            out, states = backend.decode([h[3] for h in kept], [h[2] for h in kept], ids)
            lps = [dict(zip(ids, row.tolist())) for row in out]
        else:
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
