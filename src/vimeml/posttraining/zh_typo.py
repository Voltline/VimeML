"""Keyboard typos on pinyin keys (26-key QWERTY) for the `typo` kind of the zh three-task data.

A typo sample is a full or mixed pinyin sample (zh_pinyin.Pinyin.sample) whose keys get one or
two letter edits; the output stays the correct text. Edits never touch the apostrophe and two
edits never fall in the same syllable. Edit kinds (weights from the [typo] config):

  substitute  a letter becomes one of its QWERTY neighbours (same row left/right, the two
              diagonal keys above and below)
  delete      a letter is dropped (a one-letter syllable may vanish)
  insert      a letter is added after a letter of the syllable: with repeat_share the same
              letter again (double press), otherwise a neighbour of it
  transpose   two adjacent, different letters of the syllable swap

A typo can leave keys that are still valid full pinyin (shi -> si); full_pinyin() tells, and the
builder counts both. Every choice comes from the caller's random.Random.
"""
from functools import lru_cache

ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
EDITS = ("substitute", "delete", "insert", "transpose")


def _neighbours():
    where = {c: (r, i) for r, row in enumerate(ROWS) for i, c in enumerate(row)}
    out = {}
    for c, (r, i) in where.items():
        near = ((r, i - 1), (r, i + 1), (r - 1, i), (r - 1, i + 1), (r + 1, i - 1), (r + 1, i))
        out[c] = "".join(ROWS[a][b] for a, b in near if 0 <= a < len(ROWS) and 0 <= b < len(ROWS[a]))
    return out


NEIGHBOURS = _neighbours()


def letter_units(keys, units):
    """Unit index of every character of keys (None for an apostrophe); units are the syllables'
    keys without apostrophes, as zh_pinyin.Pinyin.sample records them."""
    labels, k = [], 0
    for u, text in enumerate(units):
        if k < len(keys) and keys[k] == "'":
            labels.append(None)
            k += 1
        if keys[k:k + len(text)] != text:
            raise ValueError(f"units do not spell keys: {keys!r} {units!r}")
        labels += [u] * len(text)
        k += len(text)
    if k != len(keys):
        raise ValueError(f"units do not spell keys: {keys!r} {units!r}")
    return labels


def _edit(rng, chars, labels, unit, kind, repeat_share):
    """Apply one edit inside `unit` in place. False if the kind does not fit the unit."""
    at = [i for i, x in enumerate(labels) if x == unit]
    if kind == "transpose":
        pairs = [i for i in at if i + 1 in at and chars[i] != chars[i + 1]]
        if not pairs:
            return False
        i = rng.choice(pairs)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return True
    i = rng.choice(at)
    if kind == "substitute":
        chars[i] = rng.choice(NEIGHBOURS[chars[i]])
    elif kind == "delete":
        del chars[i], labels[i]
    elif kind == "insert":
        new = chars[i] if rng.random() < repeat_share else rng.choice(NEIGHBOURS[chars[i]])
        chars.insert(i + 1, new)
        labels.insert(i + 1, unit)
    else:
        raise ValueError(kind)
    return True


def _tidy(chars):
    """Drop apostrophes left at either end or doubled by a deleted one-letter syllable."""
    out = []
    for c in chars:
        if c == "'" and (not out or out[-1] == "'"):
            continue
        out.append(c)
    while out and out[-1] == "'":
        out.pop()
    return "".join(out)


def add_typos(rng, keys, units, config):
    """(typo keys, [edit kinds]) or None. config: two_share, weights {edit: w}, repeat_share."""
    letters = [u for u, text in enumerate(units) if text]
    if not letters:
        return None
    count = 2 if len(letters) >= 2 and rng.random() < config["two_share"] else 1
    chosen = sorted(rng.sample(letters, count), reverse=True)        # right to left: earlier indices stay valid
    chars, labels = list(keys), letter_units(keys, units)
    weights = config["weights"]
    done = []
    for unit in chosen:
        kinds = [k for k in EDITS if weights.get(k, 0) > 0]
        while kinds:
            kind = rng.choices(kinds, weights=[weights[k] for k in kinds])[0]
            if _edit(rng, chars, labels, unit, kind, config["repeat_share"]):
                done.append(kind)
                break
            kinds.remove(kind)
        else:
            return None
    out = _tidy(chars)
    if not out or out.replace("'", "") == keys.replace("'", ""):
        return None
    return out, done[::-1]


def full_pinyin(keys, syllables):
    """True if keys split into valid full syllables (apostrophes are fixed boundaries)."""
    syllables = frozenset(syllables)
    longest = max(map(len, syllables))

    @lru_cache(maxsize=None)
    def ok(part, start):
        if start == len(part):
            return True
        return any(part[start:end] in syllables and ok(part, end)
                   for end in range(start + 1, min(len(part), start + longest) + 1))

    return all(piece and ok(piece, 0) for piece in keys.split("'"))
