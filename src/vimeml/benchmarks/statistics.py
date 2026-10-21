"""Cluster-aware paired statistics for frozen IME tracks.

Items written by the same author, or built from the same sentence template, tend to
flip together, so the effective sample is smaller than the item count and per-item
McNemar p-values are optimistic. Clusters join items that share an author/document
group (the homophone group on the challenge track) or whose NFKC targets (digits masked) have character-trigram Jaccard >= TEMPLATE.
Paired tests then resample or sign-flip whole clusters.
"""
import random
import re
import unicodedata
from math import comb

TEMPLATE = 0.5
_DIGITS = re.compile(r'[0-9０-９]+')


def exact_mcnemar(gain, loss):
    n, k = gain + loss, min(gain, loss)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def _trigrams(text):
    text = _DIGITS.sub('0', unicodedata.normalize('NFKC', text))
    return {text[i:i + 3] for i in range(max(1, len(text) - 2))}


def _group(row):
    p = row.get('provenance') or {}
    if str(p.get('group_id', '')).startswith('homophone:'):
        return p['group_id']  # paired items of one homophone group
    for key in ('author_id', 'source_doc_id'):
        if p.get(key) is not None:
            return f"{row.get('source', '')}:{key}:{p[key]}"
    return None


def clusters(rows, template=TEMPLATE):
    """Cluster label per row: union of shared author/document group and template similarity."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    first = {}
    for i, row in enumerate(rows):
        group = _group(row)
        if group is not None:
            union(i, first.setdefault(group, i))
    if template is not None:
        grams = [_trigrams(r['answers'][0]) for r in rows]
        for i in range(len(rows)):
            for j in range(i):
                if len(grams[i] & grams[j]) >= template * len(grams[i] | grams[j]):
                    union(i, j)
    return [find(i) for i in range(len(rows))]


def _cluster_sums(diff, labels):
    sums = {}
    for d, c in zip(diff, labels):
        sums[c] = sums.get(c, 0) + d
    return list(sums.values())


def sign_flip_p(diff, labels, draws=20000, seed=0):
    """Two-sided paired permutation p-value flipping the sign of whole clusters."""
    sums = [s for s in _cluster_sums(diff, labels) if s]
    observed = abs(sum(sums))
    if not sums:
        return 1.0
    if len(sums) <= 20:
        hits = sum(abs(sum(s if mask >> i & 1 else -s for i, s in enumerate(sums))) >= observed
                   for mask in range(2 ** len(sums)))
        return hits / 2 ** len(sums)
    rng = random.Random(seed)
    hits = sum(abs(sum(s if rng.random() < .5 else -s for s in sums)) >= observed for _ in range(draws))
    return (hits + 1) / (draws + 1)


def bootstrap_interval(diff, labels, draws=10000, seed=0, level=.95):
    """Percentile interval of the net paired difference, resampling clusters with replacement."""
    sums = _cluster_sums(diff, labels)
    rng = random.Random(seed)
    totals = sorted(sum(rng.choice(sums) for _ in sums) for _ in range(draws))
    tail = (1 - level) / 2
    return totals[int(tail * draws)], totals[min(draws - 1, int((1 - tail) * draws))]


def _cluster_sizes(labels):
    sizes = {}
    for c in labels:
        sizes[c] = sizes.get(c, 0) + 1
    return sizes


def design_effect(diff, labels):
    """Variance of the paired total under clustering relative to independent items."""
    mean = sum(diff) / len(diff)
    iid = sum((d - mean) ** 2 for d in diff)
    sums, sizes = {}, _cluster_sizes(labels)
    for d, c in zip(diff, labels):
        sums[c] = sums.get(c, 0) + d
    clustered = sum((s - mean * sizes[c]) ** 2 for c, s in sums.items())
    return clustered / iid if iid else 1.0


def paired(before, after, labels, seed=0):
    """Paired comparison of two boolean outcome lists on the same frozen cases."""
    gain = sum(b and not a for a, b in zip(before, after))
    loss = sum(a and not b for a, b in zip(before, after))
    diff = [int(b) - int(a) for a, b in zip(before, after)]
    low, high = bootstrap_interval(diff, labels, seed=seed)
    return {'gain': gain, 'loss': loss, 'net': gain - loss, 'exact_mcnemar_p': exact_mcnemar(gain, loss),
            'clusters': len(set(labels)), 'discordant_clusters': sum(1 for s in _cluster_sums(diff, labels) if s),
            'cluster_sign_flip_p': sign_flip_p(diff, labels, seed=seed),
            'cluster_bootstrap_95': [low, high], 'design_effect': design_effect(diff, labels)}
