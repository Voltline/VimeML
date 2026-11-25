"""Build and freeze ime-kana-ja-v1 (ja-gen plan, stage 0.2): kana -> text conversion spans.

Sources are WRIME ver2 and JMultiWOZ user turns. Every post and dialogue that entered an earlier
Japanese benchmark draft or release is excluded. All authors and users had already appeared
(deviation 1 of the plan), so the new development/test split is author/user-disjoint only
within this release. Readings are kept only where UniDic and Sudachi agree, as in the Standard
draft; there is no manual review. No language model and no candidate export is used here.

    python scripts/ja/benchmarks/prepare_kana_ime_ja.py build
"""
import argparse
import collections
import difflib
import hashlib
import json
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/ja/benchmarks'))
from vimeml.benchmarks.readings import DualReading
from vimeml.training.data import write_json
from prepare_standard_ime import extract, normalized, sources
from prepare_standard_ime_v2 import chat_lines, fresh, read
from standard_ime import notice, track_files

VERSION = 'ime-kana-ja-v1'
SEED = 20261125
BENCH = ROOT / 'artifacts/benchmarks'
QUOTAS = {'wrime': 400, 'jmultiwoz': 200}
POOL = {'wrime': 1600, 'jmultiwoz': 800}
AUTHOR_CAP = 20
ROLES = ('development', 'test')
WEB = ROOT / 'outputs/corpus-fast-v1/train.txt'


def length_band(row):
    n = len(row['expected_output'][0])
    return 'short' if n <= 10 else 'medium' if n <= 20 else 'long'


def earlier_files():
    """Every item file of every earlier Japanese benchmark, draft, review copy and candidate import."""
    names = ('evaluation_items.json', 'candidate-pool.json')
    return sorted(p for p in BENCH.rglob('*.json') if p.name in names and VERSION not in p.parts)


def earlier_items():
    documents, queries, used = set(), set(), []
    for path in earlier_files():
        rows = read(path)
        if not isinstance(rows, list):
            continue
        hits = 0
        for row in rows:
            p = row.get('provenance') or {}
            if p.get('source') in QUOTAS and 'document_id' in p:
                documents.add((p['source'], p['document_id']))
                hits += 1
            if 'input' in row:
                queries.add((row.get('context_text', ''), row['input']))
        if hits:
            used.append(str(path.relative_to(BENCH)).replace('\\', '/'))
    return documents, queries, used


def assign_roles(records, rng):
    """Half of each source's authors/users go to each role (seeded); the old v1 roles are ignored."""
    roles = {}
    for source in QUOTAS:
        groups = sorted({r['group_id'] for r in records if r['source'] == source})
        rng.shuffle(groups)
        roles.update({g: ROLES[i % 2] for i, g in enumerate(groups)})
    for r in records:
        r['role'] = roles[r['group_id']]
    return {s: {role: sum(1 for g, v in roles.items() if g.startswith(s) and v == role) for role in ROLES}
            for s in QUOTAS}


def build_pool(rng, documents, queries):
    records, _ = sources(ROOT / 'datasets/ime-standard-ja-v1', rng)
    groups = assign_roles(records, rng)
    reader = DualReading()
    pool, rejects, counts, seen_docs, signatures = [], collections.Counter(), collections.Counter(), set(), set()
    for record in records:
        key = record['source'], record['role']
        doc = record['source'], record['document_id']
        if counts[key] >= POOL[record['source']] or doc in seen_docs:
            continue
        if doc in documents:
            rejects['earlier_benchmark_document'] += 1
            continue
        case, reason = extract(record, reader, rng)
        if case is None:
            rejects[f'{record["source"]}:{reason}'] += 1
            continue
        signature = normalized(case['provenance']['cleaned_message'])
        query = case['context_text'], case['input']
        if query in queries:
            rejects['earlier_benchmark_query'] += 1
            continue
        if signature in signatures or query in signatures:
            rejects['duplicate_message_or_query_context'] += 1
            continue
        signatures.update((signature, query))
        seen_docs.add(doc)
        pool.append(case)
        counts[key] += 1
        if len(pool) % 500 == 0:
            print(f'pool {len(pool)}', flush=True)
        if all(counts[(s, r)] >= POOL[s] for s in QUOTAS for r in ROLES):
            break
    stats = {'pool_counts': {f'{s}/{r}': n for (s, r), n in sorted(counts.items())}, 'groups_per_role': groups,
             'rejections': dict(rejects), 'dictionary_versions': reader.versions}
    return pool, stats


def text_lines():
    with WEB.open(encoding='utf-8') as stream:
        for line in stream:
            yield 'corpus-fast-v1/train.txt', line
    for split, line in chat_lines():
        yield f'corpus-v41-chat-8k/{split}', line


def overlap_scan(pool):
    """Normalized containment or SequenceMatcher >= .85 near a 12-character anchor, as in Standard v1/v2."""
    texts = [normalized(r['provenance']['cleaned_message']) for r in pool]
    lookup = collections.defaultdict(set)
    for i, text in enumerate(texts):
        if len(text) >= 12:
            lookup[text[:12]].add(i)
            lookup[text[-12:]].add(i)
    import ahocorasick
    matcher = ahocorasick.Automaton()
    for anchor, ids in lookup.items():
        matcher.add_word(anchor, (anchor, tuple(ids)))
    matcher.make_automaton()
    excluded, matches, counts = set(), [], collections.Counter()
    for n, (name, line) in enumerate(text_lines(), 1):
        counts[name] += 1
        text = normalized(line)
        for end, (anchor, ids) in matcher.iter(text):
            for i in ids:
                if i in excluded:
                    continue
                candidate = texts[i]
                if candidate in text:
                    ratio = 1.0
                else:
                    location = end - len(anchor) + 1
                    if anchor == candidate[-12:]:
                        location -= len(candidate) - 12
                    window = text[max(0, location):max(0, location) + len(candidate)]
                    ratio = difflib.SequenceMatcher(None, candidate, window, autojunk=False).ratio()
                if ratio >= .85:
                    excluded.add(i)
                    matches.append({'pool_position': i, 'file': name, 'line': counts[name], 'ratio': round(ratio, 3)})
        if n % 2_000_000 == 0:
            print(f'overlap scan {n:,} lines, {len(excluded)} excluded', flush=True)
    return excluded, {'files': dict(counts), 'excluded': len(excluded), 'matches': matches,
                      'method': '12-character anchors; normalized containment or SequenceMatcher >= .85',
                      'limits': 'Messages under 12 normalized characters are not screened. Stage-1 training text is '
                                'deduplicated against this release on the training side.'}


def select(pool, excluded, rng):
    chosen = {}
    for role in ROLES:
        rows_out = []
        for source, target in QUOTAS.items():
            buckets = collections.defaultdict(list)
            for i, row in enumerate(pool):
                p = row['provenance']
                if i not in excluded and p['source'] == source and p['role'] == role:
                    buckets[(bool(row['context_text']), length_band(row))].append(row)
            for values in buckets.values():
                rng.shuffle(values)
            authors, selected = collections.Counter(), []
            while len(selected) < target and any(buckets.values()):
                for key in sorted(buckets):
                    if not buckets[key] or len(selected) == target:
                        continue
                    row = buckets[key].pop()
                    if authors[row['provenance']['author_id']] >= AUTHOR_CAP:
                        continue
                    authors[row['provenance']['author_id']] += 1
                    row['index'] = f'{VERSION}-{role}-{source}-{len(selected) + 1:04}'
                    selected.append(row)
            if len(selected) != target:
                raise ValueError(f'{source}/{role}: only {len(selected)} eligible cases; no padding.')
            rows_out.extend(selected)
        chosen[role] = rows_out
    return chosen


def hiragana(text):
    return ''.join(chr(ord(c) - 0x60) if 'ァ' <= c <= 'ヶ' else c for c in text)


def build(args):
    out = args.output
    fresh(out)
    rng = random.Random(SEED)
    documents, queries, earlier = earlier_items()
    print(f'{len(documents)} earlier documents from {len(earlier)} files', flush=True)
    pool, stats = build_pool(rng, documents, queries)
    write_json(out / 'candidate-pool.json', pool)
    excluded, audit = overlap_scan(pool)
    write_json(out / 'overlap-audit.json', audit)
    tracks = select(pool, excluded, random.Random(SEED + 1))
    groups = [{r['provenance']['group_id'] for r in tracks[role]} for role in ROLES]
    if groups[0] & groups[1]:
        raise ValueError('Development and test authors/users overlap.')
    for role, rows in tracks.items():
        for row in rows:
            row['reading_hiragana'] = hiragana(row['input'])
            row['review'] = {'status': 'two_dictionary_agreement', 'human_native_review': False,
                             'not_formal_gold': True, 'acceptable_forms_exhaustive': False}
        track_files(out / role, rows)
    strata = {role: dict(sorted(collections.Counter(
        f'{"with" if r["context_text"] else "without"}-context/{length_band(r)}' for r in rows).items()))
        for role, rows in tracks.items()}
    manifest = {
        'format': 'vimeml_kana_ime_ja_v1', 'version': VERSION, 'release_id': uuid.uuid4().hex,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
        'splits': {role: len(rows) for role, rows in tracks.items()},
        'sources': {role: dict(collections.Counter(r['provenance']['source'] for r in rows)) for role, rows in tracks.items()},
        'authors': {role: len(g) for role, g in zip(ROLES, groups)}, 'strata': strata,
        'labels_frozen': True, 'review_completed': False, 'label_quality': 'two_dictionary_agreement',
        'test_policy': 'Sealed: read once, in the preregistered confirmation of the ja-gen plan (stage 5).',
        'metric': 'Top-1 of the displayed list, notation-lenient (vimeml.benchmarks.notation); exact Top-1 also reported',
        'exclusions': {'earlier_benchmark_files': earlier, 'earlier_documents': len(documents), 'author_cap': AUTHOR_CAP,
                       'dialogue_cap': 'one item per JMultiWOZ dialogue, one per WRIME post'},
        'build': stats,
        'licenses': {'wrime': 'CC-BY-NC-ND-4.0; local research use only',
                     'jmultiwoz': 'CC-BY-SA-4.0'},
        'models_scored': False, 'candidates_exported': False,
    }
    write_json(out / 'manifest.json', manifest)
    notice(out)
    sums = {str(p.relative_to(out)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob('*')) if p.is_file()}
    (out / 'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n, h in sums.items()), encoding='utf-8', newline='\n')
    print(json.dumps({k: manifest[k] for k in ('release_id', 'splits', 'sources', 'authors', 'strata')},
                     ensure_ascii=False, indent=1))
    print('manifest sha256', sums['manifest.json'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--output', type=Path, default=BENCH / VERSION)
    args = parser.parse_args()
    build(args)


if __name__ == '__main__':
    main()
