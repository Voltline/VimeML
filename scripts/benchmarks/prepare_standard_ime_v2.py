"""Second Standard IME blind release, drawn with the v1 protocol from unused v1 pool cases.

The v1 draft pool already holds conversion spans extracted, without LM scores or
candidate coverage, from blind-role WRIME writers and JMultiWOZ users, and screened
against the old training text. Only 750 of its blind-role cases were ever selected.
This release selects 500 WRIME + 250 JMultiWOZ new cases from the rest:

- documents, cleaned messages and context/reading pairs of every v1 development or
  blind case are excluded, so no v1 dialogue or post is reused;
- an added screen against the V4.1+ chat bank (RealPersonaChat/MRMP) removes contained
  or near-duplicate messages;
- at most 20 cases per writer/user, and no target within character-trigram Jaccard
  0.5 of an already selected target (limits JMultiWOZ template families);
- the same context/length strata as v1.

Commands: prepare (unscored draft), review-pack (AI review sheet with dictionary
spelling proposals), freeze (AI expert labels, never native gold).
"""
import argparse
import collections
import copy
import difflib
import json
import random
import re
import struct
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/benchmarks'))
from vimeml.benchmarks.statistics import TEMPLATE, _trigrams
from vimeml.training.data import write_json
from prepare_standard_ime import normalized
from standard_ime import notice, track_files

V1 = ROOT / 'artifacts/benchmarks/ime-standard-ja-v1-draft'
CHAT = ROOT / 'artifacts/token-data/corpus-v41-chat-8k'
TOKENIZER = ROOT / 'artifacts/tokenizers/v4-a/ja-unigram-8192/tokenizer.model'
VERSION = 'ime-standard-ja-v2-blind'
SEED = 20261021
QUOTAS = {'wrime': 500, 'jmultiwoz': 250}
AUTHOR_CAP = 20


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fresh(path):
    if path.exists():
        raise ValueError(f'Preserve existing output: {path}; choose a fresh path.')
    path.mkdir(parents=True)


def chat_lines():
    import sentencepiece as spm
    processor = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    for split in ('train', 'validation'):
        tokens = (CHAT / f'{split}.tokens.bin').read_bytes()
        raw = (CHAT / f'{split}.offsets.bin').read_bytes()
        offsets = struct.unpack(f'<{len(raw) // 8}Q', raw)
        ids = struct.unpack(f'<{len(tokens) // 2}H', tokens)
        for start, end in zip(offsets, offsets[1:]):
            yield split, processor.decode(list(ids[start + 1:end - 1]))


def chat_screen(cases):
    """Exclude cases whose message is contained in, or a >=.85 near match of, a chat-bank sentence."""
    texts = [normalized(c['provenance']['cleaned_message']) for c in cases]
    lookup = collections.defaultdict(set)
    for i, text in enumerate(texts):
        if len(text) >= 12:
            lookup[text[:12]].add(i)
            lookup[text[-12:]].add(i)
    excluded, matches, lines = set(), [], 0
    for split, line in chat_lines():
        lines += 1
        text = normalized(line)
        for start in range(max(0, len(text) - 11)):
            for i in lookup.get(text[start:start + 12], ()):
                if i in excluded:
                    continue
                candidate = texts[i]
                ratio = 1.0 if candidate in text else difflib.SequenceMatcher(None, candidate, text, autojunk=False).ratio()
                if ratio >= .85:
                    excluded.add(i)
                    matches.append({'case_position': i, 'chat_split': split, 'ratio': round(ratio, 3)})
    return excluded, {'chat_bank': str(CHAT.relative_to(ROOT)), 'sentences': lines, 'excluded': len(excluded),
                      'matches': matches, 'method': '12-character anchors; normalized containment or SequenceMatcher >= .85',
                      'limits': 'Messages under 12 normalized characters are not screened; V4.3 expansion web shards are not available locally'}


def prepare(args):
    out = args.output
    fresh(out)
    pool = read(V1 / 'candidate-pool.json')
    v1_excluded = set(read(V1 / 'overlap-audit.json')['excluded_pool_positions'])
    used_docs, used_messages, used_queries = set(), set(), set()
    v1_groups = {}
    for role in ('development', 'blind'):
        rows = read(V1 / role / 'evaluation_items.json')
        v1_groups[role] = {r['provenance']['group_id'] for r in rows}
        for r in rows:
            p = r['provenance']
            used_docs.add((p['source'], p['document_id']))
            used_messages.add(normalized(p['cleaned_message']))
            used_queries.add((r['context_text'], r['input']))
    eligible, reasons = [], collections.Counter()
    for position, row in enumerate(pool):
        p = row['provenance']
        if p['role'] != 'blind':
            continue
        if position in v1_excluded:
            reasons['v1_training_overlap'] += 1
        elif (p['source'], p['document_id']) in used_docs:
            reasons['v1_selected_document'] += 1
        elif normalized(p['cleaned_message']) in used_messages or (row['context_text'], row['input']) in used_queries:
            reasons['v1_message_or_query'] += 1
        else:
            eligible.append(row)
    if {r['provenance']['group_id'] for r in eligible} & v1_groups['development']:
        raise ValueError('A blind-role case belongs to a v1 development writer/user.')
    excluded, chat_report = chat_screen(eligible)
    reasons['chat_bank_overlap'] = len(excluded)
    eligible = [r for i, r in enumerate(eligible) if i not in excluded]
    rng = random.Random(SEED)
    selected = []
    for source, target in QUOTAS.items():
        buckets = collections.defaultdict(list)
        for row in eligible:
            if row['provenance']['source'] == source:
                length = 'short' if len(row['input']) <= 16 else 'medium' if len(row['input']) <= 32 else 'long'
                buckets[(bool(row['context_text']), length)].append(row)
        for key in sorted(buckets):
            rng.shuffle(buckets[key])
        authors, grams, chosen, seen = collections.Counter(), [], [], set()
        while len(chosen) < target and any(buckets.values()):
            for key in sorted(buckets):
                if not buckets[key]:
                    continue
                row = copy.deepcopy(buckets[key].pop())
                p = row['provenance']
                signature = normalized(p['cleaned_message']), (row['context_text'], row['input'])
                gram = _trigrams(row['expected_output'][0])
                if authors[p['author_id']] >= AUTHOR_CAP or signature[0] in seen or signature[1] in seen:
                    reasons['author_cap_or_duplicate'] += 1
                    continue
                if any(len(gram & g) >= TEMPLATE * len(gram | g) for g in grams):
                    reasons['template_family'] += 1
                    continue
                row['index'] = f'ime-standard-v2-blind-{source}-{len(chosen) + 1:04}'
                p.update(role='blind', release=VERSION, v1_pool_role='blind_unselected')
                chosen.append(row)
                authors[p['author_id']] += 1
                grams.append(gram)
                seen.update(signature)
                if len(chosen) == target:
                    break
        if len(chosen) != target:
            raise ValueError(f'{source}: only {len(chosen)} eligible cases; no padding or silent quota change.')
        selected.extend(chosen)
    directory = out / 'blind'
    directory.mkdir()
    track_files(directory, selected)
    stats = read(directory / 'stats.json')
    stats.update(sources=dict(collections.Counter(r['provenance']['source'] for r in selected)),
                 unique_authors=len({r['provenance']['author_id'] for r in selected}),
                 context_reading_strata=dict(collections.Counter(
                     f"{'with' if r['context_text'] else 'without'}-context/"
                     f"{'short' if len(r['input']) <= 16 else 'medium' if len(r['input']) <= 32 else 'long'}" for r in selected)))
    write_json(directory / 'stats.json', stats)
    write_json(out / 'chat-overlap-audit.json', chat_report)
    v1_manifest = read(V1 / 'manifest.json')
    manifest = {**{k: v1_manifest[k] for k in ('format', 'primary_tracks', 'aggregate_policy', 'score_policy',
                                                 'blind_policy', 'distribution_policy', 'source_revisions', 'licenses')},
                'version': f'{VERSION}-draft', 'status': 'awaiting_label_review_and_candidate_export',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
                'parent_pool': 'ime-standard-ja-v1-draft/candidate-pool.json (blind-role, unselected)',
                'splits': {'blind': len(selected)}, 'labels_formal_gold': False, 'models_scored': False,
                'blind_lm_scored': False, 'candidate_generation_uses_model_scores': False, 'native_review_completed': False,
                'split_policy': 'Writers/users disjoint from v1 development; same blind-role writers/users as the consumed v1 blind, '
                                'but no v1 document, message or context/reading pair; wizard-disjointness not enforced',
                'selection_policy': f'Seed {SEED}; v1 strata; <= {AUTHOR_CAP} per writer/user; target trigram Jaccard < {TEMPLATE}',
                'exclusions': dict(reasons)}
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({'selected': len(selected), 'eligible_after_screen': len(eligible), 'exclusions': reasons,
                      'authors': stats['unique_authors']}, ensure_ascii=False, indent=1))


PAIRS = [('分か', 'わか'), ('可愛い', 'かわいい'), ('可愛く', 'かわいく'), ('下さい', 'ください'), ('出来', 'でき'),
         ('良い', 'よい'), ('良く', 'よく'), ('ご飯', 'ごはん'), ('眼鏡', 'めがね', 'メガネ'), ('子供', '子ども'),
         ('頑張', 'がんば'), ('事', 'こと'), ('為', 'ため'), ('無い', 'ない'), ('無く', 'なく'),
         ('買い取り', '買取', '買取り'), ('かもしれ', 'かも知れ'), ('我々', 'われわれ'), ('言え', 'いえ'),
         ('有り', 'あり'), ('無理', 'むり')]


def review_pack(args):
    """Review sheet in the v1 format: no model output, no candidate pool or order."""
    from vimeml.benchmarks.readings import DualReading
    reader = DualReading()
    rows = read(args.prepared / 'blind/evaluation_items.json')
    if args.output.exists():
        raise ValueError(f'Preserve existing review sheet: {args.output}')
    lines = []
    for row in rows:
        target = row['expected_output'][0]
        forms = {target}
        for group in PAIRS:
            for form in sorted(forms)[:16]:
                for current in group:
                    if current in form:
                        forms.update(form.replace(current, alt) for alt in group)
            forms = set(sorted(forms)[:32]) | {target}
        suggestions = []
        for form in sorted(forms - {target})[:32]:
            check = reader.analyse(form)
            if not check['unknown'] and check['unidic'] == row['input'] and check['sudachi'] == row['input']:
                suggestions.append(form)
        lines.append({'id': row['index'], 'source': row['provenance']['source'], 'left_context': row['context_text'],
                      'target': target, 'proposed_reading': row['input'], 'accepted_answers': [target],
                      'original_message': row['provenance']['cleaned_message'], 'spelling_suggestions': suggestions[:15],
                      'decision': 'pending', 'reviewer': '', 'review_type': 'ai_expert',
                      'native_japanese_review': False, 'notes': ''})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in lines), encoding='utf-8')


def freeze(args):
    fresh(args.output)
    manifest = read(args.prepared / 'manifest.json')
    original = read(args.prepared / 'blind/evaluation_items.json')
    reviews = [json.loads(line) for line in args.reviews.read_text(encoding='utf-8').splitlines() if line.strip()]
    by_id = {r['id']: r for r in reviews}
    if len(by_id) != len(reviews) or set(by_id) != {r['index'] for r in original}:
        raise ValueError('Review IDs do not match the prepared draft.')
    selected, quarantine = [], []
    for row in original:
        review = by_id[row['index']]
        if review['left_context'] != row['context_text'] or review['target'] != row['expected_output'][0]:
            raise ValueError('Review context/target changed; prepare a new version.')
        if not review['reviewer'].strip() or not review['notes'].strip():
            raise ValueError(f'Review not complete: {row["index"]}')
        if review['review_type'] != 'ai_expert' or review['native_japanese_review'] is not False:
            raise ValueError('AI expert review must be identified explicitly, without a native-human claim.')
        if review['decision'] == 'exclude':
            quarantine.append({'case': row, 'review': review})
            continue
        if review['decision'] != 'accept':
            raise ValueError(f'Pending review: {row["index"]}')
        answers = review['accepted_answers']
        if not answers or len(set(answers)) != len(answers) or any(not isinstance(a, str) or not a.strip() for a in answers):
            raise ValueError('Accepted forms must be a nonempty unique string list.')
        if not re.fullmatch(r'[ァ-ヺー。、！？「」『』（）・：，,.!?]{4,64}', review['proposed_reading']):
            raise ValueError('Reviewed input must follow the 4-64-character orthographic katakana protocol.')
        frozen = copy.deepcopy(row)
        frozen['input'] = review['proposed_reading']
        frozen['expected_output'] = answers
        frozen['review'] = {'status': 'ai_expert_reviewed', 'human_native_review': False, 'not_formal_gold': True,
                            'reviewer': review['reviewer'], 'notes': review['notes'], 'acceptable_forms_exhaustive': False}
        selected.append(frozen)
    if {r['provenance']['source'] for r in selected} != set(QUOTAS):
        raise ValueError('Both sources must remain.')
    dev = read(ROOT / 'artifacts/benchmarks/ime-standard-ja-v1-ai-expert-r1/development/evaluation_items.json')
    if {(r['context_text'], r['input']) for r in dev} & {(r['context_text'], r['input']) for r in selected}:
        raise ValueError('Reviewed blind input/context pairs overlap v1 development.')
    track_files(args.output / 'blind', selected)
    changed = sum(by_id[r['index']]['proposed_reading'] != r['input'] for r in original)
    multiple = sum(len(r['expected_output']) > 1 for r in selected)
    manifest.update(version=VERSION, release_id=uuid.uuid4().hex, status='labels_frozen_awaiting_candidates',
                    frozen_utc=datetime.now(timezone.utc).isoformat(), splits={'blind': len(selected)},
                    labels_frozen=True, review_completed=True, label_quality='ai_expert_reviewed',
                    review_policy='Frozen pre-score AI expert references; not human native gold',
                    label_change_policy='Immutable release; corrections require a new release ID and an explicit changelog',
                    change_note=f'{len(original)} cases reviewed before any candidate export or scoring; {len(quarantine)} excluded, '
                                f'{changed} readings corrected, {multiple} with multiple acceptable forms')
    write_json(args.output / 'manifest.json', manifest)
    write_json(args.output / 'review-decisions.json', reviews)
    write_json(args.output / 'quarantine.json', quarantine)
    for name in ('chat-overlap-audit.json',):
        write_json(args.output / name, read(args.prepared / name))
    notice(args.output)
    print(manifest['change_note'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest='command', required=True)
    sub = subs.add_parser('prepare')
    sub.add_argument('--output', type=Path, default=ROOT / f'artifacts/benchmarks/{VERSION}-draft')
    sub = subs.add_parser('review-pack')
    sub.add_argument('--prepared', type=Path, default=ROOT / f'artifacts/benchmarks/{VERSION}-draft')
    sub.add_argument('--output', type=Path, default=ROOT / f'artifacts/benchmarks/{VERSION}-review/blind-review.jsonl')
    sub = subs.add_parser('freeze')
    sub.add_argument('--prepared', type=Path, default=ROOT / f'artifacts/benchmarks/{VERSION}-draft')
    sub.add_argument('--reviews', type=Path, required=True)
    sub.add_argument('--output', type=Path, default=ROOT / f'artifacts/benchmarks/{VERSION}-ai-expert')
    args = parser.parse_args()
    {'prepare': prepare, 'review-pack': review_pack, 'freeze': freeze}[args.command](args)


if __name__ == '__main__':
    main()
