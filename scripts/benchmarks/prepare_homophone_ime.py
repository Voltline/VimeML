"""Written-text homophone challenge track from one pinned Japanese Wikipedia shard.

Standard IME targets are conversational, and the error analysis left few semantic
errors there; the AJIMEE gap is encyclopedic homophones. This track tests that gap
directly, with no model involved in selection:

- lexicon: UniDic lemmas (common nouns, verbs, adjectives) sharing reading, part of
  speech and conjugation type, each lemma >= 2 characters and >= MIN_COUNT tokens in
  the shard. Equal UniDic lemma means a notation variant (聞く/聴く), so groups contain
  only different words, consistent with vimeml.benchmarks.notation;
- spans: a 6-20 character conversion span containing one group word at word
  boundaries, with paragraph left context, readings agreed by UniDic and Sudachi
  standalone and in context (the Standard extraction rule);
- pair design: for each sampled group, one item whose reference is the most frequent
  lemma and one whose reference is the second, from different articles, so always
  choosing the frequent word scores about half;
- screens: sentence containment/near match against the local training text (Standard
  v1 audit) and any 12-character overlap with AJIMEE.

Commands: download (pinned shard), prepare (unscored draft), review-pack (AI review
sheet), freeze (AI expert labels, never native gold).
"""
import argparse
import collections
import copy
import json
import random
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/benchmarks'))
from vimeml.training.data import write_json
from prepare_standard_ime import RUN, KANJI, normalized, overlap_audit
from prepare_standard_ime_v2 import PAIRS, read, fresh
from standard_ime import track_files

VERSION = 'ime-homophone-ja-v1'
SEED = 20261021
REPO, REVISION = 'wikimedia/wikipedia', 'b04c8d1ceb2f5cd4588862100d08de323dccfbaa'
SHARD = '20231101.ja/train-00014-of-00015.parquet'
SOURCE = ROOT / f'datasets/{VERSION}/jawiki'
GROUPS = 320
MIN_COUNT = 30
PER_LEMMA = 3
CONTEXT = 200
CONTENT = ('名詞', '動詞', '形容詞')
FUNCTION = ('助詞', '助動詞', '接尾辞', '補助記号')
LATIN = re.compile(r'[A-Za-z0-9Ａ-Ｚａ-ｚ０-９]')


def download(args):
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(REPO, SHARD, repo_type='dataset', revision=REVISION, local_dir=SOURCE))
    write_json(SOURCE / 'download-manifest.json', {
        'repository': f'https://huggingface.co/datasets/{REPO}', 'revision': REVISION, 'file': SHARD,
        'bytes': path.stat().st_size, 'license': 'CC-BY-SA-4.0 (Wikipedia text); GFDL dual-licensed upstream',
        'purpose': 'Evaluation-only; never a training input',
        'shard_choice': 'Last shard of the 20231101 dump (latest article IDs)'})
    print(path)


def lines():
    import pyarrow.parquet as pq
    table = pq.read_table(SOURCE / SHARD, columns=['id', 'url', 'title', 'text'])
    for row in table.to_pylist():
        for number, line in enumerate(row['text'].split('\n')):
            if len(line) >= 30 and not line[0].isspace() and len(LATIN.findall(line)) <= .2 * len(line):
                yield row, number, line


def homophone_key(word):
    """(reading, POS, conjugation type), lemma for a content word that can head a group."""
    f = word.feature
    if (word.is_unk or f.pos1 not in CONTENT or f.pos2 in ('固有名詞', '数詞', '非自立可能') or not f.lemma
            or '-' in f.lemma or len(f.lemma) < 2 or not KANJI.search(f.lemma) or not KANJI.search(word.surface)
            or not f.lForm or len(f.lForm) < 2 or f.pos1 == '名詞' and f.kana != f.lForm):
        return None, None
    return (f.lForm, f.pos1, f.cType), f.lemma


def lexicon(reader):
    counts = collections.Counter()
    for _, _, line in lines():
        for word in reader.mecab(line):
            key, lemma = homophone_key(word)
            if key:
                counts[key, lemma] += 1
    groups = collections.defaultdict(dict)
    for (key, lemma), n in counts.items():
        if n >= MIN_COUNT:
            groups[key][lemma] = n
    return {'|'.join(key): dict(sorted(members.items(), key=lambda x: -x[1]))
            for key, members in sorted(groups.items()) if len(members) >= 2}


def boundaries(reader, line, words):
    sudachi = {w.begin() for w in reader.sudachi.tokenize(line, reader.mode)} | {len(line)}
    starts, offset = [], 0
    for i, word in enumerate(words):
        previous = words[i - 1].feature if i else None
        compound = previous is not None and (previous.pos1 == '接頭辞' or previous.pos1 == '名詞' and word.feature.pos1 == '名詞')
        starts.append((offset, offset in sudachi and not compound and word.feature.pos1 not in FUNCTION
                       and word.feature.pos2 != '非自立可能'))
        offset += len(word.surface)
    return starts, sudachi


def spans(reader, line, words, h, rng):
    """Seeded word-boundary span [start, end) of 6-20 characters around word h."""
    starts, sudachi = boundaries(reader, line, words)
    ends = []
    for j in range(h, len(words)):
        if words[j].feature.pos1 == '補助記号':
            break
        end = starts[j][0] + len(words[j].surface)
        if j + 1 == len(words) or words[j + 1].feature.pos1 == '補助記号' or starts[j + 1][1]:
            ends.append(end)
    options = []
    for s in range(max(0, h - 3), h + 1):
        if not starts[s][1] or any(w.feature.pos1 == '補助記号' for w in words[s:h]):
            continue
        options += [(starts[s][0], e) for e in ends if 6 <= e - starts[s][0] <= 20]
    rng.shuffle(options)
    return options


def sentence(line, start, end):
    left = line.rfind('。', 0, start) + 1
    right = line.find('。', end)
    return line[left:len(line) if right < 0 else right + 1]


def candidates(reader, groups, rng):
    """Up to PER_LEMMA seeded reservoir occurrences per (group, lemma), then extraction checks."""
    members = {(key, lemma) for key, m in groups.items() for lemma in m}
    seen, reservoir = collections.Counter(), collections.defaultdict(list)
    for row, number, line in lines():
        for h, word in enumerate(reader.mecab(line)):
            key, lemma = homophone_key(word)
            if key is None or ('|'.join(key), lemma) not in members:
                continue
            slot = '|'.join(key), lemma
            seen[slot] += 1
            sample = (row['id'], number, h)
            if len(reservoir[slot]) < 4 * PER_LEMMA:
                reservoir[slot].append(sample)
            else:
                k = rng.randrange(seen[slot])
                if k < 4 * PER_LEMMA:
                    reservoir[slot][k] = sample
    wanted = collections.defaultdict(list)
    for slot, samples in reservoir.items():
        for a, n, h in samples:
            wanted[a, n].append((slot, h))
    found, rejects = collections.defaultdict(list), collections.Counter()
    for row, number, line in lines():
        if (row['id'], number) not in wanted:
            continue
        words = list(reader.mecab(line))
        for slot, h in wanted[row['id'], number]:
            if len(found[slot]) >= PER_LEMMA:
                continue
            case, reason = extract(reader, row, number, line, words, h, slot, groups, rng)
            if case is None:
                rejects[reason] += 1
            else:
                found[slot].append(case)
    return found, rejects


def extract(reader, row, number, line, words, h, slot, groups, rng):
    key, lemma = slot
    reason = 'no_agreed_span'
    for start, end in spans(reader, line, words, h, rng)[:8]:
        target = line[start:end]
        context = line[:start]
        if not RUN.fullmatch(target) or not KANJI.search(target):
            continue
        if len(context) < 10:
            reason = 'short_context'
            continue
        if len(context) > CONTEXT:
            cut = context.find('。', len(context) - CONTEXT)
            context = context[cut + 1:] if 0 <= cut < len(context) - 10 else context[-CONTEXT:]
        standalone = reader.analyse(target)
        contextual = reader.analyse(line[:end], start)
        if (standalone['unknown'] or contextual['unknown'] or not standalone['agreement'] or not standalone['kana_only']
                or standalone['unidic'] != contextual['unidic'] or standalone['sudachi'] != contextual['sudachi']):
            continue
        reading, pos, ctype = key.split('|')
        return {'index': 'pending', 'context_text': context, 'input': standalone['unidic'], 'expected_output': [target],
                'category': 'jawiki-homophone',
                'provenance': {'source': 'jawiki', 'source_doc_id': row['id'], 'title': row['title'], 'url': row['url'],
                               'line': number, 'span_start': start, 'span_end': end,
                               'cleaned_message': sentence(line, start, end),
                               'homophone': {'surface': words[h].surface, 'lemma': lemma, 'reading': reading, 'pos': pos,
                                             'conjugation': ctype, 'competitors': [m for m in groups[key] if m != lemma],
                                             'group_counts': groups[key]},
                               'dataset': f'{REPO}@{REVISION}/{SHARD}', 'license': 'CC-BY-SA-4.0'},
                'review': {'status': 'pending', 'human_native_review': False, 'not_formal_gold': True,
                           'reading_evidence': 'Two orthographic dictionaries agree; review required',
                           'acceptable_forms_exhaustive': False}}, None
    return None, reason


def ajimee_screen(pool):
    """Pool positions sharing any normalized 12-character substring with an AJIMEE sentence."""
    def grams(text):
        return {text[i:i + 12] for i in range(max(0, len(text) - 11))}
    seen = set()
    for row in read(ROOT / 'artifacts/benchmarks/ajimee-jwtd-v2-v1/evaluation_items.json'):
        seen |= grams(normalized(row['original_text']))
    return {i for i, r in enumerate(pool) if grams(normalized(r['provenance']['cleaned_message'])) & seen}


def prepare(args):
    from vimeml.benchmarks.readings import DualReading
    out = args.output
    fresh(out)
    reader = DualReading()
    rng = random.Random(SEED)
    groups = lexicon(reader)
    write_json(out / 'lexicon.json', groups)
    print(f'{len(groups)} homophone groups', flush=True)
    found, rejects = candidates(reader, groups, rng)
    pool = [case for slot in sorted(found) for case in found[slot]]
    for case in pool:
        case['provenance']['document_id'] = case['provenance']['source_doc_id']
    training = overlap_audit(pool, out)
    ajimee = ajimee_screen(pool)
    reasons = collections.Counter(extraction=sum(rejects.values()), training_overlap=len(training), ajimee_overlap=len(ajimee))
    by_slot = collections.defaultdict(list)
    for i, case in enumerate(pool):
        if i not in training and i not in ajimee:
            h = case['provenance']['homophone']
            by_slot['|'.join((h['reading'], h['pos'], h['conjugation'])), h['lemma']].append(case)
    order = sorted(groups)
    rng.shuffle(order)
    selected, articles, used_groups = [], set(), 0
    for key in order:
        if used_groups == GROUPS:
            break
        pair = list(groups[key])[:2]
        picks = []
        for lemma in pair:
            pick = next((c for c in by_slot[key, lemma] if c['provenance']['source_doc_id'] not in articles
                         and all(c['provenance']['source_doc_id'] != p['provenance']['source_doc_id'] for p in picks)), None)
            if pick is None:
                break
            picks.append(pick)
        if len(picks) < 2:
            reasons['group_without_both_lemmas'] += 1
            continue
        used_groups += 1
        for rank, case in enumerate(picks):
            case = copy.deepcopy(case)
            case['index'] = f'{VERSION}-{len(selected) + 1:04}'
            case['provenance'].update(group_rank=rank, group_id=f'homophone:{key}', release=VERSION)
            selected.append(case)
            articles.add(case['provenance']['source_doc_id'])
    if used_groups != GROUPS:
        raise ValueError(f'Only {used_groups} complete groups; no padding or silent quota change.')
    track_files(out / 'challenge', selected)
    stats = read(out / 'challenge/stats.json')
    stats.update(groups=used_groups, pos=dict(collections.Counter(r['provenance']['homophone']['pos'] for r in selected)),
                 target_length=dict(collections.Counter('6-10' if len(r['expected_output'][0]) <= 10 else '11-20' for r in selected)))
    write_json(out / 'challenge/stats.json', stats)
    write_json(out / 'manifest.json', {
        'format': 'vimeml_standard_ime_preparation_v1', 'version': f'{VERSION}-draft',
        'status': 'awaiting_label_review_and_candidate_export', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'seed': SEED, 'source': {'dataset': f'https://huggingface.co/datasets/{REPO}', 'revision': REVISION, 'file': SHARD,
                                 'license': 'CC-BY-SA-4.0'},
        'splits': {'challenge': len(selected)}, 'labels_formal_gold': False, 'models_scored': False,
        'candidate_generation_uses_model_scores': False, 'native_review_completed': False,
        'lexicon_policy': f'UniDic lemma groups by (lForm, POS, conjugation); lemmas >= 2 characters, >= {MIN_COUNT} tokens in the shard; '
                          'no proper nouns, numerals or auxiliary-like verbs; nouns read without rendaku',
        'selection_policy': f'Seed {SEED}; {GROUPS} uniformly sampled groups x two most frequent lemmas; one item per article; '
                            f'{PER_LEMMA} extracted candidates per lemma from a seeded reservoir',
        'exclusions': dict(reasons), 'extraction_rejections': dict(rejects),
        'contamination': 'Wikipedia is not a direct training source, but web crawls may contain copies; '
                         'screened only against the local training text, not the V4.3 expansion shards'})
    print(json.dumps({'selected': len(selected), 'pool': len(pool), 'exclusions': reasons, 'stats': stats}, ensure_ascii=False, indent=1))


def review_pack(args):
    """AI review sheet in the Standard format plus the homophone group; no model output or candidates."""
    from vimeml.benchmarks.readings import DualReading
    reader = DualReading()
    rows = read(args.prepared / 'challenge/evaluation_items.json')
    if args.output.exists():
        raise ValueError(f'Preserve existing review sheet: {args.output}')
    sheet = []
    for row in rows:
        target = row['expected_output'][0]
        forms = {target}
        for group in PAIRS:
            for current in group:
                if current in target:
                    forms.update(target.replace(current, alt) for alt in group)
        suggestions = [f for f in sorted(forms - {target}) if (lambda c: not c['unknown'] and c['unidic'] == row['input']
                                                               and c['sudachi'] == row['input'])(reader.analyse(f))]
        h = row['provenance']['homophone']
        sheet.append({'id': row['index'], 'source': 'jawiki', 'left_context': row['context_text'], 'target': target,
                      'proposed_reading': row['input'], 'accepted_answers': [target],
                      'original_message': row['provenance']['cleaned_message'], 'title': row['provenance']['title'],
                      'homophone': {k: h[k] for k in ('surface', 'lemma', 'reading', 'competitors')},
                      'spelling_suggestions': suggestions, 'decision': 'pending', 'reviewer': '',
                      'review_type': 'ai_expert', 'native_japanese_review': False, 'notes': ''})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in sheet), encoding='utf-8')


def notice(output):
    (output / 'NOTICE.md').write_text("""# Source attribution and use scope

Local research benchmark. Conversion spans, readings and left contexts are
derived from Japanese Wikipedia (dump 20231101, via the Hugging Face dataset
wikimedia/wikipedia at a pinned revision; see manifest.json). Wikipedia text is
licensed CC BY-SA 4.0; adapted material keeps that license. Each case records the
source article title and URL in provenance for attribution.

Selection used dictionaries and corpus counts only: no LM scores, candidate pools
or model outputs. Labels are AI expert reviewed, not human native gold.
""", encoding='utf-8')


def freeze(args):
    fresh(args.output)
    manifest = read(args.prepared / 'manifest.json')
    original = read(args.prepared / 'challenge/evaluation_items.json')
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
    track_files(args.output / 'challenge', selected)
    changed = sum(by_id[r['index']]['proposed_reading'] != r['input'] for r in original)
    multiple = sum(len(r['expected_output']) > 1 for r in selected)
    complete = collections.Counter(r['provenance']['group_id'] for r in selected)
    manifest.update(version=VERSION, release_id=uuid.uuid4().hex, status='labels_frozen_awaiting_candidates',
                    frozen_utc=datetime.now(timezone.utc).isoformat(), splits={'challenge': len(selected)},
                    labels_frozen=True, review_completed=True, label_quality='ai_expert_reviewed',
                    review_policy='Frozen pre-score AI expert references; not human native gold',
                    label_change_policy='Immutable release; corrections require a new release ID and an explicit changelog',
                    complete_pairs=sum(n == 2 for n in complete.values()),
                    change_note=f'{len(original)} cases reviewed before any candidate export or scoring; {len(quarantine)} excluded, '
                                f'{changed} readings corrected, {multiple} with multiple acceptable forms')
    write_json(args.output / 'manifest.json', manifest)
    write_json(args.output / 'review-decisions.json', reviews)
    write_json(args.output / 'quarantine.json', quarantine)
    write_json(args.output / 'overlap-audit.json', read(args.prepared / 'overlap-audit.json'))
    notice(args.output)
    print(manifest['change_note'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest='command', required=True)
    subs.add_parser('download')
    sub = subs.add_parser('prepare')
    sub.add_argument('--output', type=Path, default=ROOT / f'artifacts/benchmarks/{VERSION}-draft')
    sub = subs.add_parser('review-pack')
    sub.add_argument('--prepared', type=Path, default=ROOT / f'artifacts/benchmarks/{VERSION}-draft')
    sub.add_argument('--output', type=Path, default=ROOT / f'artifacts/benchmarks/{VERSION}-review/challenge-review.jsonl')
    sub = subs.add_parser('freeze')
    sub.add_argument('--prepared', type=Path, default=ROOT / f'artifacts/benchmarks/{VERSION}-draft')
    sub.add_argument('--reviews', type=Path, required=True)
    sub.add_argument('--output', type=Path, default=ROOT / f'artifacts/benchmarks/{VERSION}-ai-expert')
    args = parser.parse_args()
    {'download': download, 'prepare': prepare, 'review-pack': review_pack, 'freeze': freeze}[args.command](args)


if __name__ == '__main__':
    main()
