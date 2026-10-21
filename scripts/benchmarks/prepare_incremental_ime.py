"""Prototype segmented (incremental-input) items from Standard v1 development messages.

A real IME user converts a message phrase by phrase, and each conversion sees the text
already committed. This prototype cuts each v1 development message into consecutive
conversion segments at word boundaries (the Standard extraction rule: UniDic/Sudachi
boundary, not inside a compound, not starting with a function or auxiliary-like word),
4-64 kana each, readings agreed by both dictionaries standalone and in context. Text
outside Japanese runs (digits, Latin, emoji) is committed literally.

Every segment item carries the gold committed prefix as left context (teacher-forced).
Candidate pools are exported once; free-running evaluation then replaces the context
by the model's own earlier choices, which is valid only if the n-best pool does not
depend on left context (checked by the blank-context development export).

Labels are dictionary-derived drafts on already exposed development messages: a
protocol prototype, never a capability claim.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/benchmarks'))
from vimeml.training.data import write_json
from prepare_standard_ime import RUN
from prepare_standard_ime_v2 import read, fresh
from standard_ime import track_files

VERSION = 'ime-incremental-ja-proto1'
SOURCE = ROOT / 'artifacts/benchmarks/ime-standard-ja-v1-ai-expert-r1/development/evaluation_items.json'
TARGET = 8
FUNCTION = ('助詞', '助動詞', '接尾辞', '補助記号')


def cuts(reader, run):
    """Allowed segment starts inside one Japanese run."""
    words = list(reader.mecab(run))
    sudachi = {w.begin() for w in reader.sudachi.tokenize(run, reader.mode)}
    allowed, offset = set(), 0
    for i, word in enumerate(words):
        f, previous = word.feature, words[i - 1].feature if i else None
        compound = previous is not None and (previous.pos1 == '接頭辞' or previous.pos1 == '名詞' and f.pos1 == '名詞')
        after_mark = previous is not None and previous.pos1 == '補助記号'
        if offset in sudachi and (after_mark or not compound and f.pos1 not in FUNCTION and f.pos2 != '非自立可能'):
            allowed.add(offset)
        offset += len(word.surface)
    return sorted(allowed - {0})


def segment_run(reader, run):
    """Greedy segments of about TARGET characters; None when a reading is not agreed."""
    bounds, start = [], 0
    for cut in cuts(reader, run):
        if cut - start >= TARGET:
            bounds.append((start, cut))
            start = cut
    bounds.append((start, len(run)))
    readings = []
    for i, (a, b) in enumerate(bounds):
        standalone, contextual = reader.analyse(run[a:b]), reader.analyse(run[:b], a)
        if (standalone['unknown'] or contextual['unknown'] or not standalone['agreement'] or not standalone['kana_only']
                or standalone['unidic'] != contextual['unidic'] or standalone['sudachi'] != contextual['sudachi']):
            return None
        readings.append(standalone['unidic'])
    while len(bounds) > 1 and len(readings[-1]) < 4:
        (a, _), (_, b) = bounds[-2], bounds[-1]
        merged = reader.analyse(run[:b], a)
        if merged['unknown'] or merged['unidic'] != readings[-2] + readings[-1]:
            return None
        bounds[-2:], readings[-2:] = [(a, b)], [merged['unidic']]
    if any(not 4 <= len(r) <= 64 for r in readings):
        return None
    return list(zip(bounds, readings))


def prepare(args):
    from vimeml.benchmarks.readings import DualReading
    reader = DualReading()
    fresh(args.output)
    items, rejects, messages = [], collections.Counter(), 0
    for row in read(SOURCE):
        p = row['provenance']
        text = p['cleaned_message']
        segments = []
        for run in RUN.finditer(text):
            if len(run.group()) < 4:
                continue
            parts = segment_run(reader, run.group())
            if parts is None:
                segments = None
                break
            segments += [(run.start() + a, run.start() + b, reading) for (a, b), reading in parts]
        if not segments:
            rejects['unsegmentable_message' if segments is None else 'no_japanese_run'] += 1
            continue
        messages += 1
        for n, (start, end, reading) in enumerate(segments, 1):
            items.append({'index': f'{VERSION}-{messages:04}-{n:02}', 'context_text': text[:start], 'input': reading,
                          'expected_output': [text[start:end]], 'category': row['category'],
                          'provenance': {**{k: p[k] for k in ('source', 'author_id', 'group_id', 'document_id')},
                                         'parent_case': row['index'], 'message': messages, 'segment': n,
                                         'segments': len(segments), 'span_start': start, 'span_end': end,
                                         'cleaned_message': text},
                          'review': {'status': 'dictionary_draft', 'human_native_review': False, 'not_formal_gold': True,
                                     'acceptable_forms_exhaustive': False}})
    track_files(args.output / 'segmented', items)
    stats = read(args.output / 'segmented/stats.json')
    lengths = collections.Counter(min(i['provenance']['segments'], 6) for i in items if i['provenance']['segment'] == 1)
    stats.update(messages=messages, segments_per_message={('6+' if k == 6 else str(k)): v for k, v in sorted(lengths.items())},
                 rejections=dict(rejects))
    write_json(args.output / 'segmented/stats.json', stats)
    write_json(args.output / 'manifest.json', {
        'version': VERSION, 'status': 'prototype_awaiting_candidate_export', 'parent': str(SOURCE.relative_to(ROOT)),
        'splits': {'segmented': len(items)}, 'messages': messages, 'labels_formal_gold': False,
        'label_quality': 'dictionary_draft', 'exposure': 'Standard v1 development messages (exposed)',
        'segmentation': f'Greedy word-boundary cuts after >= {TARGET} characters; final segments under 4 kana merged',
        'evaluation_modes': {'teacher_forced': 'gold committed prefix as context (as exported)',
                             'free_running': "model's earlier choices replace the gold prefix; requires context-independent pools"}})
    print(json.dumps({'messages': messages, 'items': len(items), 'rejections': rejects, 'stats': stats}, ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=ROOT / f'artifacts/benchmarks/{VERSION}')
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
