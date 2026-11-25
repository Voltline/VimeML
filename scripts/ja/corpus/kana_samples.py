"""ja-gen stage 1.3: text-level reading samples (span / sentence) from the ja-gen corpora.

Sources: `web` (FineWeb-2 export ja-gen-export-v1, one paragraph = one line of a document),
`lm` (the V4 LM corpus outputs/corpus-fast-v1, one line) and `chat` (the V4.1 chat bank, one
utterance). A paragraph is selected with the source's fraction (doc_hash(label/select, key));
every selected paragraph draws one kind, span or sentence (configs/ja/data-ja-kana.toml),
from random.Random(doc_hash(label, key)), so a rebuild is byte-identical.

Readings (deviation D4): a typeable run is analysed by UniDic (unidic-lite) and Sudachi (mode C)
and cut at the character boundaries the two share. A segment has a reading when its two
normalized readings agree (Sudachi's reading is kept, ヂ/ヅ included) or when its (surface,
UniDic, Sudachi) triple is in configs/ja/ja-reading-table.tsv. Five agreed readings users rarely
type are replaced (AGREED, deviation D5: 私, 日本, 明日, 言う, 何). A segment holding an unknown word
with kanji, or a reading outside kana and Japanese punctuation, has none. A target takes only
segments with readings. Readings are written in hiragana with punctuation as typed (、。！？「」).

  span      a typeable run (Standard rule, 6-100 characters, with kanji); the target starts at
            a word that may begin a conversion (not a particle, auxiliary, suffix or symbol,
            not after a prefix, not inside a noun-noun compound), ends at the run end or before
            a later such word; context = the paragraph before it, last 64 characters.
  sentence  a whole typeable sentence with kanji or katakana that starts at such a word, no
            context.

  build   jobs -> <output>/parts/<job>.parquet + <job>.json (skips finished jobs; --only, --exclude
          and --sources split the work between machines; the Mac runs web jobs only)
  merge   checks that every job finished and writes manifest.json (counts, stats, part hashes)

    python -X utf8 scripts/ja/corpus/kana_samples.py build [--sources web] [--only|--exclude NAMES] [--limit N]
    python -X utf8 scripts/ja/corpus/kana_samples.py merge
"""
import argparse
import collections
import hashlib
import json
import random
import re
import sys
import time
import tomllib
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/ja/corpus'))
from vimeml.benchmarks.readings import KANA, DualReading, katakana  # noqa: E402
from reading_pairs import load_table, norm  # noqa: E402

FORMAT = 'vimeml_ja_kana_samples_v1'
RUN = re.compile(r'[ぁ-ゖァ-ヺー一-龠々〆ヵヶ、。！？「」『』（）・…〜～]+')
KANJI = re.compile(r'[一-龠々]')
CONVERT = re.compile(r'[一-龠々ァ-ヺ]')
CONTROLS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
SPLIT = re.compile(r'(?<=[。！？!?])')
NO_START = ('助詞', '助動詞', '接尾辞', '補助記号')
# Deviation D5: readings both dictionaries agree on but users rarely type, for a segment that is
# exactly this word: (surface, agreed reading) -> (typed reading, characters after which the
# agreed reading stays). 私共 stays ワタクシドモ; 何 stays ナン before だ/で/と/の/て/な/じ.
AGREED = {('私', 'ワタクシ'): ('ワタシ', ('共', 'ども')),
          ('日本', 'ニッポン'): ('ニホン', ()),
          ('明日', 'アス'): ('アシタ', ()),
          ('言う', 'ユウ'): ('イウ', ()),
          ('何', 'ナン'): ('ナニ', tuple('だでとのてなじ'))}
COLUMNS = ('source', 'key', 'kind', 'context', 'reading', 'output')
SPLITS = ('train', 'validation')

_state = {}


def doc_hash(label, key):
    return int.from_bytes(hashlib.sha256(f'{label}\0{key}'.encode('utf-8')).digest()[:8], 'big')


def clean(text):
    return CONTROLS.sub('', unicodedata.normalize('NFC', text)).strip()


def hiragana(text):
    return ''.join(chr(ord(c) - 0x60) if 'ァ' <= c <= 'ヶ' else c for c in text)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- readings

def analyse(text):
    """[(start, end, reading katakana or None, may_start)] over boundaries shared by both dictionaries."""
    reader, table = _state['reader'], _state['table']
    uni, offset = [], 0
    for w in reader.mecab(text):
        s = w.surface
        start = text.index(s, offset) if s else offset
        k = katakana(s)
        reading = k if KANA.fullmatch(k) else (w.feature.kana or s)
        unknown = bool(w.is_unk and KANJI.search(s))
        uni.append((start, start + len(s), reading, w.feature.pos1, unknown))
        offset = start + len(s)
    sud = []
    for w in reader.sudachi.tokenize(text, reader.mode):
        s = w.surface()
        k = katakana(s)
        reading = k if KANA.fullmatch(k) else w.reading_form()
        sud.append((w.begin(), w.end(), reading, bool(w.is_oov() and KANJI.search(s))))
    may_start = {}
    for i, (s, e, _, pos, _) in enumerate(uni):
        prev = uni[i - 1][3] if i else None
        may_start[s] = (pos not in NO_START and prev != '接頭辞' and not (prev == '名詞' and pos == '名詞'))
    common = sorted({e for _, e, *_ in uni} & {e for _, e, *_ in sud})
    out, prev = [], 0
    for end in common:
        ws_u = [w for w in uni if prev <= w[0] and w[1] <= end]
        ws_s = [w for w in sud if prev <= w[0] and w[1] <= end]
        surface = text[prev:end]
        reading = None
        if not any(w[-1] for w in ws_u) and not any(w[-1] for w in ws_s):
            ra, rb = ''.join(w[2] for w in ws_u), ''.join(w[2] for w in ws_s)
            na, nb = norm(surface, ra), norm(surface, rb)
            if na == nb:
                reading = rb
            elif (surface, na, nb) in table and not re.search(r'[^一-龠々ぁ-ゖァ-ヺー]', surface):
                reading = table[(surface, na, nb)]
            if reading is not None and not KANA.fullmatch(reading):
                reading = None
        if (surface, reading) in AGREED:
            typed, keep = AGREED[(surface, reading)]
            reading = reading if text.startswith(keep, end) else typed
        out.append((prev, end, reading, may_start.get(prev, False)))
        prev = end
    return out


def reading_of(segments, i, j):
    parts = [segments[k][2] for k in range(i, j)]
    return None if any(p is None for p in parts) else hiragana(''.join(parts))


def fits(cfg, target, reading):
    lo, hi = cfg['target_chars']
    rlo, rhi = cfg['reading_chars']
    return lo <= len(target) <= hi and rlo <= len(reading) <= rhi


def span_sample(cfg, rng, paragraph, stats):
    runs = [m for m in RUN.finditer(paragraph)
            if cfg['run_chars'][0] <= len(m.group()) <= cfg['run_chars'][1] and KANJI.search(m.group())]
    if not runs:
        stats['span_no_run'] += 1
        return None
    rng.shuffle(runs)
    for m in runs[:cfg['runs_tried']]:
        run = m.group()
        segs = analyse(run)
        options = []
        for i, (s, _, _, ok) in enumerate(segs):
            if not ok:
                continue
            ends = []
            for j in range(i + 1, len(segs) + 1):
                reading = reading_of(segs, i, j)
                if reading is None:
                    break
                target = run[s:segs[j - 1][1]]
                # a target ends at the run end or where the next word may begin a conversion
                if (j == len(segs) or segs[j][3]) and fits(cfg, target, reading) and KANJI.search(target):
                    ends.append((j, target, reading))
            if ends:
                options.append((s, ends))
        if not options:
            continue
        s, ends = rng.choice(options)
        full = ends[-1][0] == len(segs)
        if full and rng.random() < cfg['end_at_run_end_probability']:
            _, target, reading = ends[-1]
        else:
            _, target, reading = rng.choice(ends)
        start = m.start() + s
        stats['span_run_end'] += start + len(target) == m.end()
        return paragraph[max(0, start - cfg['context_chars']):start], reading, target
    stats['span_no_target'] += 1
    return None


def sentence_sample(cfg, rng, paragraph, stats):
    lo, hi = cfg['target_chars']
    options = [s.strip() for s in SPLIT.split(paragraph)]
    options = [s for s in options if lo <= len(s) <= hi and RUN.fullmatch(s) and CONVERT.search(s)]
    if not options:
        stats['sentence_none_typeable'] += 1
        return None
    rng.shuffle(options)
    for s in options[:cfg['sentences_tried']]:
        segs = analyse(s)
        if not segs[0][3]:
            continue
        reading = reading_of(segs, 0, len(segs))
        if reading is not None and fits(cfg, s, reading):
            return '', reading, s
    stats['sentence_no_reading'] += 1
    return None


def sample(source, key, paragraph, rows, stats):
    cfg, label = _state['config'], _state['config']['label']
    if doc_hash(label + '/select', f'{source}/{key}') >= cfg['fractions'][source] * 2 ** 64:
        return
    rng = random.Random(doc_hash(label, f'{source}/{key}'))
    weights = cfg['kind_weights']
    kind = rng.choices(list(weights), weights=list(weights.values()))[0]
    stats['paragraphs'] += 1
    stats[f'{kind}_drawn'] += 1
    result = (span_sample if kind == 'span' else sentence_sample)(cfg['sample'], rng, paragraph, stats)
    if result is None:
        return
    context, reading, output = result
    stats[kind] += 1
    stats[f'{kind}_output_chars'] += len(output)
    stats[f'{kind}_reading_chars'] += len(reading)
    stats[f'{kind}_with_context'] += bool(context)
    for name, value in zip(COLUMNS, (source, key, kind, context, reading, output)):
        rows[name].append(value)


# ---------------------------------------------------------------- jobs

def jobs(config, sources, keep=lambda stem: True):
    """Job list; web files whose stem `keep` rejects are not opened (a copy may be in progress)."""
    import pyarrow.parquet as pq
    out = []
    if 'web' in sources:
        web = ROOT / config['web_dir']
        per = config['web_row_groups_per_job']
        for split in SPLITS:
            for path in sorted((web / split).glob('*.parquet')):
                if not keep(path.stem):
                    continue
                groups = pq.ParquetFile(path).metadata.num_row_groups
                for g in range(0, groups, per):
                    out.append(dict(name=f'web-{split}-{path.stem}-{g:03d}', source='web', split=split,
                                    path=path.relative_to(ROOT).as_posix(), groups=[g, min(groups, g + per)]))
    if 'lm' in sources:
        lm = ROOT / config['lm_dir']
        per = config['lm_bytes_per_job']
        for split in SPLITS:
            path = lm / f'{split}.txt'
            size = path.stat().st_size
            for n, b in enumerate(range(0, size, per)):
                out.append(dict(name=f'lm-{split}-{n:03d}', source='lm', split=split,
                                path=path.relative_to(ROOT).as_posix(), bytes=[b, min(size, b + per)]))
    if 'chat' in sources:
        out.extend(dict(name=f'chat-{split}', source='chat', split=split) for split in SPLITS)
    return out


def lm_lines(path, begin, end):
    """Lines starting in [begin, end); a line belongs to the job its first byte falls in."""
    with open(path, 'rb') as f:
        if begin:
            f.seek(begin - 1)
            if f.read(1) != b'\n':
                f.readline()
        position = f.tell()
        while position < end:
            raw = f.readline()
            if not raw:
                break
            yield position, raw.decode('utf-8')
            position += len(raw)


def paragraphs(job):
    if job['source'] == 'web':
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(ROOT / job['path'])
        for g in range(*job['groups']):
            table = pf.read_row_group(g, columns=['id', 'text'])
            for ident, text in zip(table.column('id').to_pylist(), table.column('text').to_pylist()):
                for n, line in enumerate(text.split('\n')):
                    yield f'{ident}#{n}', line
    elif job['source'] == 'lm':
        for position, line in lm_lines(ROOT / job['path'], *job['bytes']):
            yield f'{job["split"]}@{position}', line
    else:
        sys.path.insert(0, str(ROOT / 'scripts/ja/benchmarks'))
        from prepare_standard_ime_v2 import chat_lines
        n = 0
        for split, line in chat_lines():
            if split == job['split']:
                yield f'{split}#{n}', line
                n += 1


def init(config):
    _state['config'] = config
    _state['reader'] = DualReading()
    _state['table'] = load_table(ROOT / config['table'])


def run_job(job):
    import pyarrow as pa
    import pyarrow.parquet as pq
    parts = ROOT / _state['config']['output_dir'] / 'parts'
    started, rows, stats = time.time(), {c: [] for c in COLUMNS}, collections.Counter()
    for key, text in paragraphs(job):
        paragraph = clean(text)
        if len(paragraph) < 4:
            continue
        sample(job['source'], key, paragraph, rows, stats)
    table = pa.table({c: pa.array(rows[c], pa.string()) for c in COLUMNS})
    tmp = parts / f'{job["name"]}.parquet.tmp'
    pq.write_table(table, tmp, compression='zstd', row_group_size=200_000)
    tmp.replace(parts / f'{job["name"]}.parquet')
    record = dict(job=job, rows=table.num_rows, stats=dict(stats), seconds=round(time.time() - started, 1),
                  sha256=sha256(parts / f'{job["name"]}.parquet'))
    (parts / f'{job["name"]}.json').write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding='utf-8')
    return job['name'], table.num_rows, record['seconds']


def cmd_build(config, args):
    parts = ROOT / config['output_dir'] / 'parts'
    parts.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only.split(',')) if args.only else None
    unwanted = set(args.exclude.split(',')) if args.exclude else set()

    def keep(stem):
        return stem not in unwanted and (wanted is None or any(stem in w for w in wanted))

    todo = jobs(config, args.sources.split(','), keep)
    if args.only:
        wanted = set(args.only.split(','))
        todo = [j for j in todo if j['name'] in wanted or Path(j.get('path', '')).stem in wanted]
    if args.exclude:
        unwanted = set(args.exclude.split(','))
        todo = [j for j in todo if j['name'] not in unwanted and Path(j.get('path', '')).stem not in unwanted]
    todo = [j for j in todo if not (parts / f'{j["name"]}.json').exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f'{len(todo)} jobs', flush=True)
    started = time.time()
    with ProcessPoolExecutor(args.workers or config['workers'], initializer=init, initargs=(config,)) as pool:
        for n, (name, rows, seconds) in enumerate(pool.map(run_job, todo), 1):
            print(f'{n}/{len(todo)} {name} {rows} rows {seconds}s total {time.time() - started:.0f}s', flush=True)


def cmd_merge(config, args):
    out = ROOT / config['output_dir']
    expected = jobs(config, ('web', 'lm', 'chat'))
    missing = [j['name'] for j in expected if not (out / 'parts' / f'{j["name"]}.json').exists()]
    if missing:
        raise SystemExit(f'{len(missing)} jobs missing, e.g. {missing[:5]}')
    counts, stats, files = collections.Counter(), collections.defaultdict(collections.Counter), {}
    for j in expected:
        record = json.loads((out / 'parts' / f'{j["name"]}.json').read_text(encoding='utf-8'))
        if sha256(out / 'parts' / f'{j["name"]}.parquet') != record['sha256']:
            raise SystemExit(f'{j["name"]}: parquet hash mismatch')
        group = f'{j["source"]}/{j["split"]}'
        stats[group].update(record['stats'])
        files[f'parts/{j["name"]}.parquet'] = record['sha256']
        counts[group] += record['rows']
    reader = DualReading()
    manifest = dict(format=FORMAT, label=config['label'], config=config, rows=dict(counts),
                    stats={k: dict(sorted(v.items())) for k, v in stats.items()},
                    table_sha256=sha256(ROOT / config['table']), dictionary_versions=reader.versions,
                    code_sha256={n: sha256(ROOT / 'scripts/ja/corpus' / n) for n in ('kana_samples.py', 'reading_pairs.py')},
                    files=files)
    text = json.dumps(manifest, ensure_ascii=False, indent=1)
    (out / 'manifest.json').write_text(text, encoding='utf-8')
    print(json.dumps(dict(rows=manifest['rows'], manifest_sha256=hashlib.sha256(text.encode()).hexdigest()), indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('build', 'merge'))
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ja/data-ja-kana.toml')
    parser.add_argument('--sources', default='web,lm,chat')
    parser.add_argument('--only', help='build: job names or input file stems, comma-separated')
    parser.add_argument('--exclude', help='build: job names or input file stems to skip, comma-separated')
    parser.add_argument('--limit', type=int, help='build: only the first N pending jobs')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--output', help='build: write elsewhere than output_dir (smoke runs)')
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding='utf-8'))
    if args.output:
        config['output_dir'] = args.output
    {'build': cmd_build, 'merge': cmd_merge}[args.command](config, args)


if __name__ == '__main__':
    main()
