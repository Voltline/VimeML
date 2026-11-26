"""ja-gen stage 1.5: encode the token store ja-kana-v1-12k (configs/ja/data-ja-kana-store.toml).

Kinds (one directory each, in the zh pinyin store layout; read by vimeml.training.joint_ja_kana):
  plain         V4 LM lines (outputs/corpus-fast-v1 {split}.txt)
  chat          V4.1 chat utterances
  web           ja-gen-export-v1 paragraphs not selected for reading samples (the complement of
                kana_samples.py's doc_hash(label/select, web/<id>#<n>) < fractions.web), so no web
                paragraph is both plain text and a reading sample
                -- the three plain kinds are BOS + line + EOS cut into <=128-token windows, every
                target supervised (first = 1)
  span, sentence          reading samples of the lm and chat parts of kana-samples-v1 (20M)
  web_span, web_sentence  reading samples of the web parts (200M)
                -- [BOS, CTX] + context + [READ] + reading + [OUT] + output + [EOS], first = the
                first output index; the context is cut on the left to fit 129 ids

Train lines of plain and chat equal to a validation line of either are dropped. Validation
reading samples are thinned by doc_hash(label/validation, source/key) < validation_fractions.
Lines with U+2581 or a lossy encoding are dropped and counted.

Files per kind and split: {split}.tokens.bin (<u2), {split}.offsets.bin (<u8, count+1),
{split}.first.bin (u1); parts are merged in job order, so a rebuild is byte-identical.

    python -X utf8 scripts/ja/tokenizer/encode_kana.py [--workers N]
    python -X utf8 scripts/ja/tokenizer/encode_kana.py --limit-jobs 1 --output /tmp/kana-store-smoke
"""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
import tomllib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/ja/benchmarks'))
from vimeml.posttraining.zh_pinyin import encode  # noqa: E402
from vimeml.posttraining.zh_pinyin_data import Part, merge  # noqa: E402
from vimeml.tokenizer.encode_zh13 import doc_hash  # noqa: E402
from vimeml.tokenizer.zh13 import META_SPACE, sha256, write_json  # noqa: E402

FORMAT = 'vimeml_ja_kana_v1'
KINDS = ('plain', 'chat', 'web', 'span', 'sentence', 'web_span', 'web_sentence')
SPLITS = ('train', 'validation')
NEWLINE = b'\n'
_state = {}


def line_key(text):
    return hashlib.sha256(text.encode('utf-8')).digest()[:8]


def chat_texts(split):
    from prepare_standard_ime_v2 import chat_lines
    return [t.strip() for s, t in chat_lines() if s == split and t.strip()]


def lm_texts(path, begin=0, end=None):
    """Stripped non-empty lines starting in [begin, end) (the kana_samples.lm_lines rule)."""
    end = path.stat().st_size if end is None else end
    with open(path, 'rb') as f:
        if begin:
            f.seek(begin - 1)
            if f.read(1) != NEWLINE:
                f.readline()
        position = f.tell()
        while position < end:
            raw = f.readline()
            if not raw:
                break
            position += len(raw)
            line = raw.decode('utf-8').strip()
            if line:
                yield line


def jobs(config, limit=0):
    import pyarrow.parquet as pq
    lm, web = ROOT / config['lm_dir'], ROOT / config['web_dir']
    parts = ROOT / config['samples_dir'] / 'parts'
    out = []
    for split in SPLITS:
        path = lm / f'{split}.txt'
        size, step = path.stat().st_size, config['lm_bytes_per_job']
        out += [dict(name=f'{split}-lm-{n:03d}', kind='plain', split=split, path=str(path), bytes=[b, min(size, b + step)])
                for n, b in enumerate(range(0, size, step))]
        out.append(dict(name=f'{split}-chat', kind='chat', split=split))
        for path in sorted((web / split).glob('*.parquet')):
            groups, per = pq.ParquetFile(path).metadata.num_row_groups, config['web_row_groups_per_job']
            out += [dict(name=f'{split}-web-{path.stem}-{g:03d}', kind='web', split=split, path=str(path),
                         groups=[g, min(groups, g + per)]) for g in range(0, groups, per)]
        for source in ('lm', 'chat', 'web'):
            out += [dict(name=f'{split}-read-{p.stem}', kind='read', split=split, source=source, path=str(p))
                    for p in sorted(parts.glob(f'{source}-{split}*.parquet'))]
    if limit:  # smoke: the first `limit` jobs of each (split, kind, source)
        seen, kept = Counter(), []
        for j in out:
            seen[j['split'], j['kind'], j.get('source')] += 1
            if seen[j['split'], j['kind'], j.get('source')] <= limit:
                kept.append(j)
        out = kept
    return out


def init(config, exclude):
    import sentencepiece as spm
    _state.update(config=config, exclude=exclude, sp=spm.SentencePieceProcessor(model_file=config['tokenizer_path']))


def windows(part, stats, kind, text):
    sp, minimum = _state['sp'], _state['config']['plain_minimum_tokens']
    if META_SPACE in text:
        stats[f'{kind}_meta_space'] += 1
        return
    pieces = sp.encode(text)
    if sp.decode(pieces) != text:
        stats[f'{kind}_lossy'] += 1
        return
    ids = [2] + pieces + [3]
    stats[f'{kind}_lines'] += 1
    for start in range(0, len(ids), 128):
        window = ids[start:start + 128]
        if len(window) < minimum:
            stats[f'{kind}_short_windows'] += 1
            continue
        part.add(kind, window, 1)
        stats[f'{kind}_windows'] += 1
        stats[f'{kind}_tokens'] += len(window)


def web_paragraphs(job, stats):
    """Web paragraphs (kana_samples.py keys) that kana_samples.py did not select."""
    import pyarrow.parquet as pq
    config = _state['config']
    cut = int(config['reading_fractions']['web'] * 2 ** 64)
    select = config['samples_label'] + '/select'
    handle = pq.ParquetFile(job['path'])
    for g in range(*job['groups']):
        table = handle.read_row_group(g, columns=['id', 'text'])
        for ident, text in zip(table.column('id').to_pylist(), table.column('text').to_pylist()):
            for n, line in enumerate(text.split('\n')):
                line = line.strip()
                if not line:
                    continue
                if doc_hash(select, f'web/{ident}#{n}') < cut:
                    stats['web_reading_paragraphs_skipped'] += 1
                    continue
                yield line


def run_job(job):
    import pyarrow.parquet as pq
    config, sp, exclude = _state['config'], _state['sp'], _state['exclude']
    part, stats = Part(KINDS), Counter()
    split, kind = job['split'], job['kind']
    if kind in ('plain', 'chat'):
        texts = lm_texts(Path(job['path']), *job['bytes']) if kind == 'plain' else chat_texts(split)
        for text in texts:
            if split == 'train' and line_key(text) in exclude:
                stats[f'{kind}_validation_duplicates'] += 1
                continue
            windows(part, stats, kind, text)
    elif kind == 'web':
        for text in web_paragraphs(job, stats):
            windows(part, stats, 'web', text)
    else:
        source = job['source']
        prefix = 'web_' if source == 'web' else ''
        fraction = config['validation_fractions'][source] if split == 'validation' else 1.0
        cut = int(fraction * 2 ** 64)
        table = pq.read_table(job['path'], columns=['key', 'kind', 'context', 'reading', 'output']).to_pydict()
        for key, sample_kind, context, reading, output in zip(
                table['key'], table['kind'], table['context'], table['reading'], table['output']):
            if fraction < 1.0 and doc_hash(config['label'] + '/validation', f'{source}/{key}') >= cut:
                continue
            store = prefix + sample_kind
            encoded = encode(sp, context, reading, output)
            if encoded is None:
                stats[f'{store}_skip_encode'] += 1
                continue
            ids, first, removed = encoded
            part.add(store, ids, first)
            r = ids.index(5)
            stats[f'{store}_examples_{source}'] += 1
            stats[f'{store}_tokens'] += len(ids)
            stats[f'{store}_output_tokens'] += len(ids) - first - 1
            stats[f'{store}_reading_tokens'] += first - r - 2
            stats[f'{store}_context_tokens'] += r - 2
            stats[f'{store}_reading_chars'] += len(reading)
            stats[f'{store}_output_chars'] += len(output)
            stats[f'{store}_context_trimmed'] += removed > 0
    part.save(Path(config['parts_dir']) / job['name'])
    return job['name'], job.get('source', kind), {k: len(part.lengths[k]) for k in KINDS}, dict(stats)


def preview(output_dir, sp, count=20):
    lines = []
    for kind in KINDS:
        d = Path(output_dir) / kind
        ids = np.fromfile(d / 'validation.tokens.bin', dtype='<u2')
        offsets = np.fromfile(d / 'validation.offsets.bin', dtype='<u8')
        first = np.fromfile(d / 'validation.first.bin', dtype='u1')
        for i in range(min(count, len(first))):
            seq = ids[offsets[i]:offsets[i + 1]].tolist()
            if first[i] == 1:
                lines.append({'kind': kind, 'text': sp.decode(seq)})
            else:
                r, o = seq.index(5), seq.index(6)
                lines.append({'kind': kind, 'context': sp.decode(seq[2:r]), 'reading': sp.decode(seq[r + 1:o]),
                              'output': sp.decode(seq[int(first[i]):-1])})
    with open(Path(output_dir) / 'preview.jsonl', 'w', encoding='utf-8', newline='\n') as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ja/data-ja-kana-store.toml')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--limit-jobs', type=int, default=0, help='smoke: first N jobs per split/kind/source (needs --output)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding='utf-8'))
    formal = ROOT / config['output_dir']
    if bool(args.limit_jobs) != bool(args.output) or (args.output and args.output.resolve() == formal.resolve()):
        parser.error('--limit-jobs and --output go together, outside output_dir')
    out = args.output or formal
    if out.exists():
        raise SystemExit(f'{out} exists; remove it first')
    began = time.time()

    tokenizer_dir = ROOT / config['tokenizer_dir']
    tok = json.loads((tokenizer_dir / 'manifest.json').read_text(encoding='utf-8'))
    model = tokenizer_dir / tok['model']['directory'] / 'tokenizer.model'
    if tok.get('status') != 'complete' or sha256(model) != tok['model']['tokenizer_model_sha256']:
        raise SystemExit('tokenizer manifest incomplete or tokenizer.model changed')
    samples = ROOT / config['samples_dir']
    samples_manifest = json.loads((samples / 'manifest.json').read_text(encoding='utf-8'))
    if (samples_manifest['label'] != config['samples_label']
            or samples_manifest['config']['fractions'] != config['reading_fractions']):
        raise SystemExit('samples_label / reading_fractions differ from kana-samples-v1')
    if not args.limit_jobs:
        for name, digest in samples_manifest['files'].items():
            if sha256(samples / name) != digest:
                raise SystemExit(f'{name} changed since kana_samples.py merge')
    web = ROOT / config['web_dir']
    sums = dict(line.split()[::-1] for line in (web / 'SHA256SUMS').read_text(encoding='utf-8').splitlines() if line.strip())
    web_files = sorted((web / 'train').glob('*.parquet')) + sorted((web / 'validation').glob('*.parquet'))
    if not args.limit_jobs:
        for path in web_files:
            if sha256(path) != sums[path.relative_to(web).as_posix()]:
                raise SystemExit(f'{path} differs from export-v1 SHA256SUMS')

    exclude = {line_key(t) for t in lm_texts(ROOT / config['lm_dir'] / 'validation.txt')}
    exclude |= {line_key(t) for t in chat_texts('validation')}
    config = dict(config, tokenizer_path=str(model), parts_dir=str(out / 'parts'))
    todo = jobs(config, args.limit_jobs)
    print(f'{len(todo)} jobs, {len(exclude)} validation lines excluded from train', flush=True)
    manifest = dict(format=FORMAT, status='partial', label=config['label'], stores={}, stats={})
    with ProcessPoolExecutor(args.workers or config['workers'], initializer=init, initargs=(config, exclude)) as pool:
        for split in SPLITS:
            names, stats, sources = [], Counter(), {}
            mine = [j for j in todo if j['split'] == split]
            for n, (name, source, counts, part_stats) in enumerate(pool.map(run_job, mine), 1):
                names.append(name)
                stats.update(part_stats)
                for kind, c in counts.items():
                    if c:
                        sources.setdefault(kind, Counter())[source] += c
                if n % 20 == 0 or n == len(mine):
                    print(f'{split} {n}/{len(mine)} jobs {time.time() - began:.0f}s', flush=True)
            manifest['stores'][split] = merge(config['parts_dir'], names, out, split, KINDS)
            for kind in KINDS:
                manifest['stores'][split][kind]['sources'] = dict(sources.get(kind, {}))
            manifest['stats'][split] = dict(sorted(stats.items()))
    shutil.rmtree(config['parts_dir'])
    import sentencepiece as spm
    preview(out, spm.SentencePieceProcessor(model_file=str(model)))
    manifest.update(
        status='complete',
        tokenizer_sha256=tok['model']['tokenizer_model_sha256'],
        tokenizer_manifest_sha256=sha256(tokenizer_dir / 'manifest.json'),
        samples_manifest_sha256=sha256(samples / 'manifest.json'),
        export_manifest_sha256=sha256(web / 'manifest.json'),
        lm_sha256={s: sha256(ROOT / config['lm_dir'] / f'{s}.txt') for s in SPLITS},
        validation_lines_excluded=len(exclude),
        config={k: v for k, v in config.items() if k not in ('parts_dir', 'tokenizer_path')},
        limit_jobs=args.limit_jobs,
        code_sha256={p: sha256(ROOT / p) for p in ('scripts/ja/tokenizer/encode_kana.py',
                                                   'src/vimeml/posttraining/zh_pinyin.py',
                                                   'src/vimeml/posttraining/zh_pinyin_data.py')},
        libraries={'sentencepiece': importlib.metadata.version('sentencepiece')},
        seconds=round(time.time() - began, 1))
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({s: {k: v['examples'] for k, v in st.items()} for s, st in manifest['stores'].items()}),
          'manifest_sha256', sha256(out / 'manifest.json'), flush=True)


if __name__ == '__main__':
    main()
