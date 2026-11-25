"""ja-gen stage 1.4: train the ja-kana-v1 12K tokenizer (configs/ja/tokenizer-ja-kana.toml).

Sample (written once to train-sample.txt): web paragraphs of ja-gen-export-v1 train (the zh
sample_shard rule, equal quota per FineWeb-2 shard), V4 LM lines (outputs/corpus-fast-v1
train.txt, uniform over lines), V4.1 chat train utterances, and hiragana readings drawn from
the train reading samples (outputs/ja-corpus/kana-samples-v1, merged). Settings and reserved
ids are those of the zh 12K tokenizer; required_chars makes every kana a piece.

Checks (all must pass): special and reserved ids, 256 byte pieces, newline one id, literal
markers stay text, every kana in required_chars encodes to its own single piece, and exact
roundtrip of the export validation documents, 20,000 V4 LM validation lines and the
validation readings.

    python -X utf8 scripts/ja/tokenizer/train_kana.py
"""
import argparse
import json
import random
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/ja/benchmarks'))
from vimeml.tokenizer.zh13 import (check_files, export_identity, paragraphs, sample_shard, sha256,  # noqa: E402
                                   shard_of, verify, web_train_files, write_json)


def reservoir(stream, k, rng):
    out = []
    for n, item in enumerate(stream):
        if n < k:
            out.append(item)
        else:
            j = rng.randrange(n + 1)
            if j < k:
                out[j] = item
    return out


def text_lines(path, max_bytes):
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and len(line.encode('utf-8')) <= max_bytes:
                yield line


def readings(samples, split, rng, k):
    import pyarrow.parquet as pq
    files = sorted((samples / 'parts').glob(f'*-{split}-*.parquet')) + sorted((samples / 'parts').glob(f'chat-{split}.parquet'))
    rows = [pq.ParquetFile(f).metadata.num_rows for f in files]
    total, out = sum(rows), []
    for f, n in zip(files, rows):  # each part gives its share of k, drawn uniformly inside it
        if n:
            column = pq.read_table(f, columns=['reading']).column('reading').to_pylist()
            out += rng.sample(column, min(n, round(k * n / total)))
    return out


def sample(config, export, out):
    import pyarrow.parquet as pq
    rng = random.Random(config['seed'])
    shards = {}
    for path in web_train_files(export):
        shards.setdefault(shard_of(path), []).append(path)
    quota = config['web_lines'] // len(shards) + 1
    web = []
    for shard in sorted(shards):
        handles = [('web', pq.ParquetFile(p)) for p in shards[shard]]
        web += [line for _, line in sample_shard(rng, handles, quota, config['max_line_bytes'])]
    web = web[:config['web_lines']]
    lm = reservoir(text_lines(ROOT / config['lm_dir'] / 'train.txt', config['max_line_bytes']), config['lm_lines'], rng)
    from prepare_standard_ime_v2 import chat_lines
    chat = reservoir((t.strip() for s, t in chat_lines() if s == 'train' and t.strip()), config['chat_lines'], rng)
    read = readings(ROOT / config['samples_dir'], 'train', rng, config['reading_lines'])
    lines = web + lm + chat + read
    rng.shuffle(lines)
    path = out / 'train-sample.txt'
    path.write_bytes(('\n'.join(lines) + '\n').encode('utf-8'))
    return path, dict(web_lines=len(web), lm_lines=len(lm), chat_lines=len(chat), reading_lines=len(read),
                      per_shard_quota=quota, characters=sum(map(len, lines)),
                      reading_characters=sum(map(len, read)), sha256=sha256(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/ja/tokenizer-ja-kana.toml')
    parser.add_argument('--output', type=Path, help='write elsewhere than output_dir (smoke runs)')
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding='utf-8'))
    export, out = ROOT / config['export_dir'], args.output or ROOT / config['output_dir']
    samples = ROOT / config['samples_dir']
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} is not empty')
    if not (samples / 'manifest.json').exists():
        raise SystemExit(f'{samples} has no manifest.json; run kana_samples.py merge first')
    identity = export_identity(export)
    validation = sorted((export / 'validation').glob('*.parquet'))
    check_files(export, web_train_files(export) + validation, identity, verify=True)
    out.mkdir(parents=True, exist_ok=True)
    began = time.time()
    sample_path, info = sample(config, export, out)
    print(json.dumps(info), flush=True)

    import pyarrow.parquet as pq
    import sentencepiece as spm
    folder = out / f'ja-unigram-{config["vocab_size"]}'
    folder.mkdir()
    spm.SentencePieceTrainer.train(
        input=str(sample_path), model_prefix=str(folder / 'tokenizer'), model_type='unigram',
        vocab_size=config['vocab_size'], character_coverage=config['character_coverage'], byte_fallback=True,
        required_chars=config['required_chars'],
        normalization_rule_name='identity', remove_extra_whitespaces=False, add_dummy_prefix=False,
        split_digits=False, hard_vocab_limit=True, pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        control_symbols=config['control_symbols'], user_defined_symbols=config['user_defined_symbols'],
        input_sentence_size=0, shuffle_input_sentence=False, max_sentence_length=config['max_line_bytes'],
        seed_sentencepiece_size=1_000_000, num_threads=config['num_threads'], minloglevel=2)
    model = folder / 'tokenizer.model'

    rng = random.Random(config['seed'] + 1)
    texts = [t for p in validation for t in pq.read_table(p, columns=['text'])['text'].to_pylist()]
    texts += reservoir(text_lines(ROOT / config['lm_dir'] / 'validation.txt', config['max_line_bytes']), 20000, rng)
    val_readings = readings(samples, 'validation', rng, 50000)
    checks = verify(model, {**config, 'vocab': config['vocab_size']}, texts + val_readings)
    sp = spm.SentencePieceProcessor(model_file=str(model))
    split_kana = [c for c in config['required_chars']
                  if len(sp.encode(c)) != 1 or sp.is_byte(sp.encode(c)[0]) or sp.piece_to_id(c) == sp.unk_id()]
    if split_kana:
        raise SystemExit(f'kana without a piece of their own: {"".join(split_kana)}')
    def tokens(xs):
        return sum(len(sp.encode(x)) for x in xs)
    stats = dict(validation_documents=dict(characters=sum(map(len, texts)), tokens=tokens(texts)),
                 validation_readings=dict(characters=sum(map(len, val_readings)), tokens=tokens(val_readings)),
                 kana_single_pieces=len(config['required_chars']))
    manifest = dict(format='vimeml_ja_kana_tokenizer_v1', status='complete', config=config,
                    export={k: v for k, v in identity.items() if k != 'sums'},
                    samples_manifest_sha256=sha256(samples / 'manifest.json'), sample=info,
                    model=dict(directory=folder.relative_to(out).as_posix(), tokenizer_model_sha256=sha256(model),
                               **checks, **stats),
                    libraries=dict(sentencepiece=spm.__version__),
                    code_sha256=sha256(Path(__file__)), seconds=round(time.time() - began, 1))
    write_json(out / 'manifest.json', manifest)
    print(json.dumps(manifest['model'], ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
