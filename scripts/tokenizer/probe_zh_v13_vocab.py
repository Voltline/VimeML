"""Exploratory Chinese tokenizer probe for vimeml-tiny-zh-v1: vocab 8K-32K and character coverage.

Tokenizer statistics only; no LM is trained. Inputs: export-r1 train (sample) and validation, and
the Standard zh benchmark development pools (blind items are filtered out and never processed).

usage: python -I -X utf8 scripts/tokenizer/probe_zh_v13_vocab.py ROOT OUT
Report: docs/zh-CN/reports/zh-v1-20261106/tokenizer-vocab.md
"""
import hashlib, json, random, re, sys, time
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
EXPORT = ROOT / 'outputs/zh-corpus/export-r1'
BENCH = ROOT / 'outputs/benchmarks/ime-standard-zh-v1/frozen-r1'
SEED = 20261106
WEB_LINES, CHAT_LINES, MAX_BYTES = 1_500_000, 80_000, 16384
# 8192 @ 0.9999 was also tried and cannot train: that coverage needs 10,417 required characters.
VARIANTS = [(8192, 0.9995), (12288, 0.9995), (16384, 0.9995), (24576, 0.9995), (32768, 0.9995),
            (12288, 0.9999)]
HAN = re.compile(r'[㐀-䶿一-鿿豈-﫿\U00020000-\U0003134f]')
# Fixed ~20M architecture family of the vimeml-tiny-zh-v1 plan (d384, SwiGLU d_ff 960, tied, learned pos 128).
D, DFF, CTX, BUDGET = 384, 960, 128, 20_040_000
PER_LAYER = 4 * D * D + 3 * D * DFF + 2 * D

if OUT.exists():
    raise SystemExit(f'{OUT} exists')
OUT.mkdir(parents=True)
log = open(OUT / 'run.log', 'a', encoding='utf-8')


def say(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True); log.write(msg + '\n'); log.flush()


def paragraphs(texts):
    for t in texts:
        for line in t.split('\n'):
            line = line.strip()
            if line and len(line.encode()) <= MAX_BYTES:
                yield line


# ---- 1. training sample ----------------------------------------------------------------------
rng = random.Random(SEED)
t0 = time.time()
web_files = sorted((EXPORT / 'train').glob('web-*.parquet'))
assert len(web_files) == 38, len(web_files)
per_file = WEB_LINES // len(web_files) + 1
web = []
for path in web_files:
    f = pq.ParquetFile(path)
    groups = sorted(rng.sample(range(f.metadata.num_row_groups), min(2, f.metadata.num_row_groups)))
    lines = list(paragraphs(f.read_row_groups(groups, columns=['text'])['text'].to_pylist()))
    web += rng.sample(lines, min(per_file, len(lines)))
web = web[:WEB_LINES]
chat = pq.read_table(EXPORT / 'train/chat-lccc-train.parquet', columns=['text'])['text'].to_pylist()
chat = [c for c in rng.sample(chat, CHAT_LINES) if c.strip()]
sample = web + chat
rng.shuffle(sample)
sample_path = OUT / 'train-sample.txt'
sample_path.write_text('\n'.join(sample) + '\n', encoding='utf-8')
sample_info = {'web_lines': len(web), 'chat_lines': len(chat), 'chars': sum(map(len, sample)),
               'han': sum(len(HAN.findall(s)) for s in sample),
               'sha256': hashlib.sha256(sample_path.read_bytes()).hexdigest(), 'seed': SEED,
               'web_files': [p.name for p in web_files], 'seconds': round(time.time() - t0, 1)}
say('sample', json.dumps(sample_info))
del web, chat, sample

# ---- 2. evaluation texts ---------------------------------------------------------------------
val_web = []
for path in sorted((EXPORT / 'validation').glob('web-*.parquet')):
    val_web += pq.read_table(path, columns=['text'])['text'].to_pylist()
val_chat = pq.read_table(EXPORT / 'validation/chat-lccc-validation.parquet', columns=['text'])['text'].to_pylist()
dev = {}
with open(BENCH / 'items.jsonl', encoding='utf-8') as stream:
    for line in stream:
        item = json.loads(line)
        if item['split'] != 'development':
            continue  # blind items are never processed
        dev[item['id']] = item
with open(BENCH / 'pool-full.jsonl', encoding='utf-8') as stream:
    pools = [json.loads(l) for l in stream]
pools = [p for p in pools if p['id'] in dev]
assert len(pools) == len(dev) == 1456, (len(pools), len(dev))
say('eval', len(val_web), 'web docs', len(val_chat), 'chat utterances', len(pools), 'dev pools')

han_counts = {}
for t in val_web + val_chat:
    for ch in HAN.findall(t):
        han_counts[ch] = han_counts.get(ch, 0) + 1
han_total = sum(han_counts.values())


def text_stats(sp, texts):
    ids = sp.encode(texts)
    tokens = sum(map(len, ids))
    byte_tokens = sum(1 for x in ids for i in x if sp.is_byte(i))
    chars = sum(map(len, texts))
    han = sum(len(HAN.findall(t)) for t in texts)
    return {'tokens': tokens, 'chars': chars, 'han': han, 'tokens_per_char': tokens / chars,
            'tokens_per_han': tokens / max(han, 1), 'chars_per_token': chars / tokens,
            'byte_token_fraction': byte_tokens / tokens, 'context126_chars': 126 * chars / tokens}


def pool_stats(sp):
    cand_tokens = cand_n = cand_byte = target_tokens = prefix_kept = joint_extra = 0
    for p in pools:
        left = dev[p['id']]['left_context']
        left_ids = sp.encode(left)
        target_tokens += len(sp.encode(dev[p['id']]['target']))
        for c in p['candidates']:
            ids = sp.encode(c['text'])
            cand_tokens += len(ids); cand_n += 1
            cand_byte += any(sp.is_byte(i) for i in ids)
            joint = sp.encode(left + c['text'])
            kept = joint[:len(left_ids)] == left_ids
            prefix_kept += kept
            joint_extra += len(joint) - len(left_ids)
    return {'tokens_per_candidate': cand_tokens / cand_n, 'candidates_with_byte_fallback': cand_byte / cand_n,
            'tokens_per_target': target_tokens / len(pools),
            'joint_left_prefix_kept': prefix_kept / cand_n, 'joint_tokens_after_left': joint_extra / cand_n}


def vocab_stats(sp):
    single, multi, other, lens = set(), 0, 0, []
    for i in range(sp.get_piece_size()):
        if sp.is_byte(i) or sp.is_control(i) or sp.is_unknown(i):
            continue
        piece = sp.id_to_piece(i)
        hans = HAN.findall(piece)
        if len(piece) == 1 and hans:
            single.add(piece)
        elif hans and len(hans) == len(piece):
            multi += 1; lens.append(len(piece))
        else:
            other += 1
    missing = {ch: n for ch, n in han_counts.items() if ch not in single}
    return {'han_single_pieces': len(single), 'han_multi_pieces': multi, 'other_pieces': other,
            'han_multi_mean_len': sum(lens) / max(len(lens), 1),
            'val_distinct_han': len(han_counts), 'val_distinct_han_without_piece': len(missing),
            'val_han_occurrences_byte_fallback': sum(missing.values()) / han_total,
            'top_missing': sorted(missing, key=missing.get, reverse=True)[:30]}


def params(vocab, tokens_per_han):
    emb = vocab * D
    layers = round((BUDGET - emb - CTX * D - D) / PER_LAYER)
    backbone = layers * PER_LAYER
    total = emb + CTX * D + backbone + D
    # Multiply-adds per token at full context: backbone weights + attention over CTX + output head.
    macs_token = backbone + layers * 2 * CTX * D + vocab * D
    return {'layers': layers, 'embedding': emb, 'backbone': backbone, 'total': total,
            'head_macs_share': vocab * D / macs_token, 'macs_per_token': macs_token,
            'macs_per_han': macs_token * tokens_per_han}


# ---- 3. train and measure --------------------------------------------------------------------
results = []
for vocab, coverage in VARIANTS:
    name = f'zh-unigram-{vocab}-cov{str(coverage)[2:]}'
    prefix = OUT / name / 'tokenizer'
    prefix.parent.mkdir()
    t0 = time.time()
    spm.SentencePieceTrainer.train(
        input=str(sample_path), model_prefix=str(prefix), model_type='unigram', vocab_size=vocab,
        character_coverage=coverage, byte_fallback=True, normalization_rule_name='identity',
        remove_extra_whitespaces=False, add_dummy_prefix=False, split_digits=False,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3, hard_vocab_limit=True,
        input_sentence_size=2_000_000, shuffle_input_sentence=False, max_sentence_length=MAX_BYTES,
        seed_sentencepiece_size=1_000_000, num_threads=16, minloglevel=2)
    seconds = time.time() - t0
    sp = spm.SentencePieceProcessor(model_file=str(prefix) + '.model')
    r = {'name': name, 'vocab': vocab, 'coverage': coverage, 'train_seconds': round(seconds),
         'val_web': text_stats(sp, val_web), 'val_chat': text_stats(sp, val_chat),
         'dev_pools': pool_stats(sp), 'vocab_stats': vocab_stats(sp)}
    r['params'] = params(vocab, r['val_web']['tokens_per_han'])
    (OUT / name / 'stats.json').write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding='utf-8')
    results.append(r)
    say(name, f'{seconds:.0f}s', 'tok/han', round(r['val_web']['tokens_per_han'], 4),
        'byte', round(r['val_web']['byte_token_fraction'], 4), 'L', r['params']['layers'])

(OUT / 'comparison.json').write_text(json.dumps({'sample': sample_info, 'results': results},
                                                ensure_ascii=False, indent=1), encoding='utf-8')
say('done')
