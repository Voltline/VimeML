"""Post-hoc: byte-fallback rate on validation web excluding newline (<0x0A>), per probe tokenizer.

usage: python -I -X utf8 scripts/tokenizer/probe_zh_v13_bytes.py ROOT PROBE_DIR
"""
import glob, json, sys
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm

ROOT, PROBE = Path(sys.argv[1]), Path(sys.argv[2])
out = PROBE / 'bytes-excluding-newline.json'
if out.exists():
    raise SystemExit(f'{out} exists')
texts = []
for p in sorted((ROOT / 'outputs/zh-corpus/export-r1/validation').glob('web-*.parquet')):
    texts += pq.read_table(p, columns=['text'])['text'].to_pylist()
newlines = sum(t.count('\n') for t in texts)
res = {}
for model in sorted(PROBE.glob('zh-unigram-*/tokenizer.model')):
    sp = spm.SentencePieceProcessor(model_file=str(model))
    nl = sp.piece_to_id('<0x0A>')
    ids = sp.encode(texts)
    tokens = sum(map(len, ids))
    byte = sum(1 for x in ids for i in x if sp.is_byte(i))
    nlc = sum(x.count(nl) for x in ids)
    res[model.parent.name] = {'tokens': tokens, 'byte_tokens': byte, 'newline_byte_tokens': nlc,
                              'byte_fraction_excl_newline': (byte - nlc) / (tokens - nlc),
                              'tokens_excl_newline': tokens - nlc}
    print(model.parent.name, res[model.parent.name], flush=True)
out.write_text(json.dumps({'newlines_in_text': newlines, 'results': res}, indent=1), encoding='utf-8')
