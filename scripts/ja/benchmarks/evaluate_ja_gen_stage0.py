"""ja-gen plan stage 0: Standard development pool coverage (0.1) and development baselines (0.3).

Reads only development pools: Standard v1 development at n-best 20/50/100 and ime-kana-ja-v1
development at n-best 20. The kana test track is refused. V4.2 epoch 20 (FP32, same checkpoint
and tokenizer as the frozen comparisons) reranks the n-best 20 pools.

Correctness of a displayed Top-k list is judged against every accepted answer:
- exact: the candidate equals an answer;
- notation-lenient: `notation.relation` is '=' or 'A' (equal UniDic lemma sequences and nested
  kanji sets) for some answer. Unlike `notation.error_class`, this also credits a notation variant
  when no answer is in the pool, which is what the generation line needs (generated candidates are
  outside the AzooKey pool).

    python scripts/ja/benchmarks/evaluate_ja_gen_stage0.py --output outputs/ja-gen-stage0-20261125
"""
import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from vimeml.training.data import file_sha, write_json
from vimeml.benchmarks.standard_ime import load_standard_export
from vimeml.benchmarks.evaluate_ajimee import rerank
from vimeml.benchmarks.comparison import rows as read_rows
from vimeml.benchmarks.notation import relation
from vimeml.benchmarks import statistics

BENCH = ROOT / 'artifacts/benchmarks'
POOLS = {
    'standard-n20': BENCH / 'ime-candidates-v2-imported/standard-v1-development-n20',
    'standard-n50': BENCH / 'ime-candidates-v2-imported/standard-v1-development-n50',
    'standard-n100': BENCH / 'ime-candidates-v3-imported/standard-v1-development-n100',
    'kana-n20': BENCH / 'ime-candidates-v3-imported/kana-ja-v1-development-n20',
}
CHECKPOINT = ROOT / 'outputs/training-v42-20261020/remote/checkpoints/best-lm.pt'
TOKENIZER = ROOT / 'artifacts/tokenizers/v4-a/ja-unigram-8192'
RECORDED = ROOT / 'outputs/evaluation-v42-20261020/standard-development/scores.jsonl'
ORDERS = {'azookey': 'AzooKey order', 'lm_context_sum': 'V4.2 rerank (contextual sum)'}


def load(name):
    path = POOLS[name]
    if 'test' in path.name or 'blind' in path.name:
        raise ValueError('Development pools only.')
    manifest, _, rows = load_standard_export(path)
    if manifest.get('sealed') or manifest.get('split') not in ('development',):
        raise ValueError(f'{name}: not a development pool.')
    return rows


def texts(row):
    return [c['text'] for c in row['candidates']]


_RELATION = {}


def lenient(choice, answers):
    for answer in answers:
        key = choice, answer
        if key not in _RELATION:
            _RELATION[key] = relation(choice, answer)
        if _RELATION[key] in ('=', 'A'):
            return True
    return False


def hit(row, order, k, mode):
    shown = row['orders'][order][:k]
    if mode == 'exact':
        return any(c in row['answers'] for c in shown)
    return any(lenient(c, row['answers']) for c in shown)


def band(row):
    n = len(row['answers'][0])
    return '4-10' if n <= 10 else '11-20' if n <= 20 else '21-40'


def strata(row):
    return f"{band(row)}/{'context' if row['left_context'] else 'no-context'}"


def share(values):
    return {'hits': sum(values), 'cases': len(values), 'rate': round(100 * sum(values) / len(values), 1)}


def by_stratum(rows, outcome):
    groups = {}
    for row, ok in zip(rows, outcome):
        for key in (band(row), 'context' if row['left_context'] else 'no-context', row['source']):
            groups.setdefault(key, []).append(ok)
    return {k: share(v) for k, v in sorted(groups.items())}


def coverage(pools):
    """Pool ceilings at n-best 20/50/100 and where the n20 misses go."""
    n20, n50, n100 = (pools[f'standard-n{n}'] for n in (20, 50, 100))
    assert [r['id'] for r in n20] == [r['id'] for r in n50] == [r['id'] for r in n100]
    result = {}
    for n, rows in (('20', n20), ('50', n50), ('100', n100)):
        exact = [bool(set(texts(r)) & set(r['answers'])) for r in rows]
        loose = [any(lenient(c, r['answers']) for c in texts(r)) for r in rows]
        result[f'n{n}'] = {'exact': share(exact), 'notation_lenient': share(loose),
                           'mean_pool_size': round(sum(len(texts(r)) for r in rows) / len(rows), 1),
                           'exact_by_stratum': by_stratum(rows, exact)}
    missed = [i for i, r in enumerate(n20) if not set(texts(r)) & set(r['answers'])]
    deeper = {
        'n20_exact_misses': len(missed),
        'recovered_by_n50': sum(bool(set(texts(n50[i])) & set(n50[i]['answers'])) for i in missed),
        'recovered_by_n100': sum(bool(set(texts(n100[i])) & set(n100[i]['answers'])) for i in missed),
        'n20_notation_variant_in_pool': sum(any(lenient(c, n20[i]['answers']) for c in texts(n20[i])) for i in missed),
    }
    deeper['generation_only'] = deeper['n20_exact_misses'] - deeper['recovered_by_n100']
    deeper['generation_only_by_stratum'] = dict(sorted(Counter(
        strata(n20[i]) for i in missed if not set(texts(n100[i])) & set(n100[i]['answers'])).items()))
    deeper['n20_misses_by_stratum'] = dict(sorted(Counter(strata(n20[i]) for i in missed).items()))
    result['deeper_pools'] = deeper
    return result


def baselines(rows, labels):
    result = {}
    for order in ORDERS:
        entry = {}
        for mode in ('exact', 'notation_lenient'):
            for k in (1, 3):
                values = [hit(r, order, k, mode) for r in rows]
                entry[f'{mode}_top{k}'] = share(values)
            entry[f'{mode}_top1_by_stratum'] = by_stratum(rows, [hit(r, order, 1, mode) for r in rows])
        result[order] = entry
    pool_exact = [bool(set(texts(r)) & set(r['answers'])) for r in rows]
    pool_loose = [any(lenient(c, r['answers']) for c in texts(r)) for r in rows]
    result['pool_ceiling'] = {'exact': share(pool_exact), 'notation_lenient': share(pool_loose)}
    for mode in ('exact', 'notation_lenient'):
        before = [hit(r, 'azookey', 1, mode) for r in rows]
        after = [hit(r, 'lm_context_sum', 1, mode) for r in rows]
        result[f'v42_vs_azookey_{mode}_top1'] = statistics.paired(before, after, labels)
    result['clusters'] = len(set(labels))
    result['fallbacks'] = dict(Counter(k for r in rows for k in r.get('fallbacks', {})))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Use an empty output directory')
    if not torch.cuda.is_available():
        parser.error('FP32 CUDA scoring is required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    pools = {name: load(name) for name in POOLS}
    summary = {'test_read': False, 'script_sha256': file_sha(Path(__file__)),
               'checkpoint_sha256': file_sha(CHECKPOINT), 'coverage_standard_development': coverage(pools)}
    print(json.dumps(summary['coverage_standard_development']['deeper_pools'], ensure_ascii=False), flush=True)
    from vimeml.training.infer_v4 import JapaneseLMV4
    lm = JapaneseLMV4(CHECKPOINT, TOKENIZER, 'cuda')
    out.mkdir(parents=True, exist_ok=True)
    for name, key in (('standard-n20', 'standard_development'), ('kana-n20', 'kana_ja_v1_development')):
        rows = copy.deepcopy(pools[name])
        with torch.inference_mode():
            rerank(lm, rows)
        if name == 'standard-n20':
            recorded = {r['id']: r['orders']['lm_context_sum'] for r in read_rows(RECORDED)}
            summary['standard_n20_matches_recorded_v42'] = sum(
                r['orders']['lm_context_sum'] == recorded[r['id']] for r in rows)
        labels = statistics.clusters(rows, template=None)
        summary[key] = baselines(rows, labels)
        (out / f'{key}-v42-scores.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')
        print(key, json.dumps({o: {m: summary[key][o][m] for m in ('exact_top1', 'notation_lenient_top1')}
                               for o in ORDERS}, ensure_ascii=False), flush=True)
    write_json(out / 'summary.json', summary)


if __name__ == '__main__':
    main()
