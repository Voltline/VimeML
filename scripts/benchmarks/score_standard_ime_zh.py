"""Score rerankings of the frozen Standard Chinese IME benchmark (ime-standard-zh-v1-r1).

score:   rank each item's frozen candidate pool and report exact Top-k against the frozen
         target and accepted alternatives. Without --scores the ranking is the pool order,
         i.e. the librime + rime-ice baseline. With --scores, a JSONL of
         {"id", "candidates": [pool texts in pool order], "scores": [float, ...]} reranks the
         pool by descending score; ties keep pool order; "scores": null falls back to pool order
         (counted). Items whose reference is not in the pool stay in the denominator.
compare: paired comparison of two score outputs on the same version, track and split.

development may be used freely. blind needs --blind-plan: a preregistered plan committed to Git,
unchanged in the working tree, naming this benchmark version. The run records the plan commit.

usage:
  python -I -X utf8 scripts/benchmarks/score_standard_ime_zh.py score FROZEN OUT --track full [--scores F] [--name N]
  python -I -X utf8 scripts/benchmarks/score_standard_ime_zh.py compare A_OUT B_OUT OUT.json
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from vimeml.benchmarks.statistics import paired  # noqa: E402

VERSION = 'ime-standard-zh-v1-r1'
SOURCES = ('crosswoz', 'risawoz', 'kdconv')
LENGTHS = (('2-3', 2, 3), ('4-6', 4, 6), ('7-12', 7, 12))
KS = (1, 3, 5, 10, 20)


def read_jsonl(path):
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def length_bin(target):
    for name, low, high in LENGTHS:
        if low <= len(target) <= high:
            return name
    raise ValueError(f'target length outside 2-12: {target}')


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)


def check_blind_plan(path):
    plan = Path(path).resolve()
    relative = plan.relative_to(ROOT).as_posix()
    if git('ls-files', '--error-unmatch', relative).returncode:
        raise SystemExit(f'blind plan is not committed: {relative}')
    if git('diff', '--quiet', 'HEAD', '--', relative).returncode:
        raise SystemExit(f'blind plan differs from the committed version: {relative}')
    if VERSION not in plan.read_text(encoding='utf-8'):
        raise SystemExit(f'blind plan does not name {VERSION}')
    commit = git('log', '-1', '--format=%H', '--', relative).stdout.strip()
    return {'path': relative, 'commit': commit}


def rank_pool(texts, scores):
    order = sorted(range(len(texts)), key=lambda i: (-scores[i], i))
    return [texts[i] for i in order]


def rates(rows):
    n = len(rows)
    in_pool = sum(r['rank'] is not None for r in rows)
    out = {'n': n, 'in_pool': in_pool}
    for k in KS:
        hits = sum(r['rank'] is not None and r['rank'] <= k for r in rows)
        out[f'top{k}'] = hits
        out[f'top{k}_rate'] = hits / n if n else None
    out['pool_conditional_top1'] = out['top1'] / in_pool if in_pool else None
    out['mrr'] = sum(1 / r['rank'] for r in rows if r['rank'] is not None) / n if n else None
    return out


def summarize(rows):
    by_source = {s: rates([r for r in rows if r['source'] == s]) for s in SOURCES}
    macro = {key: sum(by_source[s][key] for s in SOURCES) / len(SOURCES)
             for key in [f'top{k}_rate' for k in KS] + ['pool_conditional_top1', 'mrr']}
    return {
        'all': rates(rows),
        'macro_sources': macro,
        'by_source': by_source,
        'by_length': {name: rates([r for r in rows if r['length'] == name]) for name, _, _ in LENGTHS},
        'by_stratum': {f'{s}/{name}': rates([r for r in rows if r['source'] == s and r['length'] == name])
                       for s in SOURCES for name, _, _ in LENGTHS},
    }


def score(args):
    frozen, out = Path(args.frozen), Path(args.out)
    manifest = json.loads((frozen / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['version'] != VERSION:
        raise SystemExit(f'expected {VERSION}, found {manifest["version"]}')
    if out.exists():
        raise SystemExit(f'output exists, choose a new directory: {out}')
    plan = None
    if args.split == 'blind':
        if not args.blind_plan:
            raise SystemExit('blind needs --blind-plan (a committed preregistered plan)')
        plan = check_blind_plan(args.blind_plan)
    items = {i['id']: i for i in read_jsonl(frozen / 'items.jsonl')
             if i['split'] == args.split and (args.track == 'full' or i['keys_abbrev'] is not None)}
    pools = {p['id']: p for p in read_jsonl(frozen / f'pool-{args.track}.jsonl') if p['id'] in items}
    if set(pools) != set(items):
        raise SystemExit(f'pool-{args.track}.jsonl does not cover every {args.split} item')
    scores = None
    if args.scores:
        scores = {}
        for row in read_jsonl(args.scores):
            if row['id'] in items:
                if row['id'] in scores:
                    raise SystemExit(f'duplicate id in scores: {row["id"]}')
                scores[row['id']] = row
        missing = set(items) - set(scores)
        if missing:
            raise SystemExit(f'{len(missing)} {args.split} items have no scores, e.g. {sorted(missing)[0]}')
    overlap = {r['id'] for r in read_jsonl(frozen / 'overlap' / 'overlap.jsonl')}
    rows, fallbacks = [], 0
    for item_id in sorted(items):
        item, texts = items[item_id], [c['text'] for c in pools[item_id]['candidates']]
        ranked = texts
        if scores is not None:
            entry = scores[item_id]
            if entry.get('candidates') != texts:
                raise SystemExit(f'candidates of {item_id} differ from the frozen pool order')
            values = entry.get('scores')
            if values is None:
                fallbacks += 1
            else:
                if len(values) != len(texts) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                    raise SystemExit(f'invalid scores for {item_id}')
                ranked = rank_pool(texts, values)
        accepted = {item['target'], *item['alternatives']}
        rank = next((i + 1 for i, text in enumerate(ranked) if text in accepted), None)
        rows.append({'id': item_id, 'source': item['source'], 'dialogue': f'{item["source"]}:{item["dialogue_id"]}',
                     'length': length_bin(item['target']), 'target': item['target'], 'choice': ranked[0] if ranked else None,
                     'rank': rank, 'overlap': item_id in overlap})
    summary = {
        'version': VERSION, 'track': args.track, 'split': args.split,
        'system': args.name or ('rime-pool-order' if scores is None else Path(args.scores).stem),
        'scores': str(args.scores) if args.scores else None, 'fallbacks': fallbacks, 'blind_plan': plan,
        'labels': manifest['labels'],
        **summarize(rows),
        'sensitivity_without_overlap': rates([r for r in rows if not r['overlap']]),
        'overlap_items': sum(r['overlap'] for r in rows),
    }
    out.mkdir(parents=True)
    with open(out / 'items.jsonl', 'w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    a = summary['all']
    print(f'{summary["system"]} {args.track}/{args.split}: top1 {a["top1"]}/{a["n"]} ({a["top1_rate"]:.3f}), '
          f'in pool {a["in_pool"]}, pool-conditional {a["pool_conditional_top1"]:.3f}, '
          f'macro top1 {summary["macro_sources"]["top1_rate"]:.3f}, fallbacks {fallbacks}')


def compare(args):
    runs = []
    for directory in (args.a, args.b):
        summary = json.loads((Path(directory) / 'summary.json').read_text(encoding='utf-8'))
        rows = {r['id']: r for r in read_jsonl(Path(directory) / 'items.jsonl')}
        runs.append((summary, rows))
    (sa, ra), (sb, rb) = runs
    for key in ('version', 'track', 'split'):
        if sa[key] != sb[key]:
            raise SystemExit(f'{key} differs: {sa[key]} vs {sb[key]}')
    if set(ra) != set(rb):
        raise SystemExit('the two runs cover different items')
    ids = sorted(ra)
    # One item per dialogue by construction, so dialogue clusters are single items.
    labels = [ra[i]['dialogue'] for i in ids]
    result = {'version': sa['version'], 'track': sa['track'], 'split': sa['split'],
              'before': sa['system'], 'after': sb['system']}
    for k in (1, 3):
        hit = lambda r: r['rank'] is not None and r['rank'] <= k  # noqa: E731
        result[f'top{k}'] = paired([hit(ra[i]) for i in ids], [hit(rb[i]) for i in ids], labels)
        result[f'top{k}_by_source'] = {
            s: paired([hit(ra[i]) for i in ids if ra[i]['source'] == s], [hit(rb[i]) for i in ids if ra[i]['source'] == s],
                      [labels[n] for n, i in enumerate(ids) if ra[i]['source'] == s]) for s in SOURCES}
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    t = result['top1']
    print(f'{sa["system"]} -> {sb["system"]} top1: +{t["gain"]}/-{t["loss"]} (net {t["net"]}), '
          f'McNemar p {t["exact_mcnemar_p"]:.3g}, sign-flip p {t["cluster_sign_flip_p"]:.3g}, '
          f'95% {t["cluster_bootstrap_95"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('score')
    p.add_argument('frozen'); p.add_argument('out')
    p.add_argument('--track', choices=['full', 'abbrev'], required=True)
    p.add_argument('--split', choices=['development', 'blind'], default='development')
    p.add_argument('--scores'); p.add_argument('--name'); p.add_argument('--blind-plan')
    p = commands.add_parser('compare')
    p.add_argument('a'); p.add_argument('b'); p.add_argument('out')
    args = parser.parse_args()
    {'score': score, 'compare': compare}[args.command](args)


if __name__ == '__main__':
    main()
