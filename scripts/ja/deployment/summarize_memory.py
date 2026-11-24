"""Summarize preserved fresh-process simulator memory audits (MiB, not iPhone estimates)."""
import argparse
import json
from pathlib import Path


def summarize(directory):
    reports = {name: json.loads((directory / (name + '.json')).read_text())
               for name in ('direct', 'reload', 'engine', 'lm')}
    repeat = directory / 'lm_words-repeat.json'
    if repeat.exists():
        reports['lm_words'] = json.loads(repeat.read_text())
    rows = []
    for name in ('engine', 'lm', 'lm_words'):
        if name not in reports:
            continue
        report = reports[name]
        steady = next(s for s in report['stages'] if s['stage'] == 'worker_cycles_61_80')['after']['physical_mib']
        values = [x['memory']['physical_mib'] for x in report['checkpoints']]
        rows.append({'scenario': name, 'baseline_mib': report['baseline']['physical_mib'],
                     'warm_after_80_cycles_mib': steady, 'process_peak_mib': report['final']['process_lifetime_peak_mib'],
                     'warm_checkpoints_range_mib': max(values) - min(values),
                     'after_worker_release_mib': report['final']['physical_mib'],
                     'cycles_with_suggestions': report.get('cycles_with_published_suggestions')})
    direct = reports['direct']
    deltas = {}
    for phase in ('scores', 'next_words'):
        values = [x['memory']['physical_mib'] for x in direct['checkpoints'] if x.get('stage') == phase]
        deltas[phase + '_20_to_100_growth_mib'] = values[-1] - values[0]
    reload_values = [x['memory']['physical_mib'] for x in reports['reload']['checkpoints']]
    prior = json.loads((directory / 'prior-keyboard-trace.json').read_text())
    failed = directory / 'lm_words-failed.json'
    summary = {'scope': 'optimized iOS27.0 arm64 Simulator test host, CPU_ONLY; not actual keyboard extension',
               'rows': rows, 'bounded_growth': deltas,
               'reload_1_to_20_growth_mib': reload_values[-1] - reload_values[0],
               'reload_10_to_20_growth_mib': reload_values[-1] - reload_values[9],
               'first_lm_words_failed': failed.exists(),
               'prior_extension_sampled_peak_mib': prior['peak_physical_mib'],
               'prior_extension_lm_loaded_proven': False,
               'limitations': ['Single runs with varying baselines; no fixed keyboard memory limit inferred.',
                   'No new device connection, installation or recording.',
                   'A bounded plateau does not exclude all leaks.',
                   'Prior extension trace settings/source/workload do not prove LM-active measurement.']}
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='New report path; default DIRECTORY/summary.json')
    args = parser.parse_args()
    output = args.output or args.directory / 'summary.json'
    if output.exists():
        parser.error('Output exists; use --output with a new report path.')
    report = summarize(args.directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
