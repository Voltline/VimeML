"""Summarize exported Instruments sysmon-process XML for Vime processes."""
import argparse
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


def summarize(source):
    root = ET.parse(source).getroot()
    columns = [c.findtext('mnemonic') for c in root.find('.//schema').findall('col')]
    references = {e.attrib['id']: e for e in root.iter() if 'id' in e.attrib}

    def resolve(element):
        while 'ref' in element.attrib:
            element = references[element.attrib['ref']]
        return element

    def number(element):
        element = resolve(element)
        return None if element.tag == 'sentinel' or element.text is None else float(element.text)

    processes = {}
    for row in root.findall('.//row'):
        values = dict(zip(columns, row))
        label = resolve(values['process']).get('fmt', '')
        if not (label.startswith('VimeKeyboard (') or label.startswith('Vime (')):
            continue
        footprint = number(values['memory-physical-footprint'])
        if footprint is None:
            continue
        item = {'seconds': number(values['time']) / 1e9,
                'physical_mib': footprint / 1048576,
                'resident_mib': number(values['memory-resident-size']) / 1048576,
                'private_resident_mib': number(values['memory-real-private']) / 1048576,
                'shared_resident_mib': number(values['memory-real-shared']) / 1048576,
                'threads': number(values['thread-count']),
                'recently_died': bool(number(values['recently-died']))}
        processes.setdefault(label, []).append(item)
    summary = {}
    for label, samples in processes.items():
        samples.sort(key=lambda s: s['seconds'])
        peak = max(samples, key=lambda s: s['physical_mib'])
        windows = []
        for start in range(0, int(samples[-1]['seconds']) + 1, 30):
            selected = [s['physical_mib'] for s in samples if start <= s['seconds'] < start + 30]
            if selected:
                windows.append({'start_seconds': start, 'end_seconds': start + 30,
                                'samples': len(selected), 'min_mib': min(selected),
                                'median_mib': statistics.median(selected), 'max_mib': max(selected)})
        intervals = [b['seconds'] - a['seconds'] for a, b in zip(samples, samples[1:])]
        summary[label] = {'sample_count': len(samples), 'first': samples[0], 'last': samples[-1],
                          'sampled_peak': peak, 'sampled_min_mib': min(s['physical_mib'] for s in samples),
                          'resident_metrics': {key: {'first_mib': samples[0][key],
                                                     'sampled_peak_mib': max(s[key] for s in samples),
                                                     'last_mib': samples[-1][key]}
                                               for key in ('private_resident_mib', 'shared_resident_mib', 'resident_mib')},
                          'median_interval_seconds': statistics.median(intervals) if intervals else None,
                          'recently_died_samples': sum(s['recently_died'] for s in samples),
                          'windows': windows, 'samples': samples}
    return {'source': str(source), 'metric': 'sysmon physical footprint; sampled peak, not lifetime high-water mark',
            'processes': summary}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new report path.")
    report = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: {key: value for key, value in v.items() if key != 'samples'}
                      for k, v in report['processes'].items()}, indent=2))
