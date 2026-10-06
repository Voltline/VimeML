"""Prepare or verify a local GPL-2.0 v1 release. Never trains, converts or uploads."""
import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def verifier():
    spec = importlib.util.spec_from_file_location('vimeml_release_verifier', ROOT / 'templates/huggingface/verify_release.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8').strip()


def copy_checked(source, target, digest=None):
    check = verifier()
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f'Expected a regular source file: {source}')
    expected = digest or check.sha(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError(f'Refusing to overwrite: {target}')
    shutil.copyfile(source, target)
    if check.sha(target) != expected:
        raise ValueError(f'Copy checksum mismatch: {target}')


def evaluation_summary(bundle_manifest_sha, coreml_manifest_sha):
    from vimeml.benchmarks.evaluate_ajimee import load_export, summarize
    from vimeml.deployment.bundle import read_json
    from vimeml.deployment.validation import compare_rows
    sha = verifier().sha
    hashes, results = {}, {}

    def read(path):
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
        return read_json(path)

    def rows(path, metadata):
        expected = metadata['files_sha256']['scores.jsonl']
        if sha(path) != expected:
            raise ValueError(f'Score file identity mismatch: {path}')
        hashes[path.relative_to(ROOT).as_posix()] = expected
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

    for role, benchmark, baseline in [('dev', 'ime-dev-v2', 'tiny-ja-v1-dev-v2-scores'),
                                      ('ajimee', 'ajimee-jwtd-v2-v1', 'tiny-ja-v1-ajimee')]:
        directory = ROOT / f'outputs/deployment/conservative-int8-b32-{role}-v1'
        report = read(directory / 'metrics.json')
        if (report['model']['coreml_manifest_sha256'] != coreml_manifest_sha or
                report['model']['bundle_manifest_sha256'] != bundle_manifest_sha):
            raise ValueError('Saved evaluation uses a different model.')
        current = rows(directory / 'scores.jsonl', report)
        metrics = summarize(current)
        if metrics != report['metrics']:
            raise ValueError('Saved metrics do not match their score rows.')
        frozen, provenance, inputs = load_export(ROOT / 'artifacts/benchmarks' / benchmark)
        if frozen != report['benchmark_manifest'] or provenance != report['export_provenance'] or len(inputs) != len(current):
            raise ValueError('Frozen candidate export identity changed.')
        for left, right in zip(inputs, current):
            if any(left[key] != right[key] for key in ('id', 'query', 'left_context', 'answers', 'candidates')):
                raise ValueError('Saved candidate/input mismatch.')
        prior_directory = ROOT / 'outputs/ime-eval' / baseline
        prior_path = prior_directory / ('manifest.json' if (prior_directory / 'manifest.json').exists() else 'metrics.json')
        prior_report = read(prior_path)
        if any(prior_report['model'][key] != report['model'][key]
               for key in ('checkpoint_sha256', 'tokenizer_sha256')):
            raise ValueError('FP32 baseline belongs to different frozen weights or tokenizer.')
        prior = rows(prior_directory / 'scores.jsonl', prior_report)
        fp32_metrics = summarize(prior)
        comparison = compare_rows(prior, current)
        reported_before = prior
        if role == 'dev':
            source = ROOT / 'outputs/deployment/conservative-dev-v3'
            source_metadata = read(source / 'metrics.json')
            reported_before = rows(source / 'scores.jsonl', source_metadata)
        if compare_rows(reported_before, current) != report['comparison']:
            raise ValueError('Saved comparison cannot be reproduced from its actual baseline.')
        results[role] = {'original_order': metrics['all']['azookey'],
            'fp32_lm': fp32_metrics['all']['lm_context_sum'], 'int8_lm': metrics['all']['lm_context_sum'],
            'windows_fp32_comparison': {k: v for k, v in comparison.items() if k != 'changes'},
            'original_report_comparison_baseline': 'conservative Core ML' if role == 'dev' else 'Windows FP32',
            'original_report_comparison': {k: v for k, v in report['comparison'].items() if k != 'changes'},
            'benchmark_source_sha256': frozen['source_sha256'], 'candidate_export_sha256': provenance['files_sha256']['azookey-candidates.json']}
    alignment = read(ROOT / 'outputs/deployment/conservative-int8-b32-alignment-v1/alignment.json')
    if alignment['coreml_manifest_sha256'] != coreml_manifest_sha:
        raise ValueError('Alignment belongs to a different model.')
    distribution = read(ROOT / 'outputs/deployment/conservative-int8-b32-distribution-v1/report.json')
    training = read(ROOT / 'artifacts/models/tiny-ja-v1/summary.json')
    return {'format': 'vimeml_public_evaluation_v1', 'bundle_manifest_sha256': bundle_manifest_sha,
        'coreml_manifest_sha256': coreml_manifest_sha, 'training': {
            'prediction_targets': training['total_trained_tokens'], 'data_passes': training['completed_data_passes'],
            'full_validation': training['full_validation']['best'], 'test_used_for_selection': False},
        'ranking': results, 'alignment': {'strict_passed': alignment['passed'],
            'max_logit_abs': max(row['max_abs'] for row in alignment['logits']),
            'invariants': alignment['invariants']},
        'distribution': {'original_report': distribution,
            'aggregation': 'Mean of per-example position means; not corpus token-weighted.',
            'content_token_limit': 127, 'truncation_count_recorded': False,
            'original_report_has_full_identity_metadata': False},
        'source_reports_sha256': hashes,
        'scope': 'Offline fixed pools and saved Mac observations; not app-wide accuracy or a Windows Core ML runtime rerun.'}


def prepare(output, repo_id='Voltline/vimeml-tiny-ja-v1'):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Output exists; choose a new release version.')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo_id):
        raise ValueError('Use an OWNER/REPO model repository ID.')
    if git('status', '--porcelain'):
        raise ValueError('Commit the reviewed source on the current branch before preparing a release.')
    from vimeml.deployment.bundle import verify_bundle
    from vimeml.deployment.coreml import verify_package
    check = verifier()
    bundle = ROOT / 'artifacts/deployment/tiny-ja-v1-inference-v1'
    model = ROOT / 'artifacts/deployment/tiny-ja-v1-conservative-int8-b32-v1'
    metadata = verify_bundle(bundle)
    coreml = verify_package(model)
    bundle_sha, coreml_sha = check.sha(bundle / 'manifest.json'), check.sha(model / 'manifest.json')
    if (metadata['parameter_count'] != 7386624 or coreml['kind'] != 'linear8_fp32_compute' or
            coreml['minimum_ios'] != 18 or coreml['package_bytes'] != 8077801 or
            coreml['bundle']['bundle_manifest_sha256'] != bundle_sha):
        raise ValueError('Expected the reviewed frozen v1/INT8 package.')
    summary = evaluation_summary(bundle_sha, coreml_sha)
    for role, expected in [('dev', (137, 122, 134)), ('ajimee', (200, 124, 151))]:
        for variant in ('fp32_lm', 'int8_lm'):
            actual = summary['ranking'][role][variant]
            if tuple(actual[key] for key in ('cases', 'top1_correct', 'top5_correct')) != expected:
                raise ValueError('Saved metrics differ from this v1 model card; review before publishing.')
    commit = git('rev-parse', 'HEAD')
    output.mkdir(parents=True, exist_ok=False)
    for name in [*metadata['files'], 'manifest.json']:
        copy_checked(bundle / name, output / 'inference' / name)
    for name, entry in coreml['package_files'].items():
        copy_checked(model / 'model.mlpackage' / name,
                     output / 'coreml/ios18-int8-block32/model.mlpackage' / name, entry['sha256'])
    copy_checked(model / 'manifest.json', output / 'coreml/ios18-int8-block32/manifest.json')
    # Copy committed code/configuration, never data, annotations, artifacts or secrets.
    source_files = subprocess.check_output(['git', 'ls-files', '-z', '--', 'src', 'scripts', 'configs',
        'templates/huggingface', 'requirements.txt', 'LICENSE'], cwd=ROOT).decode('utf-8').split('\0')
    for name in filter(None, source_files):
        copy_checked(ROOT / name, output / 'source' / name)
    (output / 'source/README.md').write_text(
        f'VimeML source/configuration from Git commit {commit}. GPL-2.0; see LICENSE.\n'
        'The FP32 loader checks the original model and deployment code fingerprints.\n'
        'Corpus texts, annotations, optimizer state and the separate Vime client are not included.\n', encoding='utf-8', newline='\n')
    copy_checked(ROOT / 'LICENSE', output / 'LICENSE')
    for name in ('infer.py', 'verify_release.py'):
        copy_checked(ROOT / 'templates/huggingface' / name, output / name)
    card = (ROOT / 'templates/huggingface/README.md').read_text(encoding='utf-8')
    for key, value in {'SOURCE_COMMIT': commit, 'BUNDLE_SHA': bundle_sha, 'COREML_SHA': coreml_sha,
                       'TOKENIZER_SHA': metadata['files']['tokenizer.model']['sha256'],
                       'CHECKPOINT_SHA': metadata['checkpoint_sha256']}.items():
        card = card.replace(f'@@{key}@@', value)
    if '@@' in card:
        raise ValueError('Unresolved model card field.')
    (output / 'README.md').write_text(card, encoding='utf-8', newline='\n')
    write_json(output / 'evaluation/summary.json', summary)
    files = {p.relative_to(output).as_posix(): {'bytes': p.stat().st_size, 'sha256': check.sha(p)}
             for p in sorted(output.rglob('*')) if p.is_file()}
    release = {'format': 'vimeml_hf_release_v1', 'repo_id': repo_id, 'license': 'gpl-2.0',
        'source_repository': 'https://github.com/Voltline/VimeML', 'source_commit': commit,
        'created_utc': datetime.now(timezone.utc).isoformat(), 'bundle_manifest_sha256': bundle_sha,
        'coreml_manifest_sha256': coreml_sha, 'files': files,
        'policy': 'Local preparation only. No training, conversion, compression, device access or upload.'}
    write_json(output / 'RELEASE.json', release)
    sums = {**{name: entry['sha256'] for name, entry in files.items()}, 'RELEASE.json': check.sha(output / 'RELEASE.json')}
    (output / 'SHA256SUMS.txt').write_text(''.join(f'{digest}  {name}\n' for name, digest in sorted(sums.items())),
                                         encoding='utf-8', newline='\n')
    _, verified = check.verify(output)
    print(json.dumps({'release': str(output), 'repo_id': repo_id, 'source_commit': commit,
                      'verified_files': len(verified), 'bytes': sum((output / name).stat().st_size for name in verified)}, indent=2))
    return release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('prepare', help='Create a fresh local release from the committed source and frozen artifacts.')
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--repo-id', default='Voltline/vimeml-tiny-ja-v1')
    verify = sub.add_parser('verify', help='Read-only checksum/inventory verification.')
    verify.add_argument('--release', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output, args.repo_id)
    else:
        manifest, files = verifier().verify(args.release)
        print(f"Verified {len(files)} release files for {manifest['repo_id']}")


if __name__ == '__main__':
    main()
