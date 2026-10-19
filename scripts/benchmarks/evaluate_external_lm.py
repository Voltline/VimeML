"""Evaluate a pinned Japanese external LM on the four frozen nonblind pools.

Raw joint-tokenized suffix likelihood, without instructions or answer prompts.
The vocabulary head is applied only at scored positions to fit an 8 GiB GPU.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
# Optional packages are isolated from the existing training environment.
OPTIONAL = ROOT / 'artifacts/external-eval/site-packages'
if OPTIONAL.is_dir():
    sys.path.insert(0, str(OPTIONAL))
sys.path.insert(0, str(ROOT / 'src'))

import torch
from torch.nn import functional as F
from vimeml.training.data import file_sha, write_json
from vimeml.training.evaluate_ime import common_prefix_length
from vimeml.benchmarks.evaluate_ajimee import eligibility, metrics, summarize
from vimeml.benchmarks.comparison import paired, rows as read_rows
from vimeml.benchmarks.standard_ime import source_metrics
from vimeml.posttraining.infer import TRACKS, read_snapshot

MODELS = {
    'tinyswallow': {
        'id': 'SakanaAI/TinySwallow-1.5B',
        'revision': '535d1322615c9f8171c3d60b10ef6955cc896fe9',
        'directory': 'artifacts/models/external/TinySwallow-1.5B',
        'output': 'outputs/evaluation-tinyswallow-20261016',
    },
    'rakuten-instruct': {
        'id': 'Rakuten/RakutenAI-2.0-mini-instruct',
        'revision': '6d902489587d324b7d5e201299e4e1a169f3a40b',
        'directory': 'artifacts/models/external/RakutenAI-2.0-mini-instruct',
        'output': 'outputs/evaluation-rakuten-instruct-20261016',
    },
    'sarashina': {
        'id': 'sbintuitions/sarashina2.2-1b',
        'revision': '3bb836ad7475ba192926be66651e4730825df7da',
        'directory': 'artifacts/models/external/sarashina2.2-1b',
        'output': 'outputs/evaluation-sarashina-20261016',
    },
}


class TokenizerAdapter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text, out_type=int):
        if out_type is not int:
            raise ValueError('Integer token IDs required')
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def id_to_piece(self, ident):
        return self.tokenizer.convert_ids_to_tokens(ident)


class ExternalLM:
    def __init__(self, model, tokenizer, candidate_batch=8, head_chunk=128):
        self.model = model.eval()
        self.processor = TokenizerAdapter(tokenizer)
        self.device = next(model.parameters()).device
        self.bos = model.config.bos_token_id
        if type(self.bos) is not int or candidate_batch < 1 or head_chunk < 1:
            raise ValueError('Native BOS and positive evaluation batches required')
        self.candidate_batch, self.head_chunk = candidate_batch, head_chunk
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.bos
        self.model.config.context_length = 128

    @torch.inference_mode()
    def score(self, context, candidates):
        if not candidates or any(not c for c in candidates) or len(set(candidates)) != len(candidates):
            raise ValueError('Distinct nonempty candidates required')
        context_ids = [self.bos, *self.processor.encode(context)]
        sequences = [[self.bos, *self.processor.encode(context + c)] for c in candidates]
        common = common_prefix_length([context_ids, *sequences])
        if common < 1 or any(not common < len(s) <= 129 for s in sequences):
            raise ValueError('Invalid suffix or context overflow; no truncation')
        result = []
        for start in range(0, len(sequences), self.candidate_batch):
            group = sequences[start:start + self.candidate_batch]
            width = max(len(s) - 1 for s in group)
            ids = torch.full((len(group), width), self.pad, dtype=torch.long, device=self.device)
            attention = torch.zeros_like(ids)
            row_ids, positions, targets, counts = [], [], [], []
            for row, seq in enumerate(group):
                ids[row, :len(seq) - 1] = torch.tensor(seq[:-1], device=self.device)
                attention[row, :len(seq) - 1] = 1
                suffix = seq[common:]
                counts.append(len(suffix))
                row_ids.extend([row] * len(suffix))
                positions.extend(range(common - 1, len(seq) - 1))
                targets.extend(suffix)
            hidden = self.model.model(input_ids=ids, attention_mask=attention,
                                      use_cache=False, return_dict=True).last_hidden_state
            selected = hidden[torch.tensor(row_ids, device=self.device),
                              torch.tensor(positions, device=self.device)]
            targets = torch.tensor(targets, device=self.device)
            values = []
            for offset in range(0, len(selected), self.head_chunk):
                logits = self.model.lm_head(selected[offset:offset + self.head_chunk])
                logp = -F.cross_entropy(logits.float(), targets[offset:offset + self.head_chunk], reduction='none')
                if not bool(torch.isfinite(logp).all()):
                    raise FloatingPointError('Non-finite full-vocabulary candidate probability')
                values.extend(logp.cpu().tolist())
            offset = 0
            for local, (seq, count) in enumerate(zip(group, counts, strict=True)):
                token_logp = values[offset:offset + count]
                offset += count
                suffix = seq[common:]
                result.append({'text': candidates[start + local],
                    'log_probability_sum': sum(token_logp),
                    'log_probability_mean': sum(token_logp) / count,
                    'scored_tokens': count, 'token_ids': suffix,
                    'pieces': [self.processor.id_to_piece(t) for t in suffix],
                    'token_log_probabilities': token_logp,
                    'boundary_retokenized': seq[:len(context_ids)] != context_ids})
        return {'common_prefix_tokens_including_bos': common,
                'context_tokens_including_bos': len(context_ids), 'candidates': result}


def rerank_external(lm, rows, output, track):
    for position, row in enumerate(rows):
        candidates = row['orders']['azookey']
        row['lm_scores'] = {}
        for field, context, methods in (
            ('contextual', row['left_context'], ('lm_context_sum', 'lm_context_mean_secondary')),
            ('context_free', '', ('lm_no_context_sum',)),
        ):
            reason = eligibility(lm, context, candidates)
            if reason:
                for method in methods:
                    row['orders'][method] = list(candidates)
                    row['fallbacks'][method] = reason
                continue
            # An empty original context gives exactly the same second score.
            scored = row['lm_scores'].get('contextual') if field == 'context_free' and not row['left_context'] else None
            if scored is None:
                scored = lm.score(context, candidates)
            row['lm_scores'][field] = scored
            for method in methods:
                key = 'log_probability_mean' if method.endswith('secondary') else 'log_probability_sum'
                row['orders'][method] = [x['text'] for x in sorted(scored['candidates'], key=lambda x: x[key], reverse=True)]
        if (position + 1) % 25 == 0 or position + 1 == len(rows):
            write_json(output / 'progress.json', {'status': 'running', 'track': track,
                'completed_cases': position + 1, 'total_cases': len(rows)})
            print(f'{track}: {position + 1}/{len(rows)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=tuple(MODELS), default='tinyswallow')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    selected = MODELS[args.model]
    model_id, revision = selected['id'], selected['revision']
    model_dir = ROOT / selected['directory']
    output = (args.output or ROOT / selected['output']).resolve()
    if (output / 'plan.json').exists():
        parser.error('An evaluation plan already exists; preserve this run and choose a new output')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('This frozen external evaluation requires a BF16 CUDA GPU')
    source = json.loads((ROOT / selected['output'] / 'model-source.json').read_text('utf-8'))
    if source['model_id'] != model_id or source['revision'] != revision:
        raise ValueError('Pinned official model source changed')
    files = {name: file_sha(model_dir / name) for name in source['files'] if name != '.gitattributes'}
    frozen = {track: read_snapshot(track) for track in TRACKS}
    inputs = {track: {'cases': len(rows), 'sha256': file_sha(ROOT / f'outputs/evaluation-v4-20261015/inputs/{track}.json')}
              for track, rows in frozen.items()}
    baseline_paths = {'v21': ROOT / 'outputs/evaluation-v4-20261015/v21',
                      'v4b': ROOT / 'outputs/evaluation-v4-b12cosine-20261015/best'}
    if args.model != 'tinyswallow':
        baseline_paths['tinyswallow'] = ROOT / MODELS['tinyswallow']['output']
    baseline_hashes = {name: {track: file_sha(folder / track / 'scores.jsonl') for track in TRACKS}
                       for name, folder in baseline_paths.items()}
    native = json.loads((model_dir / 'config.json').read_text('utf-8'))
    plan = {'model_id': model_id, 'revision': revision, 'model_files_sha256': files,
        'tracks': inputs, 'baseline_score_sha256': baseline_hashes,
        'precision': 'BF16 weights/forward; FP32 full-vocabulary CE; no quantization',
        'scoring': 'raw context+candidate joint tokenization; common-prefix suffix logP sum; native BOS; no final EOS',
        'primary': 'lm_context_sum', 'secondary': ['lm_context_mean_secondary', 'lm_no_context_sum'],
        'context_limit': 128, 'native_context_limit': native['max_position_embeddings'], 'candidate_batch': 8, 'head_chunk': 128,
        'instructions': False, 'chat_template': False, 'reading_conditioning': False,
        'stable_ties': True, 'overflow': 'whole-case AzooKey fallback; no truncation; full denominator',
        'training': False, 'blind_read': False, 'test_split_read': False,
        'training_overlap': 'External pretraining/distillation overlap with public benchmark text is unknown',
        'statistics': 'Paired exact McNemar descriptive results; four correlated exploratory tracks, no multiplicity-adjusted claim',
        'script_sha256': file_sha(Path(__file__)), 'torch': str(torch.__version__)}
    write_json(output / 'plan.json', plan)
    (output / 'evaluation-script.py').write_bytes(Path(__file__).read_bytes())
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    began = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
        trust_remote_code=False, use_safetensors=True, dtype=torch.bfloat16,
        attn_implementation='sdpa').to('cuda').eval()
    lm = ExternalLM(model, tokenizer)
    torch.cuda.synchronize()
    metadata = {'model_id': model_id, 'revision': revision, 'parameters': sum(p.numel() for p in model.parameters()),
        'vocab_size': model.config.vocab_size, 'native_bos': lm.bos,
        'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__), 'transformers': transformers.__version__,
        'precision': plan['precision'], 'model_loading_seconds': time.perf_counter() - began,
        'weights_allocated_MiB': torch.cuda.memory_allocated() / 2**20}
    write_json(output / 'environment.json', metadata)
    reports = {}
    for track, rows in frozen.items():
        started = time.perf_counter()
        rerank_external(lm, rows, output, track)
        destination = output / track
        destination.mkdir(parents=True, exist_ok=True)
        scores = destination / 'scores.jsonl'
        scores.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        report = {'status': 'complete', 'model': metadata, 'input': inputs[track],
                  'metrics': summarize(rows), 'elapsed_seconds': time.perf_counter() - started,
                  'scores_sha256': file_sha(scores)}
        if 'source' in rows[0]:
            report['source_metrics'] = source_metrics(rows, metrics)
        for name, folder in baseline_paths.items():
            comparison = paired(read_rows(folder / track / 'scores.jsonl'), rows)
            comparison['paired'].pop('significant_at_0_05', None)
            write_json(destination / f'paired-{name}.json', comparison)
        write_json(destination / 'metrics.json', report)
        reports[track] = report['metrics']['all']
        print(track, 'TOP1', reports[track]['lm_context_sum']['top1_correct'], '/', len(rows), flush=True)
    result = {'status': 'complete', 'model': metadata, 'tracks': reports,
        'peak_allocated_MiB': torch.cuda.max_memory_allocated() / 2**20,
        'peak_reserved_MiB': torch.cuda.max_memory_reserved() / 2**20,
        'elapsed_including_load_seconds': time.perf_counter() - began,
        'blind_read': False, 'test_split_read': False}
    write_json(output / 'summary.json', result)
    write_json(output / 'progress.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
