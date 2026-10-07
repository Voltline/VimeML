"""V2 sentence-complete validation subsets and frozen IME epoch diagnostics."""

import json
import math
from pathlib import Path

import sentencepiece as spm

from vimeml.training.data import write_json


def validation_subset(dataset, sampled_indices, tokenizer_dir):
    """Include all windows of sampled sentences so character counts are exact."""
    processor = spm.SentencePieceProcessor(model_file=str(Path(tokenizer_dir) / "tokenizer.model"))
    sentences = sorted({dataset.locate(index)[0] for index in sampled_indices})
    indices = []
    characters = 0
    dataset._open()
    try:
        for sentence in sentences:
            tokens = dataset._store[sentence]
            characters += len(processor.decode(tokens[1:-1]))
            first = sentence + sum(
                int(end - start - 1)
                for start, end, item in zip(dataset.first, dataset.end, dataset.sentences)
                if item < sentence
            )
            indices.extend(
                range(first, first + math.ceil((len(tokens) - 1) / dataset.context_length))
            )
    finally:
        dataset.close()
    return indices, characters


def epoch_ime(model, tokenizer_dir, benchmarks, root, output, epoch_number):
    from vimeml.benchmarks.evaluate_ajimee import load_export, rerank, summarize
    from vimeml.training.infer import JapaneseLM

    lm = JapaneseLM.__new__(JapaneseLM)
    lm.model = model
    lm.device = next(model.parameters()).device
    lm.processor = spm.SentencePieceProcessor(
        model_file=str(Path(tokenizer_dir) / "tokenizer.model")
    )
    lm.special = {
        name: getattr(lm.processor, f"{name}_id")() for name in ("pad", "unk", "bos", "eos")
    }
    reports = {}
    model.eval()
    try:
        for name, directory in benchmarks.items():
            manifest, provenance, rows = load_export(Path(root) / directory)
            rerank(lm, rows)
            report = {
                "epoch": epoch_number,
                "benchmark": directory,
                "benchmark_manifest": manifest,
                "export_provenance": provenance,
                "metrics": summarize(rows),
                "policy": "Frozen joint tokenization; full-vocabulary suffix logP sum; stable ties; no EOS.",
            }
            destination = Path(output) / "epoch-evaluation" / f"epoch-{epoch_number}" / name
            destination.mkdir(parents=True, exist_ok=True)
            write_json(destination / "metrics.json", report)
            with (destination / "scores.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            reports[name] = report["metrics"]["all"]["lm_context_sum"]
    finally:
        model.train()
    return reports
