"""V2 sentence-complete validation subsets and frozen IME epoch diagnostics."""

import json
import math
from pathlib import Path

import numpy as np

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
            position = int(np.searchsorted(dataset.sentences, sentence, side="left"))
            extra = 0 if position == 0 else int(dataset.end[position - 1] - dataset.sentences[position - 1] - 1)
            first = sentence + extra
            indices.extend(
                range(first, first + math.ceil((len(tokens) - 1) / dataset.context_length))
            )
    finally:
        dataset.close()
    return indices, characters


def epoch_ime(model, tokenizer_dir, benchmarks, root, output, epoch_number):
    from vimeml.benchmarks.evaluate_ajimee import load_export, metrics, rerank, summarize
    from vimeml.benchmarks.standard_ime import FORMAT, allow_evaluation, load_standard_export, source_metrics
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
            path = Path(root) / directory
            header = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            standard = header.get("format") == FORMAT
            if standard:
                if header.get("split") != "development":
                    raise ValueError("Training diagnostics may only open standard IME development")
                allow_evaluation(header)
                manifest, provenance, rows = load_standard_export(path)
            else:
                manifest, provenance, rows = load_export(path)
            rerank(lm, rows)
            report = {
                "epoch": epoch_number,
                "benchmark": directory,
                "benchmark_manifest": manifest,
                "export_provenance": provenance,
                "metrics": summarize(rows),
                "policy": "Frozen joint tokenization; full-vocabulary suffix logP sum; stable ties; no EOS.",
            }
            if standard:
                report.update(source_metrics=source_metrics(rows, metrics),
                              label_quality=manifest["label_quality"],
                              labels_formal_gold=manifest["labels_formal_gold"])
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
