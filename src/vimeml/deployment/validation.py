"""Numerical fixtures, frozen candidate evaluations and Mac-only timing probes."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from vimeml.benchmarks.ajimee import DEVELOPMENT_FORMAT, FORMAT
from vimeml.benchmarks.evaluate_ajimee import load_export, rerank, summarize
from vimeml.deployment.bundle import (fresh_directory, read_json, tree_inventory, write_json)
from vimeml.training.data import file_sha
from vimeml.training.evaluate_ime import score_candidates

SCORING_CASES = [
    {"context": "", "candidates": ["東京", "とうきょう", "東京都"]},
    {"context": "明日", "candidates": ["は", "は晴れ", "の会議"]},
    {"context": "コーヒーを", "candidates": ["飲む", "買う", "飲みたい"]},
    {"context": "ファイル", "candidates": ["名", "名前", "を開く"]},
]


def numeric_error(expected, actual, atol, rtol):
    if expected.shape != actual.shape or not np.isfinite(actual).all():
        raise ValueError("Nonfinite or mismatched comparison output.")
    delta = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    return {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
            "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "outside_tolerance": int((delta > atol + rtol * np.abs(expected)).sum()),
            "values": int(delta.size)}


@torch.inference_mode()
def reference(lm, output):
    if output.exists():
        raise ValueError("Reference output exists; choose a new directory.")
    special = lm.special
    short = lm.prefix_ids("明日の会議までに、")
    rng = np.random.default_rng(716)
    long = rng.integers(4, lm.model.config.vocab_size, size=lm.model.config.context_length).tolist()
    long[0] = special["bos"]
    changed = long.copy()
    changed[16:] = rng.integers(4, lm.model.config.vocab_size, size=len(long) - 16).tolist()
    # Explicit byte pieces exercise byte fallback even if the text is in the vocabulary.
    byte_ids = [lm.processor.piece_to_id(f"<0x{value:02X}>") for value in (0xF0, 0x9F, 0xA6, 0x84)]
    if special["unk"] in byte_ids:
        raise ValueError("Expected byte-fallback tokenizer.")
    fixtures = [
        {"id": "bos", "input_ids": [special["bos"]], "valid_length": 1},
        {"id": "short", "input_ids": short, "valid_length": len(short)},
        {"id": "right-pad", "input_ids": short + [special["pad"]] * (lm.model.config.context_length - len(short)), "valid_length": len(short)},
        {"id": "byte-eos", "input_ids": [special["bos"], *byte_ids, special["eos"]], "valid_length": 6},
        {"id": "long", "input_ids": long, "valid_length": len(long)},
        {"id": "causal-mutation", "input_ids": changed, "valid_length": len(changed)},
    ]
    arrays = {}
    for fixture in fixtures:
        logits = lm.model(torch.tensor([fixture["input_ids"]])).float().cpu().numpy()
        arrays[fixture["id"]] = logits
        fixture["last_top1"] = int(logits[0, fixture["valid_length"] - 1].argmax())
    torch.testing.assert_close(torch.from_numpy(arrays["short"]),
        torch.from_numpy(arrays["right-pad"][:, :len(short)]), atol=3e-5, rtol=3e-4)
    torch.testing.assert_close(torch.from_numpy(arrays["long"][:, :16]),
        torch.from_numpy(arrays["causal-mutation"][:, :16]), atol=3e-5, rtol=3e-4)
    scored, device_candidates = [], []
    for case in SCORING_CASES:
        result = score_candidates(lm, case["context"], case["candidates"])
        scored.append({**case, "scored": result})
        for candidate in result["candidates"]:
            seq = [special["bos"], *lm.processor.encode(case["context"] + candidate["text"], out_type=int)]
            device_candidates.append({"context": case["context"], "text": candidate["text"],
                "input_ids": seq[:-1], "targets": seq[1:],
                "score_start": result["common_prefix_tokens_including_bos"] - 1,
                "log_probability_sum": candidate["log_probability_sum"]})
    fresh_directory(output)
    np.savez_compressed(output / "logits.npz", **arrays)
    write_json(output / "fixtures.json", {"logits": fixtures, "scoring": scored})
    write_json(output / "device-fixtures.json", {"format": "vimeml_device_fixtures_v1",
        "bundle_manifest_sha256": lm.metadata["bundle_manifest_sha256"],
        "logits": fixtures, "candidates": device_candidates})
    write_json(output / "manifest.json", {"format": "vimeml_reference_v1", "status": "complete",
        "model": lm.metadata, "files": tree_inventory(output),
        "note": "Windows CPU FP32 reference. Explicit token IDs test tensor contract; device-side SentencePiece roundtrip/joint tokenization still requires separate integration tests."})


@torch.inference_mode()
def validate(lm, reference_dir, output, atol, rtol, score_atol):
    if output.exists():
        raise ValueError("Alignment output exists.")
    manifest = read_json(reference_dir / "manifest.json")
    if manifest.get("format") != "vimeml_reference_v1" or manifest.get("status") != "complete":
        raise ValueError("Invalid reference manifest.")
    if manifest["model"]["bundle_manifest_sha256"] != lm.metadata["bundle_manifest_sha256"]:
        raise ValueError("Reference belongs to another bundle.")
    for name, entry in manifest["files"].items():
        if file_sha(reference_dir / name) != entry["sha256"]:
            raise ValueError(f"Reference changed: {name}")
    fixtures = read_json(reference_dir / "fixtures.json")
    errors, predictions = [], {}
    with np.load(reference_dir / "logits.npz", allow_pickle=False) as expected:
        for fixture in fixtures["logits"]:
            ids = torch.tensor([fixture["input_ids"]], dtype=torch.long)
            actual = lm.model(ids).float().cpu().numpy()
            predictions[fixture["id"]] = actual
            valid = fixture["valid_length"]
            error = numeric_error(expected[fixture["id"]][:, :valid], actual[:, :valid], atol, rtol)
            error.update(id=fixture["id"], last_top1_equal=int(actual[0, valid - 1].argmax()) == fixture["last_top1"])
            errors.append(error)
    # Explicitly verify causality and right-PAD invariance on the converted model too.
    length = predictions["short"].shape[1]
    invariants = {
        "right_pad": numeric_error(predictions["short"], predictions["right-pad"][:, :length], atol, rtol),
        "causal": numeric_error(predictions["long"][:, :16], predictions["causal-mutation"][:, :16], atol, rtol),
    }
    score_errors = []
    for case in fixtures["scoring"]:
        scored = score_candidates(lm, case["context"], case["candidates"])
        before = case["scored"]["candidates"]
        after = scored["candidates"]
        if scored["common_prefix_tokens_including_bos"] != case["scored"]["common_prefix_tokens_including_bos"]:
            raise ValueError("Joint tokenization/common-prefix contract changed.")
        max_delta = 0.0
        for left, right in zip(before, after):
            if left["token_ids"] != right["token_ids"]:
                raise ValueError("Candidate target tokens changed.")
            max_delta = max(max_delta, abs(left["log_probability_sum"] - right["log_probability_sum"]))
        order = lambda values: [item["text"] for item in sorted(values, key=lambda item: item["log_probability_sum"], reverse=True)]
        score_errors.append({"context": case["context"], "max_sum_abs": max_delta,
                             "order_equal": order(before) == order(after), "scored": scored})
    passed = (all(item["outside_tolerance"] == 0 for item in [*errors, *invariants.values()]) and
              all(item["max_sum_abs"] <= score_atol and item["order_equal"] for item in score_errors))
    fresh_directory(output)
    report = {"format": "vimeml_alignment_v1", "passed": passed, "model": lm.metadata,
        "coreml_manifest_sha256": lm.metadata.get("coreml_manifest_sha256"),
        "reference_manifest_sha256": file_sha(reference_dir / "manifest.json"),
        "tolerances": {"logits_atol": atol, "logits_rtol": rtol, "score_sum_atol": score_atol},
        "logits": errors, "invariants": invariants, "scores": score_errors,
        "note": "Numerical fixture gate, not a quality certification. Review dev/AJIMEE and phrases separately. Quantization may fail FP16 tolerances; preserve failures and inspect before changing thresholds."}
    write_json(output / "alignment.json", report)
    return report


def compare_rows(before, after):
    if len(before) != len(after):
        raise ValueError("Baseline case count changed.")
    changed, max_delta, total_delta, score_count = [], 0.0, 0.0, 0
    for left, right in zip(before, after):
        for key in ("id", "query", "left_context", "answers", "candidates"):
            if left[key] != right[key]:
                raise ValueError(f"Baseline candidate/input mismatch: {key}")
        delta = None
        a = left.get("lm_scores", {}).get("contextual")
        b = right.get("lm_scores", {}).get("contextual")
        if a and b:
            if a["common_prefix_tokens_including_bos"] != b["common_prefix_tokens_including_bos"]:
                raise ValueError("Baseline tokenizer boundary mismatch.")
            deltas = []
            for x, y in zip(a["candidates"], b["candidates"]):
                if x["text"] != y["text"] or x["token_ids"] != y["token_ids"]:
                    raise ValueError("Baseline token targets mismatch.")
                deltas.append(abs(x["log_probability_sum"] - y["log_probability_sum"]))
            delta = max(deltas)
            max_delta = max(max_delta, delta)
            total_delta += sum(deltas)
            score_count += len(deltas)
        old, new = left["orders"]["lm_context_sum"], right["orders"]["lm_context_sum"]
        if old != new or left["fallbacks"] != right["fallbacks"]:
            changed.append({"id": right["id"], "old_order": old, "new_order": new,
                "top1_changed": old[:1] != new[:1], "answers": right["answers"],
                "max_score_sum_abs": delta, "old_fallbacks": left["fallbacks"], "new_fallbacks": right["fallbacks"]})
    return {"order_changed_cases": len(changed), "top1_changed_cases": sum(row["top1_changed"] for row in changed),
            "max_score_sum_abs": max_delta, "mean_score_sum_abs": total_delta / score_count if score_count else None,
            "changes": changed}


def evaluate(lm, benchmark, role, output, baseline=None):
    if output.exists():
        raise ValueError("Evaluation output exists; preserve the baseline.")
    manifest, provenance, rows = load_export(benchmark)
    expected_format = DEVELOPMENT_FORMAT if role == "dev" else FORMAT
    if manifest["format"] != expected_format:
        raise ValueError("Dataset role does not match the frozen benchmark format.")
    started = time.perf_counter()
    rerank(lm, rows)  # Reuse exact frozen scoring, eligibility, stable ties and whole-case fallback.
    comparison = None
    if baseline:
        baseline = Path(baseline)
        metadata_path = baseline / "manifest.json" if (baseline / "manifest.json").exists() else baseline / "metrics.json"
        prior = read_json(metadata_path)
        model = prior["model"]
        for key in ("checkpoint_sha256", "tokenizer_sha256"):
            if model[key] != lm.metadata[key]:
                raise ValueError(f"Baseline identity mismatch: {key}")
        if prior["benchmark_manifest"] != manifest or prior["export_provenance"] != provenance:
            raise ValueError("Baseline uses a different frozen dataset/candidate export.")
        digest = prior.get("files_sha256", {}).get("scores.jsonl")
        if not digest or file_sha(baseline / "scores.jsonl") != digest:
            raise ValueError("Baseline score hash mismatch.")
        prior_rows = [json.loads(line) for line in (baseline / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
        comparison = compare_rows(prior_rows, rows)
    fresh_directory(output)
    (output / "scores.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    report = {"format": "vimeml_deployment_metrics_v1", "status": "complete", "role": role,
        "model": lm.metadata, "benchmark_manifest": manifest, "export_provenance": provenance,
        "metrics": summarize(rows), "comparison": comparison,
        "files_sha256": {"scores.jsonl": file_sha(output / "scores.jsonl")},
        "elapsed_seconds": time.perf_counter() - started,
        "policy": "LM-only contextual logP sum primary. AzooKey retrieves; scores ignored except original-order ties/fallback. Mean and no-context secondary. No lambda selection on AJIMEE.",
        "note": "All cases remain in denominators. Fixed AJIMEE is an observed comparison, not a fresh blind test. Latency here includes diagnostics/context-free scores; use timing command for workload measurements."}
    write_json(output / "metrics.json", report)
    return report


@torch.inference_mode()
def timing(lm, benchmark, prompts, output, repeats, warmup):
    from vimeml.benchmarks.evaluate_ajimee import eligibility
    from vimeml.tools.phrase_demo import PhraseDemo
    if output.exists():
        raise ValueError("Timing output exists.")
    _, provenance, rows = load_export(benchmark)
    cases = read_json(prompts)
    demo = PhraseDemo(lm, cases)
    times, workloads = [], []
    for row in rows:
        texts = row["orders"]["azookey"]
        reason = eligibility(lm, row["left_context"], texts)
        if reason:
            workloads.append({"id": row["id"], "fallback": reason})
            continue
        def operation():
            return score_candidates(lm, row["left_context"], texts)
        for _ in range(warmup):
            operation()
        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter()
            operation()
            elapsed.append((time.perf_counter() - start) * 1000)
        times.extend(elapsed)
        workloads.append({"id": row["id"], "candidates": len(texts), "elapsed_ms": elapsed})
    phrase_times, phrases = [], []
    for case in cases:
        settings = {"prompt": case["prompt"], "max_tokens": 8, "mode": "beam", "count": 5, "seed": 42}
        for _ in range(warmup):
            demo.suggest(settings)
        elapsed = []
        for _ in range(repeats):
            result = demo.suggest(settings)
            elapsed.append(result["elapsed_ms"])
        phrase_times.extend(elapsed)
        phrases.append({"id": case["id"], "elapsed_ms": elapsed, "returned_count": result["returned_count"]})
    percentiles = lambda values: {"count": len(values), **({"p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)), "max_ms": max(values)} if values else {})}
    fresh_directory(output)
    write_json(output / "timing.json", {"model": lm.metadata, "warmup_per_case": warmup, "repeats": repeats,
        "prompts_sha256": file_sha(prompts), "export_provenance": provenance,
        "load_seconds": getattr(lm.model, "load_seconds", None),
        "reranking": percentiles(times), "phrases": percentiles(phrase_times),
        "reranking_cases": workloads, "phrase_cases": phrases,
        "predict_calls_total": getattr(lm.model, "calls", None),
        "predict_seconds_total": getattr(lm.model, "predict_seconds", None),
        "note": "Mac host timings, full vocab transfer and app-side Python search included. Load may compile/use cache; not true cold iPhone loading. No iPhone/keyboard memory claim."})
