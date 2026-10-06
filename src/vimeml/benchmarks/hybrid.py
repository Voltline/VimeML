"""Cache local LM scores, calibrate on independent development data, evaluate a fixed policy."""
import argparse
import json
import math
import re
import time
import unicodedata
from pathlib import Path

from vimeml.benchmarks.ajimee import DEVELOPMENT_FORMAT, ROOT, convert_items, json_bytes, sha
from vimeml.benchmarks.evaluate_ajimee import load_export, metrics, read_json, rerank, summarize

CACHE_FORMAT = "ime_hybrid_score_cache_v1"
POLICY_FORMAT = "ime_hybrid_policy_v1"
FORMULA = "azookey_score + lambda * contextual_log_probability_sum"
GRID = [0., .05, .1, .2, .3, .5, .75, 1., 1.5, 2., 3.]


def output_directory(path):
    if path.exists() and any(path.iterdir()):
        raise ValueError("Output is not empty; use a new --output to preserve previous results.")
    path.mkdir(parents=True, exist_ok=True)


def canonical(text):
    return "".join(unicodedata.normalize("NFKC", text).split())


def overlap_keys(rows):
    return ({(canonical(row["query"]), canonical(row["left_context"])) for row in rows},
            {canonical(row["left_context"] + answer) for row in rows for answer in row["answers"]})


def check_independence(development, excluded):
    dev_queries, dev_texts = overlap_keys(development)
    queries, texts = overlap_keys(excluded)
    if dev_queries & queries or dev_texts & texts:
        raise ValueError("Development examples overlap the excluded AJIMEE set (normalized query/context or context+answer).")


def prepare_development(input_path, output, excluded, labels_reviewed=False):
    raw = input_path.read_bytes()
    converted, mapping, stats = convert_items(json.loads(raw.decode("utf-8-sig")))
    keys, _ = overlap_keys(mapping)
    if len(keys) != len(mapping):
        raise ValueError("Duplicate development query/context pair.")
    if not excluded.exists():
        raise ValueError("Excluded AJIMEE source is missing; specify --exclude.")
    exclude_raw = excluded.read_bytes()
    _, exclude_mapping, _ = convert_items(json.loads(exclude_raw.decode("utf-8-sig")))
    check_independence(mapping, exclude_mapping)
    output_directory(output)
    files = {"evaluation_items.json": raw, "ajimee-input.json": json_bytes(converted), "case-map.json": json_bytes(mapping)}
    for name, content in files.items():
        (output / name).write_bytes(content)
    manifest = {"format": DEVELOPMENT_FORMAT, "status": "complete", "role": "development",
                "source_path": str(input_path.resolve()), "source_sha256": sha(raw), "stats": stats,
                "labels_reviewed": labels_reviewed,
                "label_status": "reviewed_by_user_assertion" if labels_reviewed else "provisional_synthetic_labels",
                "excluded_source_sha256": sha(exclude_raw),
                "files_sha256": {name: sha(content) for name, content in files.items()},
                "policy": "Synthetic development only; all text/references preserved. No candidate/model-score filtering. Reading and label quality require review."}
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def fingerprint(metadata, provenance):
    dictionaries = []
    for line in provenance["dictionary-versions.txt"].splitlines():
        match = re.match(r"^ ([0-9a-f]{40}) (\S+)", line)
        if not match:
            raise ValueError("Invalid/unpinned dictionary version record.")
        dictionaries.append({"commit": match[1], "path": match[2]})
    converter = provenance["converter-version.txt"].strip()
    if not re.fullmatch(r"[0-9a-f]{40}", converter) or not dictionaries:
        raise ValueError("Missing converter/dictionary commit.")
    return {"checkpoint_sha256": metadata["checkpoint_sha256"], "tokenizer_sha256": metadata["tokenizer_sha256"],
            "model_config": metadata["model"], "precision": metadata["precision"],
            "converter_commit": converter, "dictionaries": sorted(dictionaries, key=lambda row: row["path"]),
            "n_best": provenance["n_best"]}


def save_cache(benchmark, output, checkpoint, tokenizer, device, threads):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Cache output is not empty; choose a new --output.")
    manifest, provenance, rows = load_export(benchmark)
    import torch
    from vimeml.training.infer import JapaneseLM
    torch.set_num_threads(threads)
    start = time.perf_counter()
    lm = JapaneseLM(checkpoint, tokenizer, device)
    rerank(lm, rows)
    output_directory(output)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    (output / "scores.jsonl").write_bytes(content)
    cache = {"format": CACHE_FORMAT, "status": "complete", "model": lm.metadata,
             "benchmark_manifest": manifest, "export_provenance": provenance,
             "files_sha256": {"scores.jsonl": sha(content)}, "fingerprint": fingerprint(lm.metadata, provenance),
             "elapsed_seconds": time.perf_counter() - start}
    (output / "manifest.json").write_bytes(json_bytes(cache))
    return cache


def load_cache(directory):
    # Reuse the existing frozen AJIMEE evaluation without another LM forward pass.
    path = directory / "manifest.json"
    if path.exists():
        cache = read_json(path)
        if cache.get("format") != CACHE_FORMAT:
            raise ValueError("Unknown score cache format.")
    else:
        cache = read_json(directory / "metrics.json")
        if cache.get("format") != "ajimee_ime_metrics_v1":
            raise ValueError("Unknown evaluation cache format.")
    if cache.get("status") != "complete" or not cache.get("model"):
        raise ValueError("Expected complete model scores.")
    raw = (directory / "scores.jsonl").read_bytes()
    if sha(raw) != cache["files_sha256"]["scores.jsonl"]:
        raise ValueError("Cached scores changed.")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows or len(rows) != cache["benchmark_manifest"]["stats"]["cases"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Invalid score cache sample count/IDs.")
    for row in rows:
        texts = [c["text"] for c in row["candidates"]]
        if texts != row["orders"]["azookey"] or len(set(texts)) != len(texts):
            raise ValueError("Cached original candidate order changed.")
        if any(not math.isfinite(c["score"]) for c in row["candidates"]):
            raise ValueError("Nonfinite AzooKey score.")
        scores = row.get("lm_scores", {}).get("contextual")
        if "lm_context_sum" in row["fallbacks"]:
            if row["orders"]["lm_context_sum"] != texts:
                raise ValueError("Invalid whole-case fallback.")
        elif not scores or [c["text"] for c in scores["candidates"]] != texts:
            raise ValueError("Cached contextual scores do not align with original candidates.")
        else:
            for candidate in scores["candidates"]:
                probabilities = candidate["token_log_probabilities"]
                if not probabilities or len(probabilities) != candidate["scored_tokens"] or len(candidate["token_ids"]) != len(probabilities):
                    raise ValueError("Invalid cached token trace.")
                if not all(math.isfinite(v) for v in probabilities) or not math.isclose(sum(probabilities), candidate["log_probability_sum"], abs_tol=1e-6):
                    raise ValueError("Cached LM total differs from token trace.")
    cache["fingerprint"] = fingerprint(cache["model"], cache["export_provenance"])
    cache["cache_directory"] = str(directory.resolve())
    return cache, rows


def valid_weight(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("lambda must be finite and nonnegative.")
    return value


def hybrid_order(candidates, scores, weight):
    """No labels, normalization, length bonus or candidate filtering on this path."""
    valid_weight(weight)
    if weight == 0:
        return [c["text"] for c in candidates], [c["score"] for c in candidates]
    if [c["text"] for c in candidates] != [c["text"] for c in scores]:
        raise ValueError("Hybrid score alignment mismatch.")
    values = [c["score"] + weight * s["log_probability_sum"] for c, s in zip(candidates, scores)]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite combined score.")
    indices = sorted(range(len(candidates)), key=lambda i: values[i], reverse=True)
    return [candidates[i]["text"] for i in indices], values


def add_hybrid(rows, weight, details=True):
    valid_weight(weight)
    for row in rows:
        if "lm_context_sum" in row["fallbacks"] and weight > 0:
            row["orders"]["hybrid"] = list(row["orders"]["azookey"])
            row["fallbacks"]["hybrid"] = row["fallbacks"]["lm_context_sum"]
            values = None
        else:
            scored = row.get("lm_scores", {}).get("contextual", {}).get("candidates", [])
            row["orders"]["hybrid"], values = hybrid_order(row["candidates"], scored, weight)
            row["fallbacks"].pop("hybrid", None)
        if details:
            row["hybrid_scores_original_order"] = values


def script_profile(text):
    return {"kanji": sum('\u3400' <= c <= '\u9fff' for c in text),
            "hiragana": sum('\u3040' <= c <= '\u309f' for c in text),
            "katakana": sum('\u30a0' <= c <= '\u30ff' for c in text),
            "latin": sum(c.isascii() and c.isalpha() for c in text), "digits": sum(c.isdecimal() for c in text)}


def diagnostics(row):
    scored = row.get("lm_scores", {}).get("contextual", {}).get("candidates", [])
    prefixes, fallback = [], []
    for i, candidate in enumerate(scored):
        byte_count = sum(piece.startswith("<0x") for piece in candidate["pieces"])
        if byte_count:
            fallback.append({"text": candidate["text"], "byte_tokens": byte_count})
        for other in scored[i + 1:]:
            a, b = candidate, other
            if len(a["token_ids"]) > len(b["token_ids"]):
                a, b = b, a
            if len(a["token_ids"]) < len(b["token_ids"]) and b["token_ids"][:len(a["token_ids"])] == a["token_ids"]:
                prefixes.append({"shorter": a["text"], "longer": b["text"]})
    original, selected = row["orders"]["azookey"], row["orders"]["hybrid"]
    before, after = original[0] if original else "", selected[0] if selected else ""
    return {"strict_token_prefix_pairs": prefixes, "byte_fallback_candidates": fallback,
            "boundary_retokenized_candidates": sum(c["boundary_retokenized"] for c in scored),
            "top1_changed": before != after, "before_script_profile": script_profile(before),
            "after_script_profile": script_profile(after),
            "script_profile_changed": script_profile(before) != script_profile(after)}


def calibrate(cache, rows, grid):
    if cache["benchmark_manifest"].get("role") != "development" or cache["benchmark_manifest"]["format"] != DEVELOPMENT_FORMAT:
        raise ValueError("Automatic lambda selection requires an independent development cache; AJIMEE cannot be tuned.")
    if not cache["benchmark_manifest"].get("labels_reviewed"):
        raise ValueError("Development readings/references need review before calibration; prepare a reviewed version with --labels-reviewed.")
    if not grid or len(set(grid)) != len(grid):
        raise ValueError("Expected a nonempty grid of distinct lambda values.")
    records = []
    for weight in grid:
        valid_weight(weight)
        add_hybrid(rows, weight, details=False)
        records.append({"lambda": weight, "metrics": metrics(rows, "hybrid")})
    chosen = min(records, key=lambda record: (-record["metrics"]["top1_correct"],
                                            record["metrics"]["regressed_vs_azookey"], record["lambda"]))
    return {"format": POLICY_FORMAT, "status": "development_selected", "formula": FORMULA,
            "lambda": chosen["lambda"], "fingerprint": cache["fingerprint"],
            "selection_rule": "Highest development Top-1; ties prefer fewer regressions; remaining ties prefer smaller lambda.",
            "development_source_sha256": cache["benchmark_manifest"]["source_sha256"],
            "development_scores_sha256": cache["files_sha256"]["scores.jsonl"],
            "excluded_source_sha256": cache["benchmark_manifest"]["excluded_source_sha256"],
            "development_overlap_keys": {"queries": sorted([list(key) for key in overlap_keys(rows)[0]]),
                                         "texts": sorted(overlap_keys(rows)[1])},
            "grid": records}


def validate_policy(policy, cache, rows):
    if policy.get("format") != POLICY_FORMAT or policy.get("status") != "development_selected" or policy.get("formula") != FORMULA:
        raise ValueError("Invalid selected policy.")
    valid_weight(policy["lambda"])
    if policy["fingerprint"] != cache["fingerprint"]:
        raise ValueError("Policy model/tokenizer/converter/dictionary/precision/n_best differs from this cache.")
    if policy["development_source_sha256"] == cache["benchmark_manifest"]["source_sha256"]:
        raise ValueError("Selected-policy evaluation must use different data from calibration.")
    queries, texts = overlap_keys(rows)
    dev = policy["development_overlap_keys"]
    if queries & {tuple(key) for key in dev["queries"]} or texts & set(dev["texts"]):
        raise ValueError("Evaluation examples overlap calibration data.")
    return policy["lambda"]


def evaluate(cache, rows, weight, output, policy=None):
    valid_weight(weight)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty; choose a new --output.")
    add_hybrid(rows, weight)
    changed = []
    for row in rows:
        row["hybrid_diagnostics"] = diagnostics(row)
        if row["hybrid_diagnostics"]["top1_changed"]:
            before, after = row["orders"]["azookey"][0], row["orders"]["hybrid"][0]
            changed.append({"id": row["id"], "context": row["left_context"], "query": row["query"],
                            "answers": row["answers"], "before": before, "after": after,
                            "before_correct": before in row["answers"], "after_correct": after in row["answers"],
                            "diagnostics": row["hybrid_diagnostics"]})
    report = {"format": "ime_hybrid_evaluation_v1", "status": "complete", "formula": FORMULA, "lambda": weight,
              "coefficient_source": "independent_development_policy" if policy else "user_specified_not_calibrated",
              "policy": policy, "score_cache": cache, "metrics": summarize(rows),
              "diagnostics": {"strict_prefix_cases": sum(bool(r["hybrid_diagnostics"]["strict_token_prefix_pairs"]) for r in rows),
                              "byte_fallback_cases": sum(bool(r["hybrid_diagnostics"]["byte_fallback_candidates"]) for r in rows),
                              "top1_changed_cases": len(changed),
                              "script_profile_changed_cases": sum(r["hybrid_diagnostics"]["script_profile_changed"] for r in rows)},
              "policy_notes": "Same real candidate pool; stable ties; raw scores, no EOS/length normalization/extra filters. Whole-case LM fallback. Diagnostics never alter ranking. Strict reference matching; training overlap unknown."}
    output_directory(output)
    (output / "scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (output / "changed-cases.json").write_bytes(json_bytes(changed))
    report["files_sha256"] = {name: sha((output / name).read_bytes()) for name in ("scores.jsonl", "changed-cases.json")}
    (output / "metrics.json").write_bytes(json_bytes(report))
    lines = ["# AzooKey + LM组合评分", "", f"λ = {weight:g}；系数来源：{report['coefficient_source']}。", "",
             "同一冻结候选池；并列保持原顺序；超长/无法完整计分时整条回退。表记诊断不是自动语义判分。", "",
             "| 策略 | 样本数 | Top-1 | Top-5 | MinCER | 纠正 / 改坏 | 回退 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method in ("azookey", "lm_context_sum", "lm_no_context_sum", "hybrid"):
        m = report["metrics"]["all"][method]
        lines.append(f"| {method} | {m['cases']} | {m['top1_correct']} ({m['top1_accuracy']:.1%}) | {m['top5_correct']} ({m['top5_accuracy']:.1%}) | "
                     f"{m['mean_min_cer']:.5f} | {m['corrected_vs_azookey']} / {m['regressed_vs_azookey']} | {m['fallback_cases']} |")
    lines += ["", "完整分组、版本与hash见metrics.json，逐候选组合分数与诊断见scores.jsonl，所有首选变化见changed-cases.json。", ""]
    (output / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-dev", help="Prepare reviewed DeepSeek JSON for the same Mac export command.")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmarks/ime-dev-v1")
    prepare.add_argument("--exclude", type=Path, default=ROOT / "artifacts/benchmarks/ajimee-jwtd-v2-v1/evaluation_items.json")
    prepare.add_argument("--labels-reviewed", action="store_true", help="Assert readings and acceptable references have been independently reviewed.")
    score = sub.add_parser("score", help="Compute local LM once for a real candidate export.")
    score.add_argument("--benchmark", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/models/tiny-ja-v1/best.pt")
    score.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts/tokenizers/ja-unigram-16k-v1")
    score.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    score.add_argument("--threads", type=int, default=4)
    tune = sub.add_parser("tune", help="Select lambda only on reviewed independent development data.")
    tune.add_argument("--scores", type=Path, required=True)
    tune.add_argument("--grid", type=float, nargs="+", default=GRID)
    tune.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("evaluate", help="Evaluate one fixed lambda or a saved development policy.")
    run.add_argument("--scores", type=Path, default=ROOT / "outputs/ime-eval/tiny-ja-v1-ajimee")
    option = run.add_mutually_exclusive_group(required=True)
    option.add_argument("--lambda", type=float, dest="weight")
    option.add_argument("--policy", type=Path)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-dev":
            result = prepare_development(args.input, args.output, args.exclude, args.labels_reviewed)
            print(json.dumps(result["stats"], indent=2))
            print(f"Mac input: {args.output.resolve() / 'ajimee-input.json'}")
        elif args.command == "score":
            if args.threads < 1:
                raise ValueError("threads must be positive.")
            save_cache(args.benchmark, args.output, args.checkpoint, args.tokenizer, args.device, args.threads)
            print(f"Score cache: {args.output.resolve()}")
        else:
            cache, rows = load_cache(args.scores)
            if args.command == "tune":
                result = calibrate(cache, rows, args.grid)
                output_directory(args.output)
                (args.output / "policy.json").write_bytes(json_bytes(result))
                print(f"Development-selected lambda: {result['lambda']:g}")
                print(f"Policy: {args.output.resolve() / 'policy.json'}")
            else:
                policy = read_json(args.policy) if args.policy else None
                weight = validate_policy(policy, cache, rows) if policy else args.weight
                report = evaluate(cache, rows, weight, args.output, policy)
                print(json.dumps(report["metrics"]["all"]["hybrid"], indent=2))
                print(f"Report: {args.output.resolve() / 'results.md'}")
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
