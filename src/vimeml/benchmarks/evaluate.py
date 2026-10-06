"""Tie-aware accuracy of a frozen synthetic set; local LM, no API calls."""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from vimeml.benchmarks.generate import VERSION, jsonl
from vimeml.data.build import file_sha
from vimeml.review.prepare import write_json
from vimeml.training.evaluate_ime import score_candidates
from vimeml.training.infer import JapaneseLM, ROOT


def load_benchmark(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != VERSION or manifest.get("status") != "complete":
        raise ValueError("Benchmark generation/verification is not complete.")
    path = directory / "benchmark.jsonl"
    if file_sha(path) != manifest["benchmark_sha256"]:
        raise ValueError("Frozen benchmark changed; retain the original and version edits separately.")
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = set()
    for case in cases:
        if case["id"] in ids or case["task"] not in {"homophone", "phrase_contrast"}:
            raise ValueError("Duplicate ID or unknown task.")
        ids.add(case["id"])
        candidates, gold = case["candidates"], case["acceptable"]
        if len(candidates) < 2 or len(set(candidates)) != len(candidates) or len(gold) != 1 or gold[0] not in candidates:
            raise ValueError("Expected distinct candidates and one accepted gold.")
        quality = case["quality"]
        if quality["status"] != "clear" or quality["confidence"] != "high" or quality["acceptable"] != gold:
            raise ValueError("Unverified or ambiguous case in accepted dataset.")
    if not cases or len(cases) != manifest["stats"]["accepted_cases"]:
        raise ValueError("Empty benchmark or count mismatch.")
    return manifest, cases


def accuracy(rows, field="contextual", score="log_probability_sum", tolerance=1e-6):
    """Ties receive expected uniform tie-breaking credit, never array-order wins."""
    top1 = top2 = reciprocal = 0.0
    ties = 0
    for row in rows:
        candidates = row[field]["candidates"]
        values = [candidate[score] for candidate in candidates]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Nonfinite evaluation score.")
        gold = next(candidate[score] for candidate in candidates if candidate["text"] == row["acceptable"][0])
        higher = sum(value > gold + tolerance for value in values)
        equal = sum(abs(value - gold) <= tolerance for value in values)
        top1 += 1 / equal if higher == 0 else 0
        top2 += max(0, min(equal, 2 - higher)) / equal
        reciprocal += sum(1 / (higher + rank) for rank in range(1, equal + 1)) / equal
        ties += sum(abs(value - max(values)) <= tolerance for value in values) > 1
    count = len(rows)
    return {"cases": count, "top1_credit": top1, "top1_accuracy": top1 / count,
        "top2_accuracy": top2 / count, "mrr": reciprocal / count, "top_score_ties": ties}


def summarize(rows):
    if not rows:
        return {"cases": 0}
    result = {"contextual_sum_primary": accuracy(rows),
        "context_free_sum": accuracy(rows, "context_free"),
        "contextual_mean_secondary": accuracy(rows, score="log_probability_mean"),
        "shorter_token_baseline": accuracy(rows, "length_baseline", "negative_tokens"),
        "random_expected_top1": sum(1 / len(row["candidates"]) for row in rows) / len(rows)}
    matched = [row for row in rows if len({candidate["scored_tokens"] for candidate in row["contextual"]["candidates"]}) == 1]
    result["equal_scored_token_length"] = accuracy(matched) if matched else {"cases": 0}
    result["gold_strictly_shortest_scored_tokens_cases"] = sum(next(candidate["scored_tokens"] for candidate in row["contextual"]["candidates"]
        if candidate["text"] == row["acceptable"][0]) < min(candidate["scored_tokens"] for candidate in row["contextual"]["candidates"]
        if candidate["text"] != row["acceptable"][0]) for row in rows)
    result["boundary_retokenized_cases"] = sum(any(candidate["boundary_retokenized"] for candidate in row["contextual"]["candidates"]) for row in rows)
    return result


def evaluate(lm, cases):
    rows, priors = [], {}
    for index, case in enumerate(cases):
        key = tuple(case["candidates"])
        if key not in priors:
            priors[key] = score_candidates(lm, "", case["candidates"])
        contextual = score_candidates(lm, case["context"], case["candidates"])
        # Candidate-only token lengths: a crude baseline exposing tokenizer
        # length preference, not a Japanese language or AzooKey baseline.
        lengths = [{"text": text, "negative_tokens": -len(lm.processor.encode(text, out_type=int))}
                   for text in case["candidates"]]
        rows.append({**case, "contextual": contextual, "context_free": priors[key],
            "length_baseline": {"candidates": lengths}})
        if (index + 1) % 100 == 0 or index == len(cases) - 1:
            print(f"Evaluated {index + 1}/{len(cases)}", flush=True)
    by_task, by_category = defaultdict(list), defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
        by_category[row["task"] + ":" + row["category"]].append(row)
    metrics = {"all": summarize(rows), "by_task": {task: summarize(items) for task, items in by_task.items()},
        "by_category": {category: summarize(items) for category, items in by_category.items()}}
    return rows, metrics


def write_summary(path, report):
    lines = ["# 合成输入法评测", "",
        "这是AI生成并经另一请求盲核验的合成集准确率。不是实际AzooKey候选排序准确率，也不是开放式短语联想质量/命中率。",
        "生成前不使用本地LM分数选样；训练文本重叠未知。test split未使用；是否已人工抽检见下方。", "",
        f"人工抽检完成标记：{report['benchmark_manifest'].get('manual_review_done', False)}。未抽检时标签质量仍是暂定。", "",
        "| 任务 | 样本数 | 上下文 sum（主指标） | 无上下文 sum | 上下文 mean（次要） | 短token基线 |", "| --- | --- | --- | --- | --- | --- |"]
    for task, metrics in report["metrics"]["by_task"].items():
        primary = metrics["contextual_sum_primary"]
        lines.append(f"| {task} | {primary['cases']} | {primary['top1_accuracy']:.2%} | {metrics['context_free_sum']['top1_accuracy']:.2%} | {metrics['contextual_mean_secondary']['top1_accuracy']:.2%} | {metrics['shorter_token_baseline']['top1_accuracy']:.2%} |")
    lines += ["", "同音词和短语二选一分开解读。并列最高分按均匀随机打破并列计分，不靠候选顺序赢得分数。",
        "主评分、次要评分及长度基线预先定义；不按本轮结果重选最佳评分方案。等token长度子集、分主题结果见metrics.json。",
        "完整原始分数见scores.jsonl，按固定hash选取的人工检查材料见数据目录中的manual-audit.json（同时含通过/隔离样本）。",
        "其他checkpoint使用同一冻结集、单独输出目录；该集一旦用于调整策略就成为开发集，最终评估另留新样本。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "artifacts/benchmarks/ime-synthetic-v1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/models/tiny-ja-v1/best.pt")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts/tokenizers/ja-unigram-16k-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ime-eval/tiny-ja-v1-synthetic")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive.")
    manifest, cases = load_benchmark(args.benchmark)
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    lm = JapaneseLM(args.checkpoint, args.tokenizer, args.device)
    rows, metrics = evaluate(lm, cases)
    args.output.mkdir(parents=True, exist_ok=True)
    jsonl(args.output / "scores.jsonl", rows)
    report = {"format": "synthetic_ime_metrics_v1", "model": lm.metadata,
        "benchmark": str(args.benchmark.resolve()), "benchmark_manifest": manifest,
        "policy": "Joint tokenization; common-prefix suffix logP sum, no EOS; mean diagnostic; tie-aware accuracy",
        "test_split_used": False, "metrics": metrics, "elapsed_seconds": time.perf_counter() - started}
    write_json(args.output / "metrics.json", report)
    write_summary(args.output / "results.md", report)
    print(json.dumps(metrics["by_task"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output / 'results.md'}")


if __name__ == "__main__":
    main()
