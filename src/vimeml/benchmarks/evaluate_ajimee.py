"""Evaluate frozen AzooKey candidates, preserving official references and order."""
import argparse
import json
import math
import time
from pathlib import Path

from vimeml.benchmarks.ajimee import DEVELOPMENT_FORMAT, FORMAT, ROOT, convert_items, json_bytes, sha
from vimeml.benchmarks.expanded_ime import FORMAT as EXPANDED_FORMAT, load_expanded_export


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_export(directory):
    manifest = read_json(directory / "manifest.json")
    if manifest.get("format") == EXPANDED_FORMAT:
        return load_expanded_export(directory, manifest)
    if manifest.get("format") not in (FORMAT, DEVELOPMENT_FORMAT) or manifest.get("status") != "complete":
        raise ValueError("Expected a complete prepared AJIMEE manifest.")
    for name, digest in manifest["files_sha256"].items():
        if sha((directory / name).read_bytes()) != digest:
            raise ValueError(f"Prepared file changed: {name}.")
    source = directory / "evaluation_items.json"
    if sha(source.read_bytes()) != manifest["source_sha256"]:
        raise ValueError("Source snapshot hash mismatch.")
    inputs, mapping, stats = convert_items(read_json(source))
    if inputs != read_json(directory / "ajimee-input.json") or mapping != read_json(directory / "case-map.json") or stats != manifest["stats"]:
        raise ValueError("Prepared input/mapping does not match source snapshot.")
    exported = directory / "ajimee-results"
    if inputs != read_json(exported / "ajimee-input.json"):
        raise ValueError("Mac input differs from frozen local input.")
    raw = read_json(exported / "azookey-candidates.json")
    items, n_best = raw.get("items"), raw.get("n_best")
    if type(n_best) is not int or n_best < 1 or not isinstance(items, list) or len(items) != len(inputs):
        raise ValueError("Invalid n_best or exported sample count.")
    rows = []
    for position, (expected, item, case) in enumerate(zip(inputs, items, mapping)):
        if any(item.get(key) != expected[value] for key, value in
               (("query", "query"), ("answers", "answer"), ("left_context", "left_context"))):
            raise ValueError(f"Row {position}: query/context/references misaligned.")
        if item.get("right_context") not in (None, ""):
            raise ValueError(f"Row {position}: unexpected right context.")
        candidates = item.get("outputs")
        if not isinstance(candidates, list) or len(candidates) > n_best:
            raise ValueError(f"Row {position}: invalid candidate count.")
        texts = []
        for candidate in candidates:
            text, score = candidate.get("text"), candidate.get("score")
            if not isinstance(text, str) or not text or type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError(f"Row {position}: invalid candidate text/score.")
            texts.append(text)
        if len(set(texts)) != len(texts):
            raise ValueError(f"Row {position}: duplicate candidate text.")
        rank = next((i for i, text in enumerate(texts) if text in case["answers"]), -1)
        if item.get("max_rank") != rank:
            raise ValueError(f"Row {position}: CLI rank inconsistent with candidates.")
        rows.append({**case, "candidates": candidates, "orders": {"azookey": texts}, "fallbacks": {}})
    # Preserve provenance verbatim; the CLI output does not embed every flag.
    version_files = ["converter-version.txt", "dictionary-versions.txt", "swift-version.txt"]
    provenance = {name: (exported / name).read_text(encoding="utf-8-sig") for name in version_files}
    for line in provenance["dictionary-versions.txt"].splitlines():
        if not line.startswith(" "):
            raise ValueError("Dictionary submodule is uninitialized, conflicted or differs from its pinned commit.")
    provenance["files_sha256"] = {name: sha((exported / name).read_bytes()) for name in
                                  ["ajimee-input.json", "azookey-candidates.json", *version_files]}
    provenance["n_best"] = n_best
    provenance["cli_execution_seconds"] = raw.get("execution_time")
    provenance["flag_provenance"] = "Export requested with typo mode off and no Zenzai; CLI JSON does not embed these flags."
    return manifest, provenance, rows


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def min_cer(answers, hypothesis):
    return min(edit_distance(answer, hypothesis) / len(answer) for answer in answers)


def metrics(rows, method):
    count = len(rows)
    if not count:
        return {"cases": 0}
    top1 = top5 = covered = corrected = regressed = 0
    cer = reciprocal = 0.0
    for row in rows:
        order, answers = row["orders"][method], row["answers"]
        ranks = [i + 1 for i, text in enumerate(order) if text in answers]
        hit = bool(ranks) and min(ranks) == 1
        original = bool(row["orders"]["azookey"]) and row["orders"]["azookey"][0] in answers
        top1 += hit
        top5 += bool(ranks) and min(ranks) <= 5
        covered += bool(ranks)
        reciprocal += 1 / min(ranks) if ranks else 0
        cer += min_cer(answers, order[0] if order else "")
        corrected += hit and not original
        regressed += original and not hit
    return {"cases": count, "top1_correct": top1, "top1_accuracy": top1 / count,
            "top5_correct": top5, "top5_accuracy": top5 / count, "candidate_pool_covered": covered,
            "candidate_pool_recall": covered / count, "covered_top1_accuracy": top1 / covered if covered else None,
            "mean_min_cer": cer / count, "mrr": reciprocal / count,
            "corrected_vs_azookey": corrected, "regressed_vs_azookey": regressed,
            "fallback_cases": sum(method in row["fallbacks"] for row in rows)}


def eligibility(lm, context, texts):
    if not texts:
        return "empty_candidates"
    context_ids = lm.processor.encode(context, out_type=int)
    if lm.processor.decode(context_ids) != context:
        return "context_roundtrip_mismatch"
    if len(context_ids) + 1 > lm.model.config.context_length:
        return "context_length"
    for text in texts:
        ids = lm.processor.encode(context + text, out_type=int)
        if lm.processor.decode(ids) != context + text:
            return "candidate_roundtrip_mismatch"
        if not ids or len(ids) > lm.model.config.context_length:
            return "context_length"
    return None


def rerank(lm, rows):
    from vimeml.training.evaluate_ime import score_candidates
    for position, row in enumerate(rows):
        texts = row["orders"]["azookey"]
        row["lm_scores"] = {}
        for field, context, methods in (
            ("contextual", row["left_context"], ("lm_context_sum", "lm_context_mean_secondary")),
            ("context_free", "", ("lm_no_context_sum",)),
        ):
            reason = eligibility(lm, context, texts)
            if reason:
                for method in methods:
                    row["orders"][method] = list(texts)
                    row["fallbacks"][method] = reason
                continue
            scored = score_candidates(lm, context, texts)
            if any(not math.isfinite(c[key]) for c in scored["candidates"] for key in
                   ("log_probability_sum", "log_probability_mean")):
                raise ValueError(f"Nonfinite LM score: {row['id']}.")
            row["lm_scores"][field] = scored
            for method in methods:
                key = "log_probability_mean" if method.endswith("secondary") else "log_probability_sum"
                # Stable sort: ties preserve the actual AzooKey candidate order.
                row["orders"][method] = [c["text"] for c in sorted(scored["candidates"], key=lambda c: c[key], reverse=True)]
        if (position + 1) % 25 == 0 or position == len(rows) - 1:
            print(f"Evaluated {position + 1}/{len(rows)}", flush=True)


def summarize(rows):
    groups = {"all": rows, "with_context": [r for r in rows if r["left_context"]],
              "without_context": [r for r in rows if not r["left_context"]]}
    if "lm_context_sum" in rows[0]["orders"]:
        groups["contextually_scorable"] = [r for r in rows if "lm_context_sum" not in r["fallbacks"]]
    return {name: {method: metrics(items, method) for method in rows[0]["orders"]} for name, items in groups.items()}


def write_summary(path, report, rows):
    lines = ["# AJIMEE / AzooKey + Tiny LM", "",
             "完整输入、全部可接受答案、真实候选池；不注入答案，不正规化文字，不加EOS。",
             "上下文使用官方给定左文，可能跨句；联合分词的公共前缀后logP是候选评分代理，不是精确字符串条件概率。",
             "主指标为sum，mean仅作预先声明的次要诊断；没有在本评测集调组合系数。并列保持AzooKey原顺序。",
             "任何候选超出128-token评分窗口时，该策略整条回退原排序；空候选计失败，仍保留完整分母。",
             "正确答案未进入候选池的样本也保留；训练语料与公开评测文本重叠未知，分句DataLoader的test split没有用于本轮评测。", ""]
    if report.get('benchmark_manifest', {}).get('format') == EXPANDED_FORMAT:
        split = report['benchmark_manifest']['split']
        lines = [f'# Expanded IME / {split}', '',
                 '真实 AzooKey 导出与冻结候选池；当前双词典检查的参考标签仍为草稿，指标是初步诊断，不是正式母语 gold 结果。',
                 '主分数：context+candidate 联合分词，公共 token 前缀后的 full-vocabulary logP sum；不加 EOS、不截断、并列保留原顺序。',
                 '保留召回失败与回退的完整分母；mean 仅作预先声明的次要诊断。', '']
    lines.extend([
        "| 样本组 | 策略 | 样本数 | Top-1 | Top-5 | 候选池覆盖 | MinCER | 纠正 / 改坏 | 回退 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for group, methods in report["metrics"].items():
        for method, item in methods.items():
            if not item["cases"]:
                continue
            lines.append(f"| {group} | {method} | {item['cases']} | {item['top1_correct']} ({item['top1_accuracy']:.1%}) | "
                         f"{item['top5_correct']} ({item['top5_accuracy']:.1%}) | {item['candidate_pool_covered']} ({item['candidate_pool_recall']:.1%}) | "
                         f"{item['mean_min_cer']:.4f} | {item['corrected_vs_azookey']} / {item['regressed_vs_azookey']} | {item['fallback_cases']} |")
    lines += ["", "候选池覆盖是这次冻结候选下精确命中的上限，不等于理论上能达到的实际模型性能。开发机耗时不是iOS延迟。",
              "候选原分数与全部LM token分数见scores.jsonl；changed-cases.json记录纠正/改坏样本；metrics.json记录输入、候选、版本和模型hash。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "artifacts/benchmarks/ajimee-jwtd-v2-v1")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/models/tiny-ja-v1/best.pt")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts/tokenizers/ja-unigram-16k-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ime-eval/tiny-ja-v1-ajimee")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output is not empty; choose a new --output directory to preserve previous results.")
    started = time.perf_counter()
    manifest, provenance, rows = load_export(args.benchmark)
    model = None
    if not args.baseline_only:
        import torch
        from vimeml.training.infer import JapaneseLM
        torch.set_num_threads(args.threads)
        lm = JapaneseLM(args.checkpoint, args.tokenizer, args.device)
        model = lm.metadata
        rerank(lm, rows)
    report = {"format": "ajimee_ime_metrics_v1", "status": "complete", "model": model,
              "benchmark_manifest": manifest, "export_provenance": provenance, "metrics": summarize(rows),
              "policy": "Original queries/references/candidates; supplied context; suffix logP sum primary; mean secondary; stable ties; whole-case fallback; no EOS or truncation.",
              "training_overlap": "Unknown; no corpus overlap audit performed.", "test_split_used": False,
              "empty_candidate_cases": sum(not r["candidates"] for r in rows),
              "elapsed_seconds": time.perf_counter() - started}
    if manifest.get('format') == EXPANDED_FORMAT:
        report.update({'labels_formal_gold':False,'result_role':'provisional_expanded_'+manifest['split'],
                       'test_split_used':manifest['source_corpus_split']=='test',
                       'training_overlap':'Corpus source groups come from held-out validation/test; near-duplicate overlap not fully audited.'})
    changed = []
    for row in rows:
        original = row["orders"]["azookey"]
        before = bool(original) and original[0] in row["answers"]
        for method, order in row["orders"].items():
            after = bool(order) and order[0] in row["answers"]
            if before != after:
                changed.append({"id": row["id"], "method": method, "change": "corrected" if after else "regressed",
                                "query": row["query"], "context": row["left_context"], "answers": row["answers"],
                                "before": original[0], "after": order[0]})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "scores.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (args.output / "changed-cases.json").write_bytes(json_bytes(changed))
    report["files_sha256"] = {name: sha((args.output / name).read_bytes()) for name in ("scores.jsonl", "changed-cases.json")}
    (args.output / "metrics.json").write_bytes(json_bytes(report))
    write_summary(args.output / "results.md", report, rows)
    print(json.dumps(report["metrics"]["all"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output.resolve() / 'results.md'}")


if __name__ == "__main__":
    main()
