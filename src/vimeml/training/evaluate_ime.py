"""Read-only synthetic IME diagnostic; no kana search, training or API calls."""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from vimeml.training.data import file_sha, write_json
from vimeml.training.infer import JapaneseLM, ROOT, filtered_logits


def common_prefix_length(sequences):
    count = 0
    for values in zip(*sequences):
        if len(set(values)) != 1:
            break
        count += 1
    return count


@torch.inference_mode()
def score_candidates(lm, context, candidates):
    """Joint tokenization, common-prefix cancellation, no candidate-final EOS.

    A boundary can retokenize. This is a token suffix likelihood proxy, not
    an exact string probability P(candidate|context). Independently encoding
    the candidate would introduce incorrect SentencePiece boundaries.
    """
    if not candidates or len(set(candidates)) != len(candidates) or any(not s for s in candidates):
        raise ValueError("Candidates must be distinct nonempty strings.")
    context_ids = lm.prefix_ids(context)
    sequences = []
    for candidate in candidates:
        content = lm.processor.encode(context + candidate, out_type=int)
        if lm.processor.decode(content) != context + candidate:
            raise ValueError("Candidate does not roundtrip with tokenizer.")
        sequence = [lm.special["bos"], *content]
        if not 1 <= len(sequence) - 1 <= lm.model.config.context_length:
            raise ValueError("Candidate exceeds context length; no silent truncation.")
        sequences.append(sequence)
    common = common_prefix_length([context_ids, *sequences])
    if common < 1 or any(len(seq) <= common for seq in sequences):
        raise ValueError("Candidate suffix must contain prediction targets.")
    length = max(len(seq) - 1 for seq in sequences)
    inputs = torch.full((len(sequences), length), lm.special["pad"], device=lm.device, dtype=torch.long)
    targets = inputs.clone()
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    for row, seq in enumerate(sequences):
        inputs[row, :len(seq) - 1] = torch.tensor(seq[:-1], device=lm.device)
        targets[row, :len(seq) - 1] = torch.tensor(seq[1:], device=lm.device)
        mask[row, common - 1:len(seq) - 1] = True
    # Original full vocabulary probabilities, not sampling-filtered probabilities.
    log_probs = lm.model(inputs).float().log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    result = []
    for row, (candidate, seq) in enumerate(zip(candidates, sequences)):
        values = log_probs[row][mask[row]].cpu().tolist()
        suffix = seq[common:]
        result.append({"text": candidate, "log_probability_sum": sum(values),
            "log_probability_mean": sum(values) / len(values), "scored_tokens": len(values),
            "token_ids": suffix, "pieces": [lm.processor.id_to_piece(token) for token in suffix],
            "token_log_probabilities": values,
            "boundary_retokenized": seq[:len(context_ids)] != context_ids})
    return {"common_prefix_tokens_including_bos": common,
            "context_tokens_including_bos": len(context_ids), "candidates": result}


def ranking_metrics(rows, score="log_probability_sum", field="contextual"):
    hits1 = hits2 = reciprocal = corrected = regressed = 0
    groups = defaultdict(list)
    random_top1 = 0.0
    for row in rows:
        ordered = sorted(row[field]["candidates"], key=lambda item: item[score], reverse=True)
        ranks = [i + 1 for i, item in enumerate(ordered) if item["text"] in row["acceptable"]]
        rank = min(ranks)
        hits1 += rank == 1
        hits2 += rank <= 2
        reciprocal += 1 / rank
        random_top1 += len(row["acceptable"]) / len(ordered)
        groups[row["group"]].append(rank == 1)
        prior = max(row["context_free"]["candidates"], key=lambda item: item[score])["text"] in row["acceptable"]
        corrected += rank == 1 and not prior
        regressed += rank != 1 and prior
    count = len(rows)
    return {"cases": count, "top1_correct": hits1, "top1_fraction": hits1 / count,
        "top2_correct": hits2, "top2_fraction": hits2 / count, "mrr": reciprocal / count,
        "random_expected_top1_fraction": random_top1 / count,
        "all_contexts_correct_groups": sum(all(values) for values in groups.values()),
        "groups": len(groups), "corrected_vs_context_free": corrected, "regressed_vs_context_free": regressed}


@torch.inference_mode()
def next_batch(lm, sequences):
    """Batch independent active beams/samples; gather before right padding."""
    lengths = [len(seq) for seq in sequences]
    if not sequences or max(lengths) > lm.model.config.context_length:
        raise ValueError("Invalid active generation context.")
    inputs = torch.full((len(sequences), max(lengths)), lm.special["pad"], device=lm.device, dtype=torch.long)
    for row, seq in enumerate(sequences):
        inputs[row, :len(seq)] = torch.tensor(seq, device=lm.device)
    logits = lm.model(inputs).float()
    return logits[torch.arange(len(sequences), device=lm.device),
                  torch.tensor(lengths, device=lm.device) - 1]


def continuation(lm, prompt, prefix, new_ids, log_probability=None, stop="max_new_tokens"):
    text = lm.processor.decode([*prefix[1:], *new_ids])
    preserved = text.startswith(prompt)
    raw = text[len(prompt):] if preserved else text
    visible = raw.lstrip(" 、，,\n\t").rstrip("。！？!?")
    flags = []
    if not preserved:
        flags.append("prefix_changed")
    if "\ufffd" in raw:
        flags.append("replacement_character")
    if not visible:
        flags.append("empty")
    # Diagnostic only: duplicated trigrams can also be legitimate Japanese.
    if any(visible.count(visible[i:i + 3]) >= 3 for i in range(max(0, len(visible) - 2))):
        flags.append("repeated_trigram_proxy")
    return {"text": raw, "comparison_text": visible, "new_token_ids": new_ids,
            "log_probability_sum": log_probability, "stop_reason": stop, "flags": flags}


def stop_reason(lm, prompt, prefix, ids):
    if ids[-1] == lm.special["eos"]:
        return "eos"
    text = lm.processor.decode([*prefix[1:], *ids])
    if text.startswith(prompt) and text[len(prompt):].strip().endswith(tuple("。！？!?")):
        return "sentence_punctuation"
    return None


def unique_suggestions(items, count):
    result, seen = [], set()
    for item in items:
        key = item["comparison_text"]
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) == count:
            break
    return result


def generate_samples(lm, prompt, max_tokens, seed, attempts=8):
    """Greedy plus independent seeded samples in the same forward batches."""
    prefix = lm.prefix_ids(prompt)
    sequences = [[] for _ in range(attempts + 1)]
    generators = [torch.Generator(device=lm.device).manual_seed(seed + i) for i in range(attempts)]
    active = list(range(attempts + 1))
    stops = ["max_new_tokens"] * len(sequences)
    for _ in range(max_tokens):
        feasible = [i for i in active if len(prefix) + len(sequences[i]) <= lm.model.config.context_length]
        for i in set(active) - set(feasible):
            stops[i] = "context_limit"
        if not feasible:
            break
        logits = next_batch(lm, [[*prefix, *sequences[i]] for i in feasible])
        active = []
        for row, i in enumerate(feasible):
            if i == 0:
                token = int(filtered_logits(logits[row], lm.forbidden).argmax())
            else:
                distribution = filtered_logits(logits[row], lm.forbidden, .8, 50, .9).softmax(-1)
                token = int(torch.multinomial(distribution, 1, generator=generators[i - 1]))
            sequences[i].append(token)
            reason = stop_reason(lm, prompt, prefix, sequences[i])
            if reason:
                stops[i] = reason
            else:
                active.append(i)
    results = [continuation(lm, prompt, prefix, seq, stop=stop) for seq, stop in zip(sequences, stops)]
    return results[0], results[1:]


def beam_suggestions(lm, prompt, max_tokens=6, width=8, count=5, alpha=.7):
    prefix = lm.prefix_ids(prompt)
    active, finished = [([], 0.0)], []
    for depth in range(max_tokens):
        feasible = [(ids, score) for ids, score in active if len(prefix) + len(ids) <= lm.model.config.context_length]
        finished.extend((ids, score, "context_limit") for ids, score in active if len(prefix) + len(ids) > lm.model.config.context_length)
        if not feasible:
            break
        log_probs = next_batch(lm, [[*prefix, *ids] for ids, _ in feasible]).log_softmax(-1)
        log_probs[:, list(lm.forbidden)] = -torch.inf
        expanded = []
        for row, (ids, score) in enumerate(feasible):
            # An empty suggestion is not a keyboard action; only beam search has
            # this explicit constraint. Greedy/sampling retain empty outcomes.
            if not continuation(lm, prompt, prefix, ids)["comparison_text"]:
                log_probs[row, lm.special["eos"]] = -torch.inf
            values, tokens = log_probs[row].topk(min(width, log_probs.shape[1]))
            for value, token in zip(values.cpu().tolist(), tokens.cpu().tolist()):
                if value == -float("inf"):
                    continue
                new = [*ids, token]
                reason = stop_reason(lm, prompt, prefix, new)
                if reason:
                    finished.append((new, score + value, reason))
                else:
                    expanded.append((new, score + value))
        # Equal token length at each depth, so raw scores suffice here.
        active = sorted(expanded, key=lambda item: item[1], reverse=True)[:width]
        if depth == max_tokens - 1:
            finished.extend((ids, score, "max_new_tokens") for ids, score in active)
    finished.sort(key=lambda item: item[1] / max(1, len(item[0])) ** alpha, reverse=True)
    items = []
    for ids, score, stop in finished:
        item = continuation(lm, prompt, prefix, ids, score, stop)
        item["beam_score"] = score / max(1, len(ids)) ** alpha
        items.append(item)
    return unique_suggestions(items, count)


def reference_hit(items, references):
    return any(not set(item["flags"]) & {"prefix_changed", "replacement_character", "empty"}
               and item["comparison_text"].startswith(tuple(references)) for item in items)


def validate_cases(cases):
    if not cases.get("ranking") or not cases.get("suggestions"):
        raise ValueError("Both ranking and suggestion cases are required.")
    ids, groups = set(), {}
    for kind in ("ranking", "suggestions"):
        for case in cases[kind]:
            if case["id"] in ids:
                raise ValueError("Case IDs must be globally unique.")
            ids.add(case["id"])
            if kind == "ranking":
                candidates, acceptable = case["candidates"], case["acceptable"]
                if len(candidates) < 2 or len(set(candidates)) != len(candidates) or any(not c for c in candidates):
                    raise ValueError("Invalid candidate set.")
                if not acceptable or not set(acceptable) <= set(candidates) or len(set(acceptable)) != len(acceptable):
                    raise ValueError("Gold candidates must be a nonempty subset without duplicates.")
                old = groups.setdefault(case["group"], candidates)
                if old != candidates:
                    raise ValueError("Contrast group must share the same ordered candidate set.")
            elif not case["references"] or any(not ref for ref in case["references"]):
                raise ValueError("Nonempty suggestion references required.")


def write_markdown(path, report):
    metrics = report["metrics"]
    lines = ["# Tiny Japanese LM: IME diagnostic", "",
        "人工构造同音候选，非 AzooKey 实际输出。样本与训练语料是否重叠未知；不能外推真实输入法准确率。",
        "短语参考非穷尽，命中率仅为参考覆盖；未命中不等于错误。test split 未使用。", "",
        "## 候选排序", "", "| 方法 | Top-1 | Top-2 | MRR | 全组上下文正确 |", "| --- | --- | --- | --- | --- |"]
    for key, label in (("contextual_sum", "上下文 + sum（主指标）"), ("context_free_sum", "无上下文 + sum"),
                       ("contextual_mean", "上下文 + mean（次要诊断）")):
        m = metrics[key]
        lines.append(f"| {label} | {m['top1_correct']}/{m['cases']} | {m['top2_correct']}/{m['cases']} | {m['mrr']:.3f} | {m['all_contexts_correct_groups']}/{m['groups']} |")
    lines += ["", "| 上下文 | 预期 | 排序（logP sum） | 无上下文首选 |", "| --- | --- | --- | --- |"]
    for row in report["ranking"]:
        ordered = sorted(row["contextual"]["candidates"], key=lambda item: item["log_probability_sum"], reverse=True)
        prior = max(row["context_free"]["candidates"], key=lambda item: item["log_probability_sum"])["text"]
        scores = " > ".join(f"{item['text']} ({item['log_probability_sum']:.2f})" for item in ordered)
        lines.append(f"| {row['context']} | {' / '.join(row['acceptable'])} | {scores} | {prior} |")
    lines += ["", "## 短语联想", "", "| 前缀 | greedy | beam（最多5项） | sample（最多5项） |", "| --- | --- | --- | --- |"]
    for row in report["suggestions"]:
        beam = " / ".join(item["text"] for item in row["beam"])
        sample = " / ".join(item["text"] for item in row["sample"])
        lines.append(f"| {row['prompt']} | {row['greedy']['text']} | {beam} | {sample} |")
    lines += ["", "参数、token评分、原始采样（含空输出）、token IDs、停止原因、指纹见 report.json。",
        "Beam width=8，alpha=0.7，最长6个新token；PAD/UNK/BOS屏蔽，beam排除空EOS；无重复惩罚。",
        "sum 使用联合分词的公共前缀之后的原始词表 logP，不加EOS；分词边界变化见JSON。",
        "mean 是同一分数除以计分token数，可能改变长度偏好，未根据本轮结果选择评分策略。",
        "CPU/CUDA FP32运行耗时仅为开发机工具耗时，不代表iOS键盘延迟。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/models/tiny-ja-v1/best.pt")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts/tokenizers/ja-unigram-16k-v1")
    parser.add_argument("--cases", type=Path, default=ROOT / "configs/ime-eval-v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ime-eval/tiny-ja-v1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive.")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    validate_cases(cases)
    cases_sha = file_sha(args.cases)
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    lm = JapaneseLM(args.checkpoint, args.tokenizer, args.device)
    rows, priors = [], {}
    for case in cases["ranking"]:
        prior = priors.setdefault(case["group"], None)
        if prior is None:
            prior = priors[case["group"]] = score_candidates(lm, "", case["candidates"])
        rows.append({**case, "contextual": score_candidates(lm, case["context"], case["candidates"]), "context_free": prior})
    metrics = {"contextual_sum": ranking_metrics(rows),
        "context_free_sum": ranking_metrics(rows, field="context_free"),
        "contextual_mean": ranking_metrics(rows, score="log_probability_mean")}
    print(f"Ranking top1: {metrics['contextual_sum']['top1_correct']}/{len(rows)}; context-free: {metrics['context_free_sum']['top1_correct']}/{len(rows)}", flush=True)
    suggestions = []
    for index, case in enumerate(cases["suggestions"]):
        greedy, raw = generate_samples(lm, case["prompt"], 6, 42 + index * 100)
        sampled = unique_suggestions(raw, 5)
        beam = beam_suggestions(lm, case["prompt"])
        hits = {"greedy_top1": reference_hit([greedy], case["references"]),
            "sample_top5": reference_hit(sampled, case["references"]), "beam_top1": reference_hit(beam[:1], case["references"]),
            "beam_top5": reference_hit(beam, case["references"])}
        suggestions.append({**case, "greedy": greedy, "sample_raw": raw, "sample": sampled, "beam": beam, "reference_hits": hits})
        print(f"Suggestions {index + 1}/{len(cases['suggestions'])}: {case['prompt']} -> {greedy['text']}", flush=True)
    metrics["suggestion_reference_coverage"] = {method: {"hit_cases": sum(row["reference_hits"][method] for row in suggestions),
        "cases": len(suggestions)} for method in ("greedy_top1", "sample_top5", "beam_top1", "beam_top5")}
    report = {"version": "ime_evaluation_v1", "metadata": lm.metadata,
        "cases": {"path": str(args.cases.resolve()), "sha256": cases_sha, "version": cases["version"], "provenance": cases["provenance"]},
        "policy": {"ranking_primary": "Joint tokenization; sum of full-vocab logP after common prefix; no EOS",
            "ranking_secondary": "Mean per scored token, diagnostic only", "reading_used_by_model": False,
            "max_new_tokens": 6, "beam_width": 8, "beam_length_penalty_alpha": .7, "max_suggestions": 5,
            "sample_attempts": 8, "sample_temperature": .8, "sample_top_k": 50, "sample_top_p": .9,
            "seed": 42, "seed_per_case": "42 + 100 * case_index + attempt_index",
            "test_split_used": False, "threads": args.threads}, "metrics": metrics,
        "ranking": rows, "suggestions": suggestions, "elapsed_seconds_including_load": time.perf_counter() - started}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "report.json", report)
    write_markdown(args.output / "results.md", report)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Report: {args.output / 'results.md'}")


if __name__ == "__main__":
    main()
