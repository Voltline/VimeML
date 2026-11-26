"""ja-gen stage 2 evaluation: kana generation + AzooKey union rescoring and Standard dev guardrail.

Reads development pools only (never test or blind). Evaluates:
- reading arms (`mode="read"`): on ime-kana-ja-v1 development, merges AzooKey N-best 20 with
  beam-16 generated top-10 candidates, scores the union with task-mode `gen_score` and LM-mode
  `lm_score`, and ranks by `gen_score + lam * lm_score`; on Standard development (728 items),
  reranks AzooKey N-best 20 by LM score only.
- LM control arms (`mode="lm"`): reranks AzooKey N-best 20 by LM score on both development sets.

Reports notation-lenient Top-1 (primary), exact Top-1, Top-3, stratified breakdowns (length band,
context presence, source), D5 dictionary-reading subset, lambda sweep on development, and paired
author/dialogue-cluster sign-flip tests against V4.2 and paired LM controls.
"""
from collections import Counter
import json
import math
from pathlib import Path
import time

import torch

from vimeml.benchmarks import statistics
from vimeml.benchmarks.standard_ime import load_standard_export
from vimeml.benchmarks.zh13_lm_scores import load as load_zh_model, score_item
from vimeml.deployment.ja_search import EOS, LIMIT, prompt_ids, search as kana_search, to_hiragana
from vimeml.deployment.zh_torch_backend import ModelBatchBackend
from vimeml.training.data import file_sha, write_json

SPECIAL_PAD = 0
V42_KANA_LENIENT_TOP1 = 85.2
V42_STANDARD_FLOOR = 501.0
DEFAULT_LAMBDAS = (0.0, 0.8, 1.2, 1.6, 2.0, 2.5)
# Deviation D5 dictionary readings (surface, dictionary reading in hiragana)
D5_PAIRS = (("私", "わたくし"), ("日本", "にっぽん"), ("明日", "あす"), ("言う", "ゆう"), ("何", "なん"))


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_dev_pool(path):
    """Load an exported AzooKey candidate pool, strictly refusing test, blind or sealed splits."""
    path = Path(path)
    low = path.name.lower()
    if "test" in low or "blind" in low:
        raise ValueError(f"Development pools only; refusing {path}")
    manifest, _, rows = load_standard_export(path)
    if manifest.get("sealed") or manifest.get("split") != "development":
        raise ValueError(f"Development pools only; refusing split={manifest.get('split')!r}, sealed={manifest.get('sealed')!r}")
    return manifest, rows


def row_reading(row):
    """Extract normalized hiragana reading from an exported candidate row."""
    raw = row.get("query") or row.get("input_hiragana") or row.get("reading") or ""
    return to_hiragana(raw)


def is_d5_item(row):
    """True if any accepted answer and the input reading match a deviation D5 dictionary reading."""
    reading = row_reading(row)
    answers = row.get("answers") or []
    for surface, dict_read in D5_PAIRS:
        if dict_read in reading and any(surface in ans for ans in answers):
            return True
    return False


_RELATION_CACHE = {}


def lenient_match(choice, answers):
    from vimeml.benchmarks.notation import relation

    for answer in answers:
        key = (choice, answer)
        if key not in _RELATION_CACHE:
            _RELATION_CACHE[key] = relation(choice, answer)
        if _RELATION_CACHE[key] in ("=", "A"):
            return True
    return False


def hit_at_k(shown, answers, k, mode):
    top = shown[:k]
    if mode == "exact":
        return any(c in answers for c in top)
    return any(lenient_match(c, answers) for c in top)


def length_band(row):
    n = len(row["answers"][0])
    return "4-10" if n <= 10 else "11-20" if n <= 20 else "21-40"


@torch.inference_mode()
def task_scores(model, sp, reading, context, candidates, device="cpu"):
    """Exact FP32 conditional log P(candidate + EOS | [BOS, CTX] + ctx + [READ] + reading + [OUT]).

    Returns a list aligned with `candidates`; any candidate that fails tokenizer roundtrip or
    exceeds the model's context length gets None (never silently truncated).
    """
    if not candidates:
        return []
    max_ctx = model.config.context_length
    sequences, valid_idx, prompt_lens = [], [], []
    for idx, cand in enumerate(candidates):
        if not cand:
            continue
        out_ids = sp.encode(cand)
        if sp.decode(out_ids) != cand:
            continue
        out_with_eos = out_ids + [EOS]
        base = prompt_ids(sp, reading, context, limit=max_ctx + 1, min_output_room=len(out_with_eos))
        seq = base + out_with_eos
        if len(seq) - 1 > max_ctx:
            continue
        sequences.append(seq)
        valid_idx.append(idx)
        prompt_lens.append(len(base))

    scores = [None] * len(candidates)
    if not sequences:
        return scores

    width = max(len(s) - 1 for s in sequences)
    inputs = torch.full((len(sequences), width), SPECIAL_PAD, dtype=torch.long)
    targets = inputs.clone()
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    for row, (seq, plen) in enumerate(zip(sequences, prompt_lens)):
        inputs[row, :len(seq) - 1] = torch.tensor(seq[:-1], dtype=torch.long)
        targets[row, :len(seq) - 1] = torch.tensor(seq[1:], dtype=torch.long)
        # Supervise output tokens and trailing EOS (starting at target position plen - 1)
        mask[row, plen - 1:len(seq) - 1] = True

    logits = model(inputs.to(device)).float()
    logp = logits.log_softmax(-1).gather(-1, targets.to(device).unsqueeze(-1)).squeeze(-1).cpu()
    for row, orig_idx in enumerate(valid_idx):
        val = float(logp[row][mask[row]].sum())
        scores[orig_idx] = val if math.isfinite(val) else None
    return scores


def lm_scores_safe(model, sp, context, candidates, device="cpu"):
    """FP32 LM common-prefix log-prob scores via `zh13_lm_scores.score_item`, with per-item fallback."""
    if not candidates:
        return []
    try:
        return score_item(model, sp, context, candidates, device)["scores"]
    except ValueError:
        # Fallback: score eligible candidates individually so one overly long candidate does not zero out the pool
        out = [None] * len(candidates)
        eligible = []
        for idx, cand in enumerate(candidates):
            try:
                res = score_item(model, sp, context, [cand], device)
                eligible.append((idx, cand))
            except ValueError:
                continue
        if eligible:
            res = score_item(model, sp, context, [c for _, c in eligible], device)
            for (idx, _), sc in zip(eligible, res["scores"]):
                out[idx] = sc
        return out


def merge_and_rank(azookey_texts, gen_texts, gen_scores, lm_scores, lam=1.6):
    """Rank union = unique(azookey_texts + gen_texts) by `gen_score + lam * lm_score`."""
    union, seen = [], set()
    for t in list(azookey_texts) + list(gen_texts):
        if t and t not in seen:
            seen.add(t)
            union.append(t)
    if not union:
        return []
    scored = []
    for i, text in enumerate(union):
        gs = gen_scores[i] if i < len(gen_scores) else None
        ls = lm_scores[i] if i < len(lm_scores) else None
        if gs is not None and ls is not None:
            total = gs + lam * ls
            valid = 0
        elif ls is not None:
            total = lam * ls
            valid = 1
        elif gs is not None:
            total = gs
            valid = 1
        else:
            total = -math.inf
            valid = 2
        scored.append((valid, -total, i, text))
    scored.sort()
    return [t for _, _, _, t in scored]


def rank_lm_only(candidates, lm_scores):
    """Rerank candidates by LM score descending (stable on ties)."""
    order = sorted(
        range(len(candidates)),
        key=lambda i: (0 if lm_scores[i] is not None else 1, -(lm_scores[i] if lm_scores[i] is not None else -math.inf), i),
    )
    return [candidates[i] for i in order]


def share(values):
    n = len(values)
    hits = sum(bool(v) for v in values)
    return {"hits": hits, "cases": n, "rate": round(100.0 * hits / max(n, 1), 2)}


def summarize_rows(rows, shown_fn):
    """Compute primary/secondary metrics and stratified breakdowns on development rows."""
    out = {}
    for mode in ("exact", "notation_lenient"):
        for k in (1, 3):
            vals = [hit_at_k(shown_fn(r), r["answers"], k, mode) for r in rows]
            out[f"{mode}_top{k}"] = share(vals)
        groups = {}
        for r in rows:
            ok = hit_at_k(shown_fn(r), r["answers"], 1, mode)
            keys = [length_band(r), "context" if r.get("left_context") else "no-context", r.get("source", "unknown")]
            if r.get("d5_item"):
                keys.append("d5")
            for k in keys:
                groups.setdefault(k, []).append(ok)
        out[f"{mode}_top1_by_stratum"] = {k: share(v) for k, v in sorted(groups.items())}
    return out


def evaluate_arm(checkpoint, tokenizer, kana_pool_dir, standard_pool_dir, out_dir,
                 mode="read", index_path=None, device="cuda", beam=16, topk=10,
                 lam=1.6, lambdas=DEFAULT_LAMBDAS, search_mode="constrained",
                 fallback_free=True, limit=0):
    """Run Stage 2 evaluation for one checkpoint on both development pools."""
    from vimeml.deployment.ja_search import KanaIndex

    if mode not in ("read", "lm"):
        raise ValueError(f"Unknown mode {mode!r}; expected 'read' or 'lm'")
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Use an empty output directory: {out}")

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    kana_manifest, kana_rows = load_dev_pool(kana_pool_dir)
    std_manifest, std_rows = load_dev_pool(standard_pool_dir)
    if limit:
        kana_rows, std_rows = kana_rows[:limit], std_rows[:limit]

    model, sp, meta = load_zh_model(checkpoint, tokenizer, device)
    model.float().eval()
    backend = ModelBatchBackend(model, device) if mode == "read" else None
    index = KanaIndex.load(index_path) if (mode == "read" and index_path and search_mode == "constrained") else None
    if mode == "read" and search_mode == "constrained" and index is None:
        raise ValueError("Constrained reading search requires --index")

    started = time.time()
    kana_out = []
    fallbacks = 0
    for idx, row in enumerate(kana_rows, 1):
        az_texts = [c["text"] for c in row["candidates"]]
        context = row.get("left_context") or ""
        reading = row_reading(row)
        if mode == "read":
            ranked, stats = kana_search(backend, sp, index, reading, context, beam=beam, topk=topk,
                                        search_mode=search_mode, fallback_free=fallback_free)
            fallbacks += int(stats.fallback)
            gen_texts = [t for t, _, _ in ranked]
            gen_beam_scores = [s for _, s, _ in ranked]
            union = list(dict.fromkeys(az_texts + gen_texts))
            union_gen_scores = task_scores(model, sp, reading, context, union, device=device)
            union_lm_scores = lm_scores_safe(model, sp, context, union, device=device)
            shown_default = merge_and_rank(az_texts, gen_texts, union_gen_scores, union_lm_scores, lam=lam)
            kana_out.append({
                "id": row["id"], "source": row.get("source"), "group": row.get("group"),
                "provenance": row.get("provenance"), "left_context": context,
                "reading": reading, "answers": row["answers"], "d5_item": is_d5_item(row),
                "azookey": az_texts, "gen": gen_texts, "gen_beam_scores": gen_beam_scores,
                "union": union, "union_gen_scores": union_gen_scores, "union_lm_scores": union_lm_scores,
                "shown": shown_default, "search_calls": stats.calls, "search_rows": stats.rows,
                "search_mode": stats.mode, "search_fallback": stats.fallback,
            })
        else:
            lm_sc = lm_scores_safe(model, sp, context, az_texts, device=device)
            shown = rank_lm_only(az_texts, lm_sc)
            kana_out.append({
                "id": row["id"], "source": row.get("source"), "group": row.get("group"),
                "provenance": row.get("provenance"), "left_context": context,
                "reading": reading, "answers": row["answers"], "d5_item": is_d5_item(row),
                "azookey": az_texts, "union": az_texts, "union_lm_scores": lm_sc, "shown": shown,
            })
        if idx % 100 == 0 or idx == len(kana_rows):
            print(f"[kana {idx}/{len(kana_rows)}] {time.time() - started:.1f}s", flush=True)

    std_out = []
    for idx, row in enumerate(std_rows, 1):
        az_texts = [c["text"] for c in row["candidates"]]
        context = row.get("left_context") or ""
        lm_sc = lm_scores_safe(model, sp, context, az_texts, device=device)
        shown = rank_lm_only(az_texts, lm_sc)
        std_out.append({
            "id": row["id"], "source": row.get("source"), "group": row.get("group"),
            "provenance": row.get("provenance"), "left_context": context,
            "answers": row["answers"], "azookey": az_texts, "lm_scores": lm_sc, "shown": shown,
        })

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "kana-development-rows.jsonl", kana_out)
    write_jsonl(out / "standard-development-rows.jsonl", std_out)

    lambda_sweep = {}
    if mode == "read":
        for lv in lambdas:
            key = f"{lv:.2f}"
            lambda_sweep[key] = summarize_rows(
                kana_out,
                lambda r, v=lv: merge_and_rank(r["azookey"], r["gen"], r["union_gen_scores"], r["union_lm_scores"], lam=v),
            )
        # Also record generation-only (top-10 beam without AzooKey union) for diagnostics
        gen_only_summary = summarize_rows(kana_out, lambda r: r["gen"])
    else:
        gen_only_summary = None

    summary = {
        "test_read": False,
        "mode": mode,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "step": meta["step"],
        "arm": meta.get("arm"),
        "tokenizer_sha256": meta["tokenizer_model_sha256"],
        "kana_pool_manifest": kana_manifest.get("release_id") or file_sha(Path(kana_pool_dir) / "manifest.json"),
        "standard_pool_manifest": std_manifest.get("release_id") or file_sha(Path(standard_pool_dir) / "manifest.json"),
        "search": {"mode": search_mode, "beam": beam, "topk": topk, "lambda": lam,
                   "fallback_free": fallback_free, "fallbacks": fallbacks,
                   "index_sha256": file_sha(index_path) if index_path else None} if mode == "read" else None,
        "kana_development": summarize_rows(kana_out, lambda r: r["shown"]),
        "kana_generation_only": gen_only_summary,
        "kana_lambda_sweep": lambda_sweep,
        "standard_development": summarize_rows(std_out, lambda r: r["shown"]),
        "seconds": round(time.time() - started, 1),
    }
    write_json(out / "summary.json", summary)
    return summary


def evaluate_stage21_decision(read_summaries, lm_summaries, v42_kana_lenient=V42_KANA_LENIENT_TOP1,
                              standard_floor=V42_STANDARD_FLOOR):
    """Apply the registered Stage 2.1 4-seed decision rule."""
    read_kana = [s["kana_development"]["notation_lenient_top1"]["rate"] for s in read_summaries]
    lm_kana = [s["kana_development"]["notation_lenient_top1"]["rate"] for s in lm_summaries]
    read_std = [s["standard_development"]["exact_top1"]["hits"] for s in read_summaries]
    lm_std = [s["standard_development"]["exact_top1"]["hits"] for s in lm_summaries]
    mean_read_kana = sum(read_kana) / max(len(read_kana), 1)
    mean_lm_kana = sum(lm_kana) / max(len(lm_kana), 1)
    mean_read_std = sum(read_std) / max(len(read_std), 1)
    mean_lm_std = sum(lm_std) / max(len(lm_std), 1)
    cond_a = mean_read_kana > v42_kana_lenient and mean_read_kana > mean_lm_kana
    cond_b = mean_read_std >= standard_floor
    return {
        "read_kana_lenient_mean": round(mean_read_kana, 2),
        "lm_kana_lenient_mean": round(mean_lm_kana, 2),
        "v42_kana_lenient": v42_kana_lenient,
        "read_standard_exact_mean": round(mean_read_std, 2),
        "lm_standard_exact_mean": round(mean_lm_std, 2),
        "standard_floor": standard_floor,
        "pass_kana": cond_a,
        "pass_standard": cond_b,
        "keep_reading_recipe": bool(cond_a and cond_b),
    }


def paired_against_baseline(arm_rows, base_rows, lam=None, mode="notation_lenient"):
    """Author/dialogue-clustered paired sign-flip test on Top-1."""
    by_id = {r["id"]: r for r in base_rows}
    before, after, shared = [], [], []
    for r in arm_rows:
        b = by_id.get(r["id"])
        if b is None:
            continue
        shared.append(r)
        base_shown = b["orders"]["lm_context_sum"] if "orders" in b else b["shown"]
        if lam is not None and "union_gen_scores" in r:
            arm_shown = merge_and_rank(r["azookey"], r["gen"], r["union_gen_scores"], r["union_lm_scores"], lam=lam)
        else:
            arm_shown = r["shown"]
        before.append(hit_at_k(base_shown, r["answers"], 1, mode))
        after.append(hit_at_k(arm_shown, r["answers"], 1, mode))
    labels = statistics.clusters(shared, template=None)
    return statistics.paired(before, after, labels)
