"""vimeml-ja-v2-20m handoff for the Mac (Core ML) and Vime (iOS): weights, bundle, fixtures and ZIP.

Shares the 12K / d384 / 9-layer / context-128 block contract of vimeml-zh-v2-20m, so the
zh-v1 Core ML stateless and batched KV v4 (`prefix`, `score`, `scored`) exporters are reused
without architectural changes; only the Japanese weights, 12K tokenizer, reading index and
generation decoding contract (task mode with self rescoring) differ.

  weights   deployment.pt (FP32 tensors, provenance) and PyTorch FP32 logits + KV fixtures
  bundle    inference bundle from deployment.pt (`zh_v1.export_bundle`)
  fixture   Vime generation fixture: task mode, constrained beam 16 per reading position,
            top 10, self rescoring at lambda (default 1.6); then the confirmed display list:
            AzooKey N-best 20 + generated top 10, deduplicated, ordered by gen + lambda * lm
            (`kana_gen_ja.merge_and_rank`), with the token ids of every score so the device can
            reproduce each one; 120 development cases from ime-kana-ja-v1 (with their exported
            AzooKey N-best 20) + 5 hand-written Japanese conversational cases (generation only,
            no AzooKey list); test never read
  package   ZIP of bundle, FP32 fixtures, generation fixture, reading index and table with
            SHA256SUMS and CRC check

usage:
  python -X utf8 scripts/ja/deployment/prepare_ja_v2_20m.py <stage> \\
      [--model kd-cont|colloq-c40|read-s1] [--checkpoint PATH] [--output ZIP]
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

RELEASE = "vimeml-ja-v2-20m"
TOKENIZER_DIR = "artifacts/tokenizers/ja-kana-v1/ja-unigram-12288"
TOKENIZER_SHA256 = "61d71460606eed32f0a976611e5fde0dca12efeecb413c516838911a49d3aa53"
READING_TABLE = "configs/ja/ja-reading-table.tsv"
DEFAULT_KANA_POOL = "artifacts/benchmarks/ime-candidates-v3-imported/kana-ja-v1-development-n20"
DEFAULT_INDEX = "outputs/ja-gen-stage2/kana-index.json"
BEAM, TOPK, DEFAULT_LAMBDA = 16, 10, 1.6
FIXTURE_DEV_CASES = 120
FIXTURE_SEED = 20261210

# Neutral Japanese prose repeated for FP32 logits & 128-token KV alignment fixtures (not from any benchmark).
FIXTURE_TEXT = (
    "明日の午後から駅前の図書館で資料を整理する予定です。"
    "帰りに新しい本を買って、近所の喫茶店で少し休んでから家に戻ろうと思います。"
    "天気が良ければ公園を歩いて季節の景色も楽しみたいです。"
)

HANDWRITTEN_CASES = (
    {"id": "hand-01", "source": "handwritten", "group": "handwritten", "left_context": "",
     "reading": "きょうはいいてんきですね"},
    {"id": "hand-02", "source": "handwritten", "group": "handwritten", "left_context": "さっき",
     "reading": "えきまであるいていきました"},
    {"id": "hand-03", "source": "handwritten", "group": "handwritten", "left_context": "あしたの",
     "reading": "かいぎはじゅうじからです"},
    {"id": "hand-04", "source": "handwritten", "group": "handwritten", "left_context": "",
     "reading": "おつかれさまでしたまたあした"},
    {"id": "hand-05", "source": "handwritten", "group": "handwritten", "left_context": "それ",
     "reading": "ほんとうにおもしろいよね"},
)

MODELS = {
    "read-s1": {
        "model_dir": "artifacts/models/ja-kana-20m-read-s1",
        "checkpoint_sha256": "35ac57685062601d59c94e4c8f67f738cd7a51eb47e9a1ad9acb62ba0ed0924e",
        "reference": "outputs/deployment/vimeml-ja-v2-20m-read-s1-fp32-input",
        "bundle": f"artifacts/releases/{RELEASE}-read-s1/inference",
        "fixture": "outputs/vime-fixtures/ja-generation-v2-20m-read-s1-task-r1",
        "arm": "20m-read-s1",
        "phase": "ja-gen Stage 2.1 reading arm: task mode with self rescoring",
    },
    "kd-cont": {
        "model_dir": "artifacts/models/ja-kana-20m-kd-cont",
        "checkpoint_sha256": "5a44839657569b34dea57fae4a45a2050506f892a441e38715756727fe46a92f",
        "reference": "outputs/deployment/vimeml-ja-v2-20m-kd-fp32-input",
        "bundle": f"artifacts/releases/{RELEASE}-kd/inference",
        "fixture": "outputs/vime-fixtures/ja-generation-v2-20m-kd-task-r1",
        "arm": "20m-kd-cont",
        "phase": "ja-gen Stage 2.3 distilled 20M: task mode with self rescoring",
    },
    "colloq-c40": {
        "model_dir": "artifacts/models/ja-kana-20m-colloq-c40",
        "checkpoint_sha256": "19b04ddfe0f98041ffe50511cab9d64d2f2a20b2918f5e78cc78c0768981d15f",
        "reference": "outputs/deployment/vimeml-ja-v2-20m-c40-fp32-input",
        "bundle": f"artifacts/releases/{RELEASE}/inference",
        "fixture": "outputs/vime-fixtures/ja-generation-v2-20m-c40-task-r1",
        "arm": "20m-colloq-c40",
        "phase": "ja-gen Stage 3 colloquial 20M (c40), confirmed on test in stage 4: task mode with self rescoring",
    },
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_path(value, root=ROOT):
    p = Path(value)
    return p if p.is_absolute() else Path(root) / p


def prepare_weights(checkpoint, deployment, tokenizer_dir, reference_dir, *,
                    checkpoint_sha256=None, tokenizer_sha256=TOKENIZER_SHA256,
                    release_name=RELEASE, arm=None, fixture_text=FIXTURE_TEXT,
                    loader=None, sp_factory=None):
    """Strip optimizer state into deployment.pt and compute PyTorch FP32 logits + KV fixtures."""
    import numpy as np
    import torch

    checkpoint = Path(checkpoint)
    deployment = Path(deployment)
    tokenizer_dir = Path(tokenizer_dir)
    reference_dir = Path(reference_dir)
    tok_model = tokenizer_dir / "tokenizer.model"

    if deployment.exists() or reference_dir.exists():
        raise SystemExit(f"deployment.pt or FP32 reference already exists: {deployment}, {reference_dir}")

    actual_ckpt_sha = sha256(checkpoint)
    if checkpoint_sha256 and not str(checkpoint_sha256).startswith("PIN_AFTER_"):
        if actual_ckpt_sha != checkpoint_sha256:
            raise ValueError(f"Checkpoint SHA256 mismatch: expected {checkpoint_sha256}, got {actual_ckpt_sha}")
    actual_tok_sha = sha256(tok_model)
    if tokenizer_sha256 and actual_tok_sha != tokenizer_sha256:
        raise ValueError(f"Tokenizer SHA256 mismatch: expected {tokenizer_sha256}, got {actual_tok_sha}")

    torch.set_num_threads(4)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if any(t.dtype != torch.float32 for t in saved["model"].values()):
        raise ValueError("Expected FP32 master weights in checkpoint")
    exported = {
        key: saved[key]
        for key in ("format", "architecture", "model", "model_config", "step", "signatures", "tokenizer_identity")
    }
    exported["epoch"] = saved.get("epoch")
    exported["config"] = {key: (saved.get("config") or {}).get(key) for key in ("arm", "architecture", "stage")}
    exported.update(
        deployment_only=True,
        source_checkpoint_sha256=actual_ckpt_sha,
        release_name=release_name,
    )
    deployment.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, deployment)
    reloaded = torch.load(deployment, map_location="cpu", weights_only=True)
    if reloaded["model"].keys() != saved["model"].keys() or any(
        not torch.equal(value, reloaded["model"][name]) for name, value in saved["model"].items()
    ):
        raise ValueError("Inference export altered a model tensor")
    del saved

    if loader is None:
        from vimeml.benchmarks.zh13_lm_scores import load as loader
    model, sp_loaded, meta = loader(deployment, tok_model, "cpu")
    model = model.float().eval()
    if sp_factory is not None:
        processor = sp_factory(tok_model)
    else:
        processor = sp_loaded

    ctx = int(model.config.context_length)
    content = processor.encode(fixture_text * 12)
    bos = processor.bos_id() if hasattr(processor, "bos_id") else 2
    pad = processor.pad_id() if hasattr(processor, "pad_id") else 0
    if (bos, pad) != (2, 0) or len(content) < ctx - 1:
        raise ValueError("Unexpected tokenizer special IDs or fixture text too short")

    short_len = min(15, ctx - 2)
    cases = {
        "bos": [bos],
        "short": [bos, *content[:short_len]],
        "long": [bos, *content[:ctx - 1]],
    }
    cases["right_pad"] = cases["short"] + [pad] * (ctx - len(cases["short"]))
    cases["changed_future"] = cases["short"] + content[:ctx - len(cases["short"])]

    arrays = {}
    with torch.inference_mode():
        for name, ids in cases.items():
            inputs = torch.tensor([ids], dtype=torch.int32)
            arrays[f"{name}_input_ids"] = inputs.numpy()
            arrays[f"{name}_logits"] = model(inputs.long()).numpy()
        cache, steps = model.new_cache(), []
        long_ids = torch.tensor([cases["long"]], dtype=torch.int32)
        for pos in range(long_ids.shape[1]):
            steps.append(model.forward_cached(long_ids[:, pos:pos + 1], cache))
        arrays["long_kv_step_logits"] = torch.cat(steps, dim=1).numpy()

    worst = float(np.abs(arrays["long_kv_step_logits"] - arrays["long_logits"]).max())
    if worst > 3e-4:
        raise ValueError(f"forward_cached differs from forward by {worst}")

    reference_dir.mkdir(parents=True)
    np.savez_compressed(reference_dir / "logits.npz", **arrays)
    deploy_sha = sha256(deployment)
    write_json(reference_dir / "reference.json", {
        "format": "vimeml_zh_v1_mac_preparation_reference_v1",
        "release_name": release_name,
        "checkpoint_step": meta["step"],
        "checkpoint_sha256": actual_ckpt_sha,
        "deployment_checkpoint_sha256": deploy_sha,
        "tokenizer_sha256": actual_tok_sha,
        "precision": "fp32",
        "device": "cpu",
        "torch_version": torch.__version__,
        "cases": {
            name: {"length": len(ids),
                   "valid_prefix": len(cases["short"]) if name in {"right_pad", "changed_future"} else len(ids)}
            for name, ids in cases.items()
        },
        "kv_steps": {
            "case": "long",
            "array": "long_kv_step_logits",
            "max_abs_vs_stateless": worst,
            "method": "forward_cached one token at a time from an empty cache",
        },
        "policy": "PyTorch preparation fixtures; not Core ML validation, production quality or device timing.",
    })
    return {
        "release_name": release_name,
        "arm": arm or exported["config"].get("arm"),
        "checkpoint_step": meta["step"],
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": actual_ckpt_sha,
        "deployment_checkpoint": str(deployment),
        "deployment_checkpoint_sha256": deploy_sha,
        "tokenizer_sha256": actual_tok_sha,
        "model_config": exported["model_config"],
        "parameter_count": model.parameter_count(),
        "kv_cache_shape": [
            exported["model_config"]["n_layers"], 1,
            exported["model_config"]["n_heads"], ctx,
            exported["model_config"]["d_model"] // exported["model_config"]["n_heads"],
        ],
        "max_abs_kv_vs_stateless": worst,
        "optimizer_included": False,
    }


def select_fixture_cases(kana_pool_dir, dev_count=FIXTURE_DEV_CASES, seed=FIXTURE_SEED):
    """Stratified deterministic draw from development pool only + 5 hand-written Japanese cases."""
    from vimeml.benchmarks.kana_gen_ja import length_band, load_dev_pool, row_reading

    _, rows = load_dev_pool(kana_pool_dir)
    rng = random.Random(seed)
    buckets = collections.defaultdict(list)
    for row in rows:
        key = (bool(row.get("left_context")), length_band(row))
        buckets[key].append(row)
    for vals in buckets.values():
        rng.shuffle(vals)

    selected = []
    target = min(dev_count, len(rows))
    while len(selected) < target and any(buckets.values()):
        for key in sorted(buckets):
            if not buckets[key] or len(selected) >= target:
                continue
            r = buckets[key].pop()
            selected.append({
                "id": r["id"],
                "source": r.get("source", "development"),
                "group": r.get("group"),
                "left_context": r.get("left_context") or "",
                "reading": row_reading(r),
                "azookey": [cand["text"] for cand in r["candidates"]],
                "stratum": f'{"ctx" if r.get("left_context") else "noctx"}/{length_band(r)}',
            })
    for hc in HANDWRITTEN_CASES:
        selected.append({**hc, "azookey": [], "stratum": "handwritten"})
    return selected


def task_score_inputs(sp, reading, context, candidates, max_ctx):
    """Per candidate, the (prompt ids, output ids + EOS) that `kana_gen_ja.task_scores` scores, or None
    where it gives None (tokenizer roundtrip failure or longer than the context)."""
    from vimeml.deployment.ja_search import EOS, prompt_ids

    out = []
    for cand in candidates:
        ids = sp.encode(cand) if cand else []
        if not cand or sp.decode(ids) != cand:
            out.append(None)
            continue
        output = ids + [EOS]
        base = prompt_ids(sp, reading, context, limit=max_ctx + 1, min_output_room=len(output))
        out.append(None if len(base) + len(output) - 1 > max_ctx else (base, output))
    return out


def lm_score_inputs(sp, context, candidates, max_ctx):
    """Per candidate, the (full ids, first scored position) that `kana_gen_ja.lm_scores_safe` scores:
    one joint `zh13_lm_scores.score_item` call over all candidates, or, when that call is refused, one
    joint call over the candidates that are accepted alone. None where no score is given."""
    from vimeml.benchmarks.zh13_lm_scores import SPECIAL, common_prefix

    def joint(cands):
        context_ids = [SPECIAL["bos"], *sp.encode(context)]
        sequences = []
        for cand in cands:
            content = sp.encode(context + cand)
            if sp.decode(content) != context + cand:
                raise ValueError("roundtrip")
            seq = [SPECIAL["bos"], *content]
            if not 1 <= len(seq) - 1 <= max_ctx:
                raise ValueError("length")
            sequences.append(seq)
        common = common_prefix([context_ids, *sequences])
        if common < 1 or any(len(seq) <= common for seq in sequences):
            raise ValueError("no targets")
        return [(seq, common) for seq in sequences]

    if not candidates:
        return [], False
    try:
        return joint(candidates), False
    except ValueError:
        out, eligible = [None] * len(candidates), []
        for i, cand in enumerate(candidates):
            try:
                joint([cand])
                eligible.append(i)
            except ValueError:
                continue
        if eligible:
            for i, item in zip(eligible, joint([candidates[i] for i in eligible])):
                out[i] = item
        return out, True


def build_generation_fixture(model, sp, index, cases, out_dir, *,
                             release_name=RELEASE, checkpoint_sha256="",
                             bundle_manifest_sha256="", tokenizer_sha256="",
                             index_sha256="", reading_table_sha256="",
                             phase="", beam=BEAM, topk=TOPK, lam=DEFAULT_LAMBDA,
                             device="cpu"):
    """Run constrained kana beam search + self rescoring in FP32 and save Vime alignment fixture."""
    from vimeml.benchmarks.kana_gen_ja import lm_scores_safe, merge_and_rank, task_scores
    from vimeml.deployment import ja_search as js
    from vimeml.deployment.zh_torch_backend import ModelBatchBackend

    out = Path(out_dir)
    if out.exists():
        raise SystemExit(f"Fixture output already exists: {out}")

    model = model.float().eval()
    backend = ModelBatchBackend(model, device)
    results = []
    strata = collections.Counter()
    fallbacks = 0
    lm_fallback_cases = 0
    max_ctx = int(model.config.context_length)

    for c in cases:
        reading = js.to_hiragana(c["reading"])
        context = c.get("left_context") or ""
        ranked, stats = js.search(backend, sp, index, reading, context, beam=beam, topk=topk,
                                  search_mode="constrained", fallback_free=True)
        fallbacks += int(stats.fallback)
        texts = [t for t, _, _ in ranked]
        gen_scores = task_scores(model, sp, reading, context, texts, device=device) if texts else []
        lm_scores = lm_scores_safe(model, sp, context, texts, device=device) if texts else []
        top = []
        for i, (text, beam_s, ids) in enumerate(ranked):
            gs = gen_scores[i] if i < len(gen_scores) and gen_scores[i] is not None else beam_s
            ls = lm_scores[i] if i < len(lm_scores) else None
            fs = None if ls is None else round(gs + lam * ls, 5)
            top.append({
                "text": text,
                "ids": ids,
                "beam_score": round(beam_s, 5),
                "gen_score": round(gs, 5),
                "lm_score": None if ls is None else round(ls, 5),
                "final_score": fs,
            })
        order = sorted(
            range(len(top)),
            key=lambda i: (-(top[i]["final_score"] if top[i]["final_score"] is not None else top[i]["gen_score"]), i),
        )

        azookey = list(c.get("azookey") or [])
        union = list(dict.fromkeys(azookey + texts))
        union_gen = task_scores(model, sp, reading, context, union, device=device) if union else []
        union_lm = lm_scores_safe(model, sp, context, union, device=device) if union else []
        shown = merge_and_rank(azookey, texts, union_gen, union_lm, lam=lam)
        gen_inputs = task_score_inputs(sp, reading, context, union, max_ctx)
        lm_inputs, lm_fallback = lm_score_inputs(sp, context, union, max_ctx)
        lm_fallback_cases += int(lm_fallback)
        merged = []
        for i, text in enumerate(union):
            gs, ls = union_gen[i], union_lm[i]
            if (gs is None) != (gen_inputs[i] is None) or (ls is None) != (lm_inputs[i] is None):
                raise ValueError(f"score inputs disagree with scores for {c['id']} candidate {i}")
            merged.append({
                "text": text,
                "azookey_rank": azookey.index(text) if text in azookey else None,
                "gen_rank": texts.index(text) if text in texts else None,
                "gen_score": None if gs is None else round(gs, 5),
                "lm_score": None if ls is None else round(ls, 5),
                "final_score": round(gs + lam * ls, 5) if gs is not None and ls is not None else None,
                "gen_prompt_ids": None if gen_inputs[i] is None else gen_inputs[i][0],
                "gen_output_ids": None if gen_inputs[i] is None else gen_inputs[i][1],
                "lm_ids": None if lm_inputs[i] is None else lm_inputs[i][0],
                "lm_from": None if lm_inputs[i] is None else lm_inputs[i][1],
            })
        results.append({
            "id": c["id"],
            "source": c.get("source"),
            "stratum": c.get("stratum"),
            "reading": reading,
            "context": context,
            "prompt_ids": js.prompt_ids(sp, reading, context),
            "top": top,
            "rescored": [top[i]["text"] for i in order],
            "azookey": azookey,
            "merged": merged,
            "shown": shown,
            "lm_fallback": lm_fallback,
            "decode_calls": stats.calls,
            "decode_rows": stats.rows,
            "search_mode": stats.mode,
            "search_fallback": stats.fallback,
        })
        strata[c.get("stratum", "unknown")] += 1

    out.mkdir(parents=True)
    with open(out / "cases.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "format": "vime_ja_generation_fixture_v1",
        "phase": phase,
        "test_read": False,
        "model": {
            "release": release_name,
            "checkpoint_sha256": checkpoint_sha256,
            "bundle_manifest_sha256": bundle_manifest_sha256,
            "precision": "fp32 (PyTorch, CPU)",
        },
        "tokenizer_sha256": tokenizer_sha256,
        "kana_index_sha256": index_sha256,
        "reading_table_sha256": reading_table_sha256,
        "search": {
            "mode": "constrained",
            "fallback_free": True,
            "fallbacks": fallbacks,
            "beam": beam,
            "topk": topk,
            "prompt": "[2 BOS, 4 CTX] + encode(context) + [5 READ] + encode(hiragana_reading) + [6 OUT]",
            "beam_per": "hiragana reading character position; beam 16 kept per position after dedup by token ids",
            "finish": "reading consumed exactly, then + log P(EOS=3) after the last token",
            "token_limit": "prompt + output < 128",
        },
        "rescoring": {
            "lambda": lam,
            "lm_score": "same model in lm mode: BOS + context + candidate tokenized jointly, sum of log-probs "
                        "from the first token diverging from context alone, no EOS",
            "final_score": "gen_score + lambda * lm_score; descending, ties keep generation order",
        },
        "merge": {
            "azookey": "the case's AzooKey N-best (20 for development cases, converter d59a28e4c7ca, typo off, "
                       "Zenzai off; empty for hand-written cases)",
            "union": "AzooKey N-best in its order, then generated top 10 in generation order; exact-string dedup",
            "gen_score": "task-mode sum of log P(gen_output_ids | gen_prompt_ids), gen_output_ids = encode(text) + "
                         "[EOS 3]; the prompt trims left-context tokens so prompt + output - 1 <= 128; null when the "
                         "text does not roundtrip through the tokenizer or does not fit",
            "lm_score": "lm mode: sum of log P(lm_ids[t] | lm_ids[:t]) for t >= lm_from; lm_ids = BOS + "
                        "encode(context + text), lm_from = common prefix of BOS + encode(context) and every "
                        "candidate's lm_ids in one joint call (one joint call over the candidates accepted alone "
                        "when the joint call over all is refused: lm_fallback); null when the candidate is refused",
            "order": "both scores: gen_score + lambda * lm_score; candidates with only one score after all with "
                     "both, by that score (lambda * lm_score or gen_score); none last; ties keep union order",
            "shown": "the display order; shown[0] is the first candidate",
            "lm_fallback_cases": lm_fallback_cases,
        },
        "cases": len(results),
        "strata": dict(sorted(strata.items())),
        "tolerance": "FP32 reference; on-device INT8 may differ by up to ~0.4 in logits: expect matching candidate "
                     "set, scores within ~0.5, rare swaps of near-tied neighbours",
    }
    write_json(out / "meta.json", meta)
    return meta


def stage_package(output, files_map):
    """Zip `{archive_rel_path: local_path}` with SHA256SUMS and verify ZIP CRC."""
    output = Path(output)
    if output.exists():
        raise SystemExit(f"Output ZIP already exists: {output}")

    entries = sorted((arc, Path(local)) for arc, local in files_map.items())
    for arc, local in entries:
        if not local.is_file():
            raise SystemExit(f"Missing file for package ({arc}): {local}")

    sums = "".join(f"{sha256(local)}  {arc}\n" for arc, local in entries)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for arc, local in entries:
            z.write(local, arc)
        z.writestr("SHA256SUMS", sums)
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:
            raise SystemExit("ZIP CRC verification failed")
    return {
        "zip": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "files": len(entries) + 1,
    }


def collect_tree(base_dir, archive_prefix):
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        raise SystemExit(f"Missing directory: {base_dir}")
    return {
        f"{archive_prefix}/{p.relative_to(base_dir).as_posix()}": p
        for p in sorted(base_dir.rglob("*")) if p.is_file()
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["weights", "bundle", "fixture", "package"])
    p.add_argument("--model", choices=sorted(MODELS), default="colloq-c40")
    p.add_argument("--root", type=Path, default=ROOT, help="Workspace root for relative artifact paths")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--checkpoint-sha256", default=None)
    p.add_argument("--tokenizer-dir", default=TOKENIZER_DIR)
    p.add_argument("--index", default=DEFAULT_INDEX)
    p.add_argument("--kana-pool", default=DEFAULT_KANA_POOL)
    p.add_argument("--reading-table", default=READING_TABLE)
    p.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    spec = MODELS[args.model]
    root = args.root.resolve()
    checkpoint = resolve_path(args.checkpoint or f"{spec['model_dir']}/final.pt", root)
    deployment = checkpoint.parent / "deployment.pt"
    tokenizer_dir = resolve_path(args.tokenizer_dir, root)
    reference = resolve_path(spec["reference"], root)
    bundle = resolve_path(spec["bundle"], root)
    fixture = resolve_path(spec["fixture"], root)
    index_path = resolve_path(args.index, root)
    kana_pool = resolve_path(args.kana_pool, root)
    table_path = resolve_path(args.reading_table, root)
    ckpt_sha = args.checkpoint_sha256 or spec["checkpoint_sha256"]

    if args.stage == "weights":
        res = prepare_weights(
            checkpoint, deployment, tokenizer_dir, reference,
            checkpoint_sha256=ckpt_sha, tokenizer_sha256=TOKENIZER_SHA256,
            release_name=RELEASE, arm=spec["arm"],
        )
        print(json.dumps(res, ensure_ascii=False, indent=1))
    elif args.stage == "bundle":
        from vimeml.deployment import zh_v1

        res = zh_v1.export_bundle(deployment, tokenizer_dir, bundle)
        print(json.dumps({
            k: res[k] for k in ("release_name", "checkpoint_sha256", "checkpoint_step", "parameter_count")
        }, ensure_ascii=False))
    elif args.stage == "fixture":
        import torch
        from vimeml.deployment import ja_search as js, zh_v1

        torch.set_num_threads(4)
        lm = zh_v1.BundleLMZh(bundle)
        if lm.metadata["release_name"] != RELEASE:
            raise SystemExit(f"Bundle release_name={lm.metadata['release_name']!r} != {RELEASE!r}")
        index = js.KanaIndex.load(index_path)
        cases = select_fixture_cases(kana_pool)
        meta = build_generation_fixture(
            lm.model, lm.processor, index, cases, fixture,
            release_name=RELEASE,
            checkpoint_sha256=lm.metadata["checkpoint_sha256"],
            bundle_manifest_sha256=lm.metadata["bundle_manifest_sha256"],
            tokenizer_sha256=lm.metadata["tokenizer_sha256"],
            index_sha256=sha256(index_path),
            reading_table_sha256=sha256(table_path),
            phase=spec["phase"],
            lam=args.lam,
        )
        print(json.dumps({"cases": meta["cases"], "strata": meta["strata"],
                          "fallbacks": meta["search"]["fallbacks"],
                          "lm_fallback_cases": meta["merge"]["lm_fallback_cases"]}, ensure_ascii=False))
    elif args.stage == "package":
        if not args.output:
            p.error("package requires --output")
        files_map = {}
        files_map.update(collect_tree(bundle, spec["bundle"]))
        files_map.update(collect_tree(reference, spec["reference"]))
        files_map.update(collect_tree(fixture, spec["fixture"]))
        files_map["outputs/ja-gen-stage2/kana-index.json"] = index_path
        files_map["configs/ja/ja-reading-table.tsv"] = table_path
        res = stage_package(args.output, files_map)
        print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
