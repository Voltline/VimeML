"""vimeml-ja-v2-20m on the Mac: generation search and the merged display list of a KV v4 package against the FP32 fixture.

Core ML conversion, alignment, INT8 compression, the batched KV v4 build, its plumbing fixture, `measure` and the
iOS resource are the Chinese tools run with `--bundle` (scripts/zh/deployment/coreml_zh_v1.py and
coreml_zh_kvbatch4.py: same 12K / d384 / 9-layer / context-128 block). This script replaces only their pinyin
`search` stage. Per fixture case (prepare_ja_v2_20m.py `fixture`), on one merged KV v4 package:

  generation  kana-position-bucketed constrained beam 16, top 10 (`ja_search`, `logits` function), against the
              fixture's generated top 10 (`top`, beam_score)
  scores      the fixture's union (AzooKey N-best + generated top 10) scored by the package with `score` from
              the fixture's own token ids (task rows and lm rows), against gen_score / lm_score; the display
              order from those scores against `shown`
  pipeline    the package's own generated top 10, merged with the case's AzooKey list and scored by the
              package, as the device would: display top 1 against shown[0]; generation-only display top 1
              against rescored[0]

usage (Mac, KV v4 env):
  python scripts/ja/deployment/coreml_ja_v2_20m.py search <int8-merged> <output> --bundle B \\
      [--cases outputs/vime-fixtures/ja-generation-v2-20m-c40-task-r1/cases.jsonl] \\
      [--index outputs/ja-gen-stage2/kana-index.json] [--reference <src-merged>]
  --torch runs the same checks on the bundle's PyTorch FP32 KV graphs (self-test, no Core ML; <package> unused).
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_ja_v2_20m import DEFAULT_INDEX, MODELS, lm_score_inputs, task_score_inputs  # noqa: E402
from vimeml.benchmarks.kana_gen_ja import merge_and_rank  # noqa: E402
from vimeml.deployment import ja_search as js, zh_v1, zh_v1_batch3 as z3, zh_v1_batch4 as z4  # noqa: E402
from vimeml.deployment.bundle import environment, read_json, write_json  # noqa: E402
from vimeml.training.data import file_sha  # noqa: E402

CASES = ROOT / MODELS["colloq-c40"]["fixture"] / "cases.jsonl"
BUNDLE = ROOT / MODELS["colloq-c40"]["bundle"]
CONTEXT = z3.CONTEXT


def runner_for(package, model=None):
    if model is not None:
        return z4.TorchRunner(model)
    return z4.CoreMLRunner(package, functions=("logits", "scored", "score"))


def score_rows(runner, rows):
    """rows: (prefix ids, continuation ids) or None per candidate -> summed log-probs (None kept). One prompt
    call per distinct prefix, then `score` calls of 16 rows (`zh_v1_batch3.score_continuations`)."""
    out, groups = [None] * len(rows), {}
    for i, row in enumerate(rows):
        if row is not None:
            groups.setdefault(tuple(row[0]), []).append(i)
    for prefix, idx in groups.items():
        sums, _ = z3.score_continuations(runner, list(prefix), [list(rows[i][1]) for i in idx])
        for i, value in zip(idx, sums):
            out[i] = value
    return out


def lm_rows(items):
    return [None if x is None else (x[0][:x[1]], x[0][x[1]:]) for x in items]


def generation_only(texts, gen, lm, lam):
    """The fixture's `rescored` rule: gen + lam * lm descending (gen alone where lm is missing), ties keep order."""
    key = [(gen[i] + lam * lm[i]) if lm[i] is not None and gen[i] is not None else (gen[i] if gen[i] is not None
           else float("-inf")) for i in range(len(texts))]
    return [texts[i] for i in sorted(range(len(texts)), key=lambda i: (-key[i], i))]


def compare(got, want):
    ref = dict(want)
    common = [abs(s - ref[t]) for t, s in got if t in ref]
    return {"exact": [t for t, _ in got] == [t for t, _ in want],
            "top1_equal": bool(got) and bool(want) and got[0][0] == want[0][0],
            "set_equal": {t for t, _ in got} == {t for t, _ in want},
            "max_common_score_diff": max(common or [0.0])}


def max_diff(got, want):
    pairs = [(g, w) for g, w in zip(got, want) if g is not None or w is not None]
    if any((g is None) != (w is None) for g, w in pairs):
        return float("inf")
    return max([abs(g - w) for g, w in pairs] or [0.0])


def check_case(c, runner, backend, sp, index, search, lam):
    reading, context = c["reading"], c["context"]
    started = time.perf_counter()
    ranked, stats = js.search(backend, sp, index, reading, context, beam=search["beam"], topk=search["topk"],
                              search_mode=search["mode"], fallback_free=search["fallback_free"])
    search_ms = (time.perf_counter() - started) * 1000
    got = [(t, s) for t, s, _ in ranked]
    generation = compare(got, [(x["text"], x["beam_score"]) for x in c["top"]])

    merged = c["merged"]
    fixture_gen = [x["text"] for x in sorted((x for x in merged if x["gen_rank"] is not None),
                                              key=lambda x: x["gen_rank"])]
    gen = score_rows(runner, [None if x["gen_prompt_ids"] is None else (x["gen_prompt_ids"], x["gen_output_ids"])
                              for x in merged])
    lm = score_rows(runner, [None if x["lm_ids"] is None else (x["lm_ids"][:x["lm_from"]], x["lm_ids"][x["lm_from"]:])
                             for x in merged])
    final = [g + lam * l if g is not None and l is not None else None for g, l in zip(gen, lm)]
    shown_scores = merge_and_rank(c["azookey"], fixture_gen, gen, lm, lam=lam)
    scores = {"max_gen_diff": max_diff(gen, [x["gen_score"] for x in merged]),
              "max_lm_diff": max_diff(lm, [x["lm_score"] for x in merged]),
              "max_final_diff": max_diff(final, [x["final_score"] for x in merged]),
              "shown_top1_equal": shown_scores[:1] == c["shown"][:1],
              "shown_exact": shown_scores == c["shown"]}

    started = time.perf_counter()
    texts = [t for t, _ in got]
    union = list(dict.fromkeys(list(c["azookey"]) + texts))
    p_gen = score_rows(runner, task_score_inputs(sp, reading, context, union, CONTEXT))
    p_lm = score_rows(runner, lm_rows(lm_score_inputs(sp, context, union, CONTEXT)[0]))
    shown = merge_and_rank(c["azookey"], texts, p_gen, p_lm, lam=lam)
    at = {t: i for i, t in enumerate(union)}
    rescored = generation_only(texts, [p_gen[at[t]] for t in texts], [p_lm[at[t]] for t in texts], lam)
    score_ms = (time.perf_counter() - started) * 1000
    pipeline = {"shown_top1": shown[0] if shown else None,
                "shown_top1_equal": shown[:1] == c["shown"][:1],
                "rescored_top1_equal": rescored[:1] == c["rescored"][:1]}
    return {"id": c["id"], "stratum": c.get("stratum"), "generation": generation, "scores": scores,
            "pipeline": pipeline, "search_fallback": stats.fallback, "top": got,
            "search_ms": round(search_ms, 1), "score_ms": round(score_ms, 1)}


def stage_search(a):
    import sentencepiece as spm
    out = Path(a.output)
    if out.exists():
        raise SystemExit(f"{out} exists; choose a new directory")
    bundle = Path(a.bundle)
    lm = zh_v1.BundleLMZh(bundle)
    if lm.metadata["release_name"] != "vimeml-ja-v2-20m":
        raise SystemExit(f"Bundle release_name={lm.metadata['release_name']!r} is not vimeml-ja-v2-20m")
    sp = spm.SentencePieceProcessor(model_file=str(bundle / "tokenizer.model"))
    meta = read_json(Path(a.cases).parent / "meta.json")
    if meta["model"]["bundle_manifest_sha256"] != lm.metadata["bundle_manifest_sha256"]:
        raise SystemExit("Fixture was built from another bundle")
    if meta["kana_index_sha256"] != file_sha(Path(a.index)):
        raise SystemExit("Kana index differs from the fixture's")
    index = js.KanaIndex.load(a.index)
    lam = meta["rescoring"]["lambda"]
    model = lm.model.float().eval() if a.torch else None
    runner = runner_for(a.package, model)
    backend = z3.BatchBackendV3(runner, scored=False)
    ref_backend = z3.BatchBackendV3(runner_for(a.reference), scored=False) if a.reference else None

    rows = []
    for line in open(a.cases, encoding="utf-8"):
        c = json.loads(line)
        row = check_case(c, runner, backend, sp, index, meta["search"], lam)
        if ref_backend is not None:
            r_ranked, _ = js.search(ref_backend, sp, index, c["reading"], c["context"], beam=meta["search"]["beam"],
                                    topk=meta["search"]["topk"], search_mode=meta["search"]["mode"],
                                    fallback_free=meta["search"]["fallback_free"])
            row["generation_vs_reference"] = compare(row["top"], [(t, s) for t, s, _ in r_ranked])
        rows.append(row)

    def count(section, key):
        return sum(bool(r[section][key]) for r in rows)

    summary = {
        "cases": len(rows),
        "generation": {k: count("generation", k) for k in ("exact", "top1_equal", "set_equal")}
                      | {"max_common_score_diff": max(r["generation"]["max_common_score_diff"] for r in rows)},
        "scores": {k: max(r["scores"][k] for r in rows) for k in ("max_gen_diff", "max_lm_diff", "max_final_diff")}
                  | {k: count("scores", k) for k in ("shown_top1_equal", "shown_exact")},
        "pipeline": {k: count("pipeline", k) for k in ("shown_top1_equal", "rescored_top1_equal")},
        "search_fallbacks": sum(bool(r["search_fallback"]) for r in rows),
        "p50_search_ms": statistics.median(r["search_ms"] for r in rows),
        "p50_score_ms": statistics.median(r["score_ms"] for r in rows),
        "differing_cases": {
            "generation_top1": [r["id"] for r in rows if not r["generation"]["top1_equal"]],
            "scores_shown_top1": [r["id"] for r in rows if not r["scores"]["shown_top1_equal"]],
            "pipeline_shown_top1": [r["id"] for r in rows if not r["pipeline"]["shown_top1_equal"]],
            "pipeline_rescored_top1": [r["id"] for r in rows if not r["pipeline"]["rescored_top1_equal"]],
        },
    }
    if ref_backend is not None:
        summary["generation_vs_reference"] = {k: sum(bool(r["generation_vs_reference"][k]) for r in rows)
                                              for k in ("exact", "top1_equal", "set_equal")}
    out.mkdir(parents=True)
    write_json(out / "summary.json", {
        "format": "vimeml_kv_batch_v4_search_check_ja",
        "package": "pytorch-fp32 (self-test)" if a.torch else str(a.package),
        "package_manifest_sha256": None if a.torch else file_sha(Path(a.package) / "manifest.json"),
        "reference": a.reference, "fixture": str(a.cases), "summary": summary,
        "method": "generation with the `logits` function; all scores with `score` after a `scored` prompt call "
                  "(zh_v1_batch3.score_continuations); Mac python wall times include input conversion",
        "environment": environment()})
    with open(out / "cases.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("search")
    s.add_argument("package")
    s.add_argument("output")
    s.add_argument("--bundle", default=str(BUNDLE))
    s.add_argument("--cases", default=str(CASES))
    s.add_argument("--index", default=str(ROOT / DEFAULT_INDEX))
    s.add_argument("--reference", default=None, help="merged FP32 package (src-merged) to compare generation with")
    s.add_argument("--torch", action="store_true", help="PyTorch FP32 KV graphs instead of Core ML (self-test)")
    a = p.parse_args()
    {"search": stage_search}[a.stage](a)


if __name__ == "__main__":
    main()
