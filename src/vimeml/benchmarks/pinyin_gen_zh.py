"""Pinyin generation on the public set ime-pinyin-zh-v1 (development only), the Vime way.

run: every input of every development record (full pinyin plus each abbreviation) goes through
`zh_search.search` (beam per key position) with the record's in-field context, task or lm mode. The
top 10 are then scored by a prior (vimeml-tiny-zh-v1 by default) after the same context, with the
production rule (BOS + context + candidate, targets from the shared prefix on; any ineligible
candidate makes the whole call fail, and the generation order is kept). One JSONL row per input.

reprior: the same rows with the prior scores of another checkpoint (for example the model itself, as a
single-model deployment would use it); generation is not rerun.

score: Top-1/3/10 by input kind for the generation order and for genScore + λ·prior, and paired
comparisons on Top-1 against other runs, with a sentence-cluster sign-flip test.

The test split is never read here.
"""
import json
import random
import time
from pathlib import Path

KINDS = ("full", "mixed", "initials")
PERMUTATIONS = 20000
SIGN_FLIP_SEED = 1116


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def inputs(record):
    """(kind, keys) in the order of the private benchmark's harness."""
    out = [("full", record["full_input"])]
    for a in record["abbreviations"]:
        whole = len(a["abbreviated_word_indices"]) == len(record["words"])
        out.append(("initials" if whole else "mixed", a["input"]))
    return out


def load_set(frozen, limit=0):
    frozen = Path(frozen)
    records = read_jsonl(frozen / "development.jsonl")
    if any(r.get("split") != "development" for r in records):
        raise ValueError("development.jsonl holds a non-development record")
    contexts = {r["id"]: r["infield"] for r in read_jsonl(frozen / "development-contexts.jsonl")}
    return (records[:limit] if limit else records), contexts


def prior_scores(model, processor, context, candidates, device):
    from vimeml.benchmarks.zh13_lm_scores import score_item
    if not candidates:
        return None
    try:
        return score_item(model, processor, context, candidates, device)["scores"]
    except ValueError:
        return None


def run(checkpoint, tokenizer, readings, frozen, out, mode="task", beam=16, topk=10, prior=None,
        device="cpu", limit=0, use_context=True, shard=(0, 1)):
    """shard (I, N): only records whose index mod N is I; merge() joins the N shards in set order."""
    import sentencepiece as spm
    from vimeml.benchmarks.zh13_lm_scores import load
    from vimeml.deployment import zh_search as zs
    from vimeml.deployment.zh_torch_backend import ModelBatchBackend
    from vimeml.posttraining.zh_readings import ReadingTable
    from vimeml.training.data import file_sha

    model, processor, meta = load(checkpoint, tokenizer, device)
    backend = ModelBatchBackend(model, device)
    index = zs.Index(processor, ReadingTable(readings))
    prior_model = None
    if prior:
        prior_model, prior_processor, prior_meta = load(prior, tokenizer, device)
        prior_model.float().eval()
    records, contexts = load_set(frozen, limit)
    if not 0 <= shard[0] < shard[1]:
        raise ValueError("shard is I/N with 0 <= I < N")
    records = records[shard[0]::shard[1]]
    rows, started = [], time.time()
    total = sum(len(inputs(r)) for r in records)
    for record in records:
        context = contexts.get(record["id"], "") if use_context else ""
        for kind, keys in inputs(record):
            ranked, stats = zs.search(backend, processor, index, keys, context, mode, beam, topk)
            texts = [t for t, _, _ in ranked]
            rows.append({"id": record["id"], "kind": kind, "input": keys, "text": record["text"],
                         "gen": texts, "genScores": [s for _, s, _ in ranked],
                         "prior": prior_scores(prior_model, prior_processor, context, texts, device)
                         if prior_model is not None else None,
                         "calls": stats.calls, "rows": stats.rows})
            if len(rows) % 500 == 0:
                print(f"{len(rows)}/{total} {time.time() - started:.0f}s", flush=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "rows.jsonl", rows)
    info = {"checkpoint": str(checkpoint), "checkpoint_sha256": meta["checkpoint_sha256"], "step": meta["step"],
            "mode": mode, "beam": beam, "topk": topk, "context": use_context, "device": device,
            "prior": None if not prior else {"checkpoint": Path(prior).name, "sha256": prior_meta["checkpoint_sha256"]},
            "readings_sha256": file_sha(readings), "set_manifest_sha256": file_sha(Path(frozen) / "manifest.json"),
            "inputs": len(rows), "limit": limit, "seconds": round(time.time() - started, 1)}
    if shard[1] > 1:
        info["shard"] = list(shard)
    (out / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return info


def merge(shards, frozen, out):
    """One run from shard runs 0..N-1 of the same settings: rows in set order, as an unsharded run writes them."""
    shards, out = [Path(d) for d in shards], Path(out)
    metas = [json.loads((d / "meta.json").read_text("utf-8")) for d in shards]
    count = len(shards)
    if sorted(tuple(m.get("shard", (0, 1))) for m in metas) != [(i, count) for i in range(count)]:
        raise ValueError("Need shards 0..N-1 of one N")
    fixed = ("checkpoint_sha256", "step", "mode", "beam", "topk", "context", "prior", "readings_sha256",
             "set_manifest_sha256", "limit")
    if any(any(m[k] != metas[0][k] for k in fixed) for m in metas):
        raise ValueError("Shards differ in settings")
    records, _ = load_set(frozen, metas[0]["limit"])
    order = {r["id"]: n for n, r in enumerate(records)}
    rows = [row for d in shards for row in read_jsonl(d / "rows.jsonl")]
    rows.sort(key=lambda row: order[row["id"]])          # stable: a record's inputs keep their order
    if len(rows) != sum(len(inputs(r)) for r in records):
        raise ValueError("Merged shards do not cover the set")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "rows.jsonl", rows)
    info = {k: v for k, v in metas[0].items() if k != "shard"}
    info.update(inputs=len(rows), seconds=max(m["seconds"] for m in metas), shards=count)
    (out / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return info


def reprior(rows_dir, prior, tokenizer, frozen, out, device="cpu"):
    """Copy a run with its prior column recomputed by `prior` (same contexts, same candidates)."""
    from vimeml.benchmarks.zh13_lm_scores import load
    rows_dir, out = Path(rows_dir), Path(out)
    info = json.loads((rows_dir / "meta.json").read_text("utf-8"))
    model, processor, meta = load(prior, tokenizer, device)
    model.float().eval()
    records, contexts = load_set(frozen)
    rows, started = read_jsonl(rows_dir / "rows.jsonl"), time.time()
    for n, row in enumerate(rows, 1):
        context = contexts.get(row["id"], "") if info["context"] else ""
        row["prior"] = prior_scores(model, processor, context, row["gen"], device)
        if n % 500 == 0:
            print(f"{n}/{len(rows)} {time.time() - started:.0f}s", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "rows.jsonl", rows)
    info["prior"] = {"checkpoint": Path(prior).name, "parent": Path(prior).parent.name,
                     "sha256": meta["checkpoint_sha256"]}
    info["reprior_of"] = f"{rows_dir.parent.name}/{rows_dir.name}"
    (out / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return info


def ranked(row, lam):
    if not row["gen"]:
        return []
    if lam == 0 or row["prior"] is None:
        return row["gen"]
    order = sorted(range(len(row["gen"])), key=lambda i: (-(row["genScores"][i] + lam * row["prior"][i]), i))
    return [row["gen"][i] for i in order]


def hit(row, lam, k=1):
    return row["text"] in ranked(row, lam)[:k]


def table(rows, lam):
    out = {}
    for kind in ("all",) + KINDS:
        part = [r for r in rows if kind == "all" or r["kind"] == kind]
        if not part:
            continue
        out[kind] = {"n": len(part),
                     **{f"top{k}": 100 * sum(hit(r, 0, k) for r in part) / len(part) for k in (1, 3, 10)},
                     "rescored_top1": 100 * sum(hit(r, lam, 1) for r in part) / len(part),
                     "rows_per_input": sum(r["rows"] for r in part) / len(part)}
    return out


def sign_flip(base, new, lam_base, lam_new):
    """Paired Top-1 on shared inputs, clustered by sentence id."""
    key = lambda r: (r["id"], r["input"])
    b = {key(r): r for r in base}
    clusters, gained, lost = {}, 0, 0
    shared = 0
    for r in new:
        o = b.get(key(r))
        if o is None:
            continue
        shared += 1
        d = int(hit(r, lam_new)) - int(hit(o, lam_base))
        gained += d > 0
        lost += d < 0
        clusters[r["id"]] = clusters.get(r["id"], 0) + d
    values = list(clusters.values())
    observed = abs(sum(values))
    rng = random.Random(SIGN_FLIP_SEED)
    extreme = sum(abs(sum(v if rng.random() < 0.5 else -v for v in values)) >= observed
                  for _ in range(PERMUTATIONS))
    return {"inputs": shared, "diff": 100 * (gained - lost) / max(shared, 1), "gained": gained, "lost": lost,
            "p": (extreme + 1) / (PERMUTATIONS + 1)}


def report(runs, lam=1.6, base=None):
    """runs: {name: directory}. Markdown text."""
    data = {name: read_jsonl(Path(d) / "rows.jsonl") for name, d in runs.items()}
    lines = [f"# ime-pinyin-zh-v1 development: pinyin generation (λ = {lam})", "",
             "| run | kind | n | Top-1 | Top-3 | Top-10 | rescored Top-1 | rows/input |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, rows in data.items():
        for kind, t in table(rows, lam).items():
            lines.append(f"| {name} | {kind} | {t['n']} | {t['top1']:.1f} | {t['top3']:.1f} | {t['top10']:.1f} | "
                         f"{t['rescored_top1']:.1f} | {t['rows_per_input']:.1f} |")
    if base:
        lines += ["", f"Paired Top-1 against `{base}` (generation order / rescored; sentence-cluster sign-flip):", "",
                  "| run | order | diff | +/- | p |", "|---|---|---:|---|---:|"]
        for name, rows in data.items():
            if name == base:
                continue
            for label, lb, ln in (("gen", 0, 0), ("rescored", lam, lam), ("rescored vs base gen", 0, lam)):
                s = sign_flip(data[base], rows, lb, ln)
                lines.append(f"| {name} | {label} | {s['diff']:+.1f} | +{s['gained']}/-{s['lost']} | {s['p']:.2g} |")
    return "\n".join(lines) + "\n"
