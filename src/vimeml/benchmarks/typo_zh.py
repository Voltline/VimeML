"""Public keyboard-typo correction set ime-typo-zh-v1 (development only), and its scoring rules.

build: every development sentence of ime-pinyin-zh-v1-r1, full pinyin only, in three variants:
clean, one typo and two typos (zh_typo rules and weights of data-zh-pinyin-typo.toml, own seeds
"ime-typo-zh-v1/<id>/<n>"). Per input a candidate pool, made once by vimeml-tiny-zh-v1 and frozen:
  base         top 10 of the input itself (zh_search, lm mode, beam 16, the record's in-field context)
  corrections  every single edit that undoes one typo of the data rules (delete a letter, insert any
               letter, swap two adjacent letters, replace a letter by a QWERTY neighbour) and leaves
               valid full pinyin, each with its search top 1; distinct texts other than base[0], in
               spelling order. About 65 spellings per input, and the typed-correct spelling is always
               among them for one-typo inputs.
The pool costs about 140 s per input at beam 16, so build runs in shards (sentence index
mod N) and merge joins them.
Every model is scored on the same pool, so the result does not depend on who generated it.

score: per input and model, for base[0] and every correction text
  lm    LM score after the context (zh13_lm_scores.score_item, the production rule)
  task  log P(output + EOS | [2,4] + context + [5] + the typed keys + [6]) (three-task models)
The context is trimmed once per input, from the left, to fit the longest candidate.

Display rule (Vime: the correction never goes first): delta = best correction - base[0] under a rule;
the correction is shown second when delta >= margin, otherwise the list is base[0], base[1].
  typo inputs  @2 = reference among the first two; shown-correct = the shown correction is the reference
  clean inputs false display = a correction is shown
calibrate: the margin with false display <= 10 % on clean inputs that maximizes @2 on typo inputs.
The test split of ime-pinyin-zh-v1 is never read.
"""
import json
import math
import random
import time
from pathlib import Path

VERSION = "ime-typo-zh-v1-r1"
VARIANTS = ("clean", "typo1", "typo2")
LETTERS = "abcdefghijklmnopqrstuvwxyz"
FALSE_DISPLAY_LIMIT = 0.10
PERMUTATIONS = 20000
SIGN_FLIP_SEED = 1118


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---- build ---------------------------------------------------------------------------------------------

def typo_input(record, count, config):
    """(keys, edits) with `count` typos in different syllables; seeds ime-typo-zh-v1/<id>/<count>[/<retry>]."""
    from vimeml.posttraining.zh_typo import add_typos
    units = record["full_pinyin"].split()
    if "".join(units) != record["full_input"]:
        raise ValueError(f"{record['id']}: full_pinyin does not spell full_input")
    config = dict(config, two_share=1.0 if count == 2 else 0.0)
    for retry in range(20):
        rng = random.Random(f"ime-typo-zh-v1/{record['id']}/{count}" + (f"/{retry}" if retry else ""))
        out = add_typos(rng, record["full_input"], units, config)
        if out is not None and len(out[1]) == count:
            return out
    raise ValueError(f"{record['id']}: no {count}-typo input")


def single_edits(keys, syllables):
    """Every inverse of one typo edit (delete, insert any letter, adjacent swap, neighbour substitute)
    that is valid full pinyin, sorted, without keys itself."""
    from vimeml.posttraining.zh_typo import NEIGHBOURS, full_pinyin
    out = set()
    for i in range(len(keys)):
        out.add(keys[:i] + keys[i + 1:])
        out.update(keys[:i] + c + keys[i + 1:] for c in NEIGHBOURS.get(keys[i], ""))
        if i + 1 < len(keys):
            out.add(keys[:i] + keys[i + 1] + keys[i] + keys[i + 2:])
    for i in range(len(keys) + 1):
        out.update(keys[:i] + c + keys[i:] for c in LETTERS)
    out.discard(keys)
    return sorted(s for s in out if s and full_pinyin(s, syllables))


def build(frozen, out, checkpoint, tokenizer, readings, typo_config, device="cpu", limit=0, beam=16, shard=(0, 1)):
    """One shard: sentences whose index mod shard[1] is shard[0]; merge() joins the shard directories."""
    import sentencepiece as spm  # noqa: F401
    from vimeml.benchmarks.pinyin_gen_zh import load_set
    from vimeml.benchmarks.zh13_lm_scores import load
    from vimeml.deployment import zh_search as zs
    from vimeml.deployment.zh_torch_backend import ModelBatchBackend
    from vimeml.posttraining.zh_readings import ReadingTable
    from vimeml.training.data import file_sha

    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    model, processor, meta = load(checkpoint, tokenizer, device)
    backend = ModelBatchBackend(model, device)
    table = ReadingTable(readings)
    index = zs.Index(processor, table)
    records, contexts = load_set(frozen, limit)
    index_of, parts = shard
    records = [r for n, r in enumerate(records) if n % parts == index_of]
    top1 = {}

    def search(keys, context, k):
        ranked, _ = zs.search(backend, processor, index, keys, context, "lm", beam, k)
        return [t for t, _, _ in ranked]

    items, started, searches = [], time.time(), 0
    for n, record in enumerate(records, 1):
        context = contexts.get(record["id"], "")
        variants = [("clean", record["full_input"], [])]
        variants += [(f"typo{c}", *typo_input(record, c, typo_config)) for c in (1, 2)]
        for variant, keys, edits in variants:
            base = search(keys, context, 10)
            searches += 1
            corrections, seen = [], set(base[:1])
            for spelling in single_edits(keys, table.syllables):
                key = (spelling, context)
                if key not in top1:
                    found = search(spelling, context, 1)
                    top1[key] = found[0] if found else None
                    searches += 1
                text = top1[key]
                if text and text not in seen:
                    seen.add(text)
                    corrections.append({"spelling": spelling, "text": text})
            items.append({"id": record["id"], "variant": variant, "input": keys, "edits": edits,
                          "text": record["text"], "source": record["source"], "base": base,
                          "corrections": corrections})
        if n % 25 == 0 or n == len(records):
            print(f"{n}/{len(records)} sentences, {searches} searches, {time.time() - started:.0f}s", flush=True)
    out.mkdir(parents=True)
    write_jsonl(out / "development.jsonl", items)
    manifest = {"version": VERSION, "split": "development", "shard": list(shard), "sentences": len(records),
                "items": len(items), "limit": limit, "typo_config": typo_config,
                "pool": {"checkpoint": Path(checkpoint).parent.name, "checkpoint_sha256": meta["checkpoint_sha256"],
                         "mode": "lm", "beam": beam, "base_topk": 10, "searches": searches},
                "pinyin_set_manifest_sha256": file_sha(Path(frozen) / "manifest.json"),
                "readings_sha256": file_sha(readings),
                "code_sha256": {name: file_sha(Path(__file__).resolve().parents[1] / name)
                                for name in ("benchmarks/typo_zh.py", "posttraining/zh_typo.py")},
                "seconds": round(time.time() - started, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", "utf-8")
    return manifest


SAME = ("version", "limit", "typo_config", "pool", "pinyin_set_manifest_sha256", "readings_sha256", "code_sha256")


def merge(shards, out, frozen):
    """Join shard directories into one set, items in the order of the pinyin set's development records."""
    from vimeml.benchmarks.pinyin_gen_zh import load_set
    from vimeml.training.data import file_sha
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    manifests = [json.loads((Path(d) / "manifest.json").read_text("utf-8")) for d in shards]
    parts = manifests[0]["shard"][1]
    if sorted(m["shard"][0] for m in manifests) != list(range(parts)) or any(m["shard"][1] != parts for m in manifests):
        raise ValueError("shards are missing or repeated")
    for key in SAME:
        if any(m[key] != manifests[0][key] for m in manifests) and key != "pool":
            raise ValueError(f"shards differ in {key}")
        if key == "pool" and any({**m[key], "searches": 0} != {**manifests[0][key], "searches": 0} for m in manifests):
            raise ValueError("shards differ in pool")
    if file_sha(Path(frozen) / "manifest.json") != manifests[0]["pinyin_set_manifest_sha256"]:
        raise ValueError("pinyin set differs from the one the shards were built from")
    records, _ = load_set(frozen, manifests[0]["limit"])
    order = {r["id"]: n for n, r in enumerate(records)}
    items = [i for d in shards for i in read_jsonl(Path(d) / "development.jsonl")]
    items.sort(key=lambda i: (order[i["id"]], VARIANTS.index(i["variant"])))
    if len(items) != 3 * len(records) or len({(i["id"], i["variant"]) for i in items}) != len(items):
        raise ValueError("shard items do not cover the set")
    out.mkdir(parents=True)
    write_jsonl(out / "development.jsonl", items)
    first = manifests[0]
    manifest = {**{k: first[k] for k in SAME}, "split": "development", "sentences": len(records), "items": len(items),
                "coverage": {v: oracle([i for i in items if i["variant"] == v]) for v in VARIANTS},
                "corrections_per_input": {v: sum(len(i["corrections"]) for i in items if i["variant"] == v)
                                          / max(1, sum(i["variant"] == v for i in items)) for v in VARIANTS},
                "shards": len(manifests), "searches": sum(m["pool"]["searches"] for m in manifests),
                "build_seconds": round(sum(m["seconds"] for m in manifests), 1),
                "development_sha256": file_sha(out / "development.jsonl")}
    manifest["pool"] = {**first["pool"], "searches": manifest.pop("searches")}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", "utf-8")
    return manifest


def oracle(items):
    """Share of inputs whose reference is base[0], base[1] or any correction (the best any rule can do at @2
    is lower: one correction slot)."""
    n = len(items)
    return {"n": n,
            "base_top1": sum(i["base"][:1] == [i["text"]] for i in items) / max(n, 1),
            "base_top2": sum(i["text"] in i["base"][:2] for i in items) / max(n, 1),
            "in_corrections": sum(any(c["text"] == i["text"] for c in i["corrections"]) for i in items) / max(n, 1)}


# ---- score ---------------------------------------------------------------------------------------------

def task_scores(model, processor, context, keys, candidates, device="cpu", limit=128):
    """log P(output + EOS | [2,4] + context + [5] + keys + [6]) per candidate, one batch."""
    import torch
    from vimeml.posttraining.zh_pinyin import CONTROL
    r = processor.encode(keys)
    outs = [processor.encode(c) for c in candidates]
    if processor.decode(r) != keys or any(processor.decode(o) != c for o, c in zip(outs, candidates)) \
            or any(i in CONTROL for i in r + [t for o in outs for t in o]):
        raise ValueError("keys or a candidate do not roundtrip")
    budget = limit + 1 - (len(r) + max(map(len, outs)) + 5)
    if budget < 0:
        raise ValueError("keys + candidate exceed the context")
    ctx, removed = processor.encode(context), 0
    while len(ctx) > budget or any(i in CONTROL for i in ctx):
        removed += 1
        ctx = processor.encode(context[removed:])
    head = [2, 4] + ctx + [5] + r + [6]
    seqs = [head + o + [3] for o in outs]
    width = max(len(s) for s in seqs) - 1
    inputs = torch.zeros((len(seqs), width), dtype=torch.long)
    targets = torch.zeros_like(inputs)
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    for row, s in enumerate(seqs):
        inputs[row, :len(s) - 1] = torch.tensor(s[:-1])
        targets[row, :len(s) - 1] = torch.tensor(s[1:])
        mask[row, len(head) - 1:len(s) - 1] = True
    with torch.inference_mode():
        logits = model(inputs.to(device)).float()
    values = logits.log_softmax(-1).gather(-1, targets.to(device).unsqueeze(-1)).squeeze(-1).cpu()
    scores = [float(values[row][mask[row]].sum()) for row in range(len(seqs))]
    if not all(math.isfinite(v) for v in scores):
        raise ValueError("Non-finite score")
    return scores


def score(set_dir, checkpoint, tokenizer, out, frozen, rules=("lm", "task"), device="cpu", limit=0):
    """One row per item: candidates = [base[0]] + correction texts; scores per rule (None if a rule fails)."""
    import torch
    from vimeml.benchmarks.pinyin_gen_zh import load_set
    from vimeml.benchmarks.zh13_lm_scores import load, score_item
    from vimeml.training.data import file_sha

    set_dir, out = Path(set_dir), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    manifest = json.loads((set_dir / "manifest.json").read_text("utf-8"))
    items = read_jsonl(set_dir / "development.jsonl")
    items = items[:limit] if limit else items
    model, processor, meta = load(checkpoint, tokenizer, device)
    model.float().eval()
    if file_sha(Path(frozen) / "manifest.json") != manifest["pinyin_set_manifest_sha256"]:
        raise ValueError("pinyin set differs from the one the typo set was built from")
    _, contexts = load_set(frozen)
    rows, started = [], time.time()
    for n, item in enumerate(items, 1):
        context = contexts.get(item["id"], "")
        candidates = item["base"][:1] + [c["text"] for c in item["corrections"]]
        row = {"id": item["id"], "variant": item["variant"]}
        if not candidates or len(candidates) == 1:
            rows.append({**row, **{rule: None for rule in rules}})
            continue
        for rule in rules:
            try:
                with torch.inference_mode():
                    if rule == "lm":
                        row[rule] = score_item(model, processor, context, candidates, device)["scores"]
                    else:
                        row[rule] = task_scores(model, processor, context, item["input"], candidates, device)
            except ValueError:
                row[rule] = None
        rows.append(row)
        if n % 500 == 0:
            print(f"{n}/{len(items)} {time.time() - started:.0f}s", flush=True)
    out.mkdir(parents=True)
    write_jsonl(out / "rows.jsonl", rows)
    info = {"checkpoint": str(checkpoint), "checkpoint_sha256": meta["checkpoint_sha256"], "step": meta["step"],
            "rules": list(rules), "set": manifest["version"], "set_manifest_sha256": file_sha(set_dir / "manifest.json"),
            "items": len(rows), "limit": limit, "device": device, "seconds": round(time.time() - started, 1)}
    (out / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return info


# ---- metrics -------------------------------------------------------------------------------------------

def decide(item, row, rule, margin):
    """(shown correction text or None, the first two displayed texts)."""
    scores = row.get(rule) if row else None
    if scores and len(scores) > 1:
        best = max(range(1, len(scores)), key=lambda i: (scores[i], -i))
        if scores[best] - scores[0] >= margin:
            text = item["corrections"][best - 1]["text"]
            return text, item["base"][:1] + [text]
    return None, item["base"][:2]


def deltas(items, rows, rule):
    """Best-correction delta per item (None when it has no scored correction)."""
    out = []
    for item in items:
        s = rows.get((item["id"], item["variant"]), {}).get(rule)
        out.append(max(s[1:]) - s[0] if s and len(s) > 1 else None)
    return out


def metrics(items, rows, rule, margin):
    result = {}
    for variant in VARIANTS + ("typo",):
        part = [i for i in items if i["variant"] == variant or (variant == "typo" and i["variant"] != "clean")]
        if not part:
            continue
        shown = hits = correct = 0
        for item in part:
            text, first2 = decide(item, rows.get((item["id"], item["variant"])), rule, margin)
            shown += text is not None
            correct += text == item["text"]
            hits += item["text"] in first2
        n = len(part)
        result[variant] = {"n": n, "at2": 100 * hits / n, "top1": 100 * sum(i["base"][:1] == [i["text"]] for i in part) / n,
                           "shown": 100 * shown / n, "shown_correct": 100 * correct / n}
    return result


def calibrate(items, rows, rule, limit=FALSE_DISPLAY_LIMIT):
    """Margin with clean false display <= limit that maximizes typo @2 (ties: the larger margin)."""
    clean = [d for d, i in zip(deltas(items, rows, rule), items) if i["variant"] == "clean"]
    every = sorted({round(d, 4) for d in deltas(items, rows, rule) if d is not None})
    best = None
    for m in every + [math.inf]:
        shown = sum(d is not None and d >= m for d in clean) / max(len(clean), 1)
        if shown > limit:
            continue
        at2 = metrics(items, rows, rule, m)["typo"]["at2"]
        if best is None or at2 >= best[1]:
            best = (m, at2)
    return best[0]


def hits_by_item(items, rows, rule, margin):
    return {(i["id"], i["variant"]): int(i["text"] in decide(i, rows.get((i["id"], i["variant"])), rule, margin)[1])
            for i in items}


def sign_flip(items, a, b):
    """Paired @2 on typo inputs, clustered by sentence. a, b: {(id, variant): hit}."""
    clusters, gained, lost = {}, 0, 0
    for i in items:
        if i["variant"] == "clean":
            continue
        key = (i["id"], i["variant"])
        d = b[key] - a[key]
        gained += d > 0
        lost += d < 0
        clusters[i["id"]] = clusters.get(i["id"], 0) + d
    values = list(clusters.values())
    observed = abs(sum(values))
    rng = random.Random(SIGN_FLIP_SEED)
    extreme = sum(abs(sum(v if rng.random() < 0.5 else -v for v in values)) >= observed for _ in range(PERMUTATIONS))
    n = sum(i["variant"] != "clean" for i in items)
    return {"diff": 100 * (gained - lost) / max(n, 1), "gained": gained, "lost": lost,
            "p": (extreme + 1) / (PERMUTATIONS + 1)}


def load_rows(directory):
    return {(r["id"], r["variant"]): r for r in read_jsonl(Path(directory) / "rows.jsonl")}


def report(set_dir, runs, base=None, fixed_margin=3.0):
    """runs: {name: (directory, rule)}; each run at the fixed margin (lm rule) and at its calibrated margin."""
    items = read_jsonl(Path(set_dir) / "development.jsonl")
    lines = [f"# {VERSION} development", "",
             "| run | rule | margin | typo @2 | typo1 @2 | typo2 @2 | typo shown correct | clean false display | clean @2 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    hits = {}
    for name, (directory, rule) in runs.items():
        rows = load_rows(directory)
        margins = [("calibrated", calibrate(items, rows, rule))]
        if rule == "lm":
            margins.insert(0, ("fixed", fixed_margin))
        for label, m in margins:
            t = metrics(items, rows, rule, m)
            hits[f"{name}/{label}"] = hits_by_item(items, rows, rule, m)
            lines.append(f"| {name} | {rule} | {m:.2f} ({label}) | {t['typo']['at2']:.1f} | {t['typo1']['at2']:.1f} | "
                         f"{t['typo2']['at2']:.1f} | {t['typo']['shown_correct']:.1f} | {t['clean']['shown']:.1f} | "
                         f"{t['clean']['at2']:.1f} |")
    cov = {v: oracle([i for i in items if i["variant"] == v]) for v in VARIANTS}
    lines += ["", "Pool: " + ", ".join(f"{v} base top-1 {100 * c['base_top1']:.1f} / reference among corrections "
                                       f"{100 * c['in_corrections']:.1f}" for v, c in cov.items())]
    if base:
        lines += ["", f"Paired typo @2 against `{base}` (sentence-cluster sign-flip, {PERMUTATIONS} permutations):", "",
                  "| run | diff | +/- | p |", "|---|---:|---|---:|"]
        for key, h in hits.items():
            if key == base:
                continue
            s = sign_flip(items, hits[base], h)
            lines.append(f"| {key} | {s['diff']:+.1f} | +{s['gained']}/-{s['lost']} | {s['p']:.2g} |")
    return "\n".join(lines) + "\n"
