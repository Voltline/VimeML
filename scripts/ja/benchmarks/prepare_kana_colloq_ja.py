"""Build and freeze the colloquial Japanese kana benchmark ime-kana-ja-colloq-v1 (Stage 3).

Sources (evaluation-only splits, never in any training mix):
- `web_colloq`: paragraphs from `ja_colloq.py select` in the `heldout` split
  (`doc_hash("ja-colloq-heldout", doc_id) < 0.02`) with `p_chat >= p_chat_min`.
  At most one item per document so no document ever straddles development and test.

Pipeline (mirrors prepare_pinyin_colloq_zh.py and prepare_kana_ime_ja.py):
  build   select-dir -> OUT/candidate-pool.json + OUT/items.jsonl
  screen  OUT/candidate-pool.json + training corpora -> OUT/overlap-audit.json
  freeze  OUT/candidate-pool.json + OUT/overlap-audit.json -> FROZEN/{development,test}/
          (standard_ime.track_files layout + manifest.json + SHA256SUMS; test sealed)

    python -X utf8 scripts/ja/benchmarks/prepare_kana_colloq_ja.py build \\
        --select-dir outputs/ja-colloq/select-v1 --out outputs/benchmarks/ime-kana-ja-colloq-v1-build
    python -X utf8 scripts/ja/benchmarks/prepare_kana_colloq_ja.py screen \\
        --build-dir outputs/benchmarks/ime-kana-ja-colloq-v1-build \\
        --train-files outputs/corpus-fast-v1/train.txt outputs/ja-colloq/export-colloq-v1/train/*.parquet
    python -X utf8 scripts/ja/benchmarks/prepare_kana_colloq_ja.py freeze \\
        --build-dir outputs/benchmarks/ime-kana-ja-colloq-v1-build \\
        --out artifacts/benchmarks/ime-kana-ja-colloq-v1 --per-split 600
"""
import argparse
import collections
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/ja/corpus"))
sys.path.insert(0, str(ROOT / "scripts/ja/benchmarks"))

import ja_colloq as jc  # noqa: E402
import kana_samples as ks  # noqa: E402
from reading_pairs import load_table  # noqa: E402
from vimeml.training.data import write_json  # noqa: E402

VERSION = "ime-kana-ja-colloq-v1"
SEED = 20261201
ROLES = ("development", "test")
DEFAULT_P_CHAT = 0.5
DEFAULT_SAMPLE_CFG = {
    "target_chars": [4, 30],
    "reading_chars": [4, 64],
    "context_chars": 64,
    "run_chars": [6, 100],
    "end_at_run_end_probability": 0.5,
    "runs_tried": 6,
}
URL_OR_MARKUP = re.compile(r"https?://|@[A-Za-z0-9_]|<[^>]+>|\[MASK\]|\[PERSON\]")


def has_colloquial_marker(text):
    return bool(
        jc.FINAL_PARTICLES.search(text)
        or jc.CONTRACTIONS.search(text)
        or jc.CASUAL_LAUGH.search(text)
    )


def length_band(target):
    n = len(target)
    return "short" if n <= 10 else "medium" if n <= 20 else "long"


def to_katakana(text):
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in text)


def assign_role(group_id, seed=SEED):
    h = ks.doc_hash(f"{VERSION}/role/{seed}", group_id)
    return ROLES[h % 2]


def extract_colloquial_case(record, rng, sample_cfg=DEFAULT_SAMPLE_CFG, analyser=None):
    """Extract one kana-conversion span from a heldout colloquial paragraph.

    Prefers spans that contain an explicit Japanese colloquial marker (sentence-final
    particle, contraction, or casual laugh) when available.
    """
    original = record["text"]
    text = ks.clean(original)
    if not 8 <= len(text) <= 512:
        return None, "message_length"
    if URL_OR_MARKUP.search(text):
        return None, "url_markup_or_placeholder"

    analyse_fn = analyser or ks.analyse
    runs = [
        m for m in ks.RUN.finditer(text)
        if sample_cfg["run_chars"][0] <= len(m.group()) <= sample_cfg["run_chars"][1]
        and ks.KANJI.search(m.group())
    ]
    if not runs:
        return None, "no_typeable_run"

    rng.shuffle(runs)
    colloq_candidates, plain_candidates = [], []
    for m in runs[:sample_cfg["runs_tried"]]:
        run = m.group()
        segs = analyse_fn(run)
        for i, (s, _, _, ok) in enumerate(segs):
            if not ok:
                continue
            for j in range(i + 1, len(segs) + 1):
                reading_hira = ks.reading_of(segs, i, j)
                if reading_hira is None:
                    break
                target = run[s:segs[j - 1][1]]
                if not ((j == len(segs) or segs[j][3]) and ks.fits(sample_cfg, target, reading_hira)
                        and ks.KANJI.search(target)):
                    continue
                start = m.start() + s
                end = start + len(target)
                ctx = text[max(0, start - sample_cfg["context_chars"]):start]
                cand = (start, end, ctx, reading_hira, target)
                if has_colloquial_marker(target):
                    colloq_candidates.append(cand)
                else:
                    plain_candidates.append(cand)

    if colloq_candidates:
        start, end, ctx, reading_hira, target = rng.choice(colloq_candidates)
        marker = True
    elif plain_candidates:
        start, end, ctx, reading_hira, target = rng.choice(plain_candidates)
        marker = False
    else:
        return None, "no_supported_span"

    query_kata = to_katakana(reading_hira)
    doc_id = record["document_id"]
    group_id = record.get("group_id") or doc_id
    case_id = record.get("case_id") or f"{doc_id}#{record.get('para', 0)}"
    rule_band = record.get("rule_band")
    rule_score = record.get("rule_score")
    if rule_band is None or rule_score is None:
        f = jc.features(text)
        if f is not None:
            rule_band = jc.band(f[2])
            rule_score = round(float(f[2]), 3)
        else:
            rule_band = rule_band or "unknown"
            rule_score = 0.0
    return {
        "index": "pending",
        "context_text": ctx,
        "input": query_kata,
        "reading_hiragana": reading_hira,
        "expected_output": [target],
        "category": record.get("source", "web_colloq"),
        "provenance": {
            "source": record.get("source", "web_colloq"),
            "case_id": case_id,
            "document_id": doc_id,
            "group_id": group_id,
            "author_id": group_id,
            "role": assign_role(group_id),
            "rule_band": rule_band,
            "rule_score": float(rule_score),
            "p_chat": float(record.get("p_chat", 0.0)),
            "p_spoken": float(record.get("p_spoken", 0.0)),
            "colloquial_marker": marker,
            "original_message": original,
            "cleaned_message": text,
            "span_start": start,
            "span_end": end,
            "track": "colloquial",
        },
        "review": {
            "status": "d4_d5_dictionary_reading",
            "human_native_review": False,
            "not_formal_gold": True,
            "acceptable_forms_exhaustive": False,
        },
    }, None


def load_heldout_records(select_dir, p_chat_min=DEFAULT_P_CHAT, per_band_docs=6000, max_pool=0, seed=SEED):
    """Stream heldout rows with p_chat >= p_chat_min, 1 paragraph/doc, reservoir-sampled per rule band."""
    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    select_dir = Path(select_dir)
    files = sorted(select_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files in {select_dir}")

    band_names = tuple(b for b, *_ in jc.BANDS)
    rng = np.random.default_rng(seed)
    reservoirs = {b: [] for b in band_names}
    seen_band = {b: 0 for b in band_names}
    seen_docs = set()

    for path in files:
        pf = pq.ParquetFile(path)
        for g in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(g, columns=["id", "para", "text", "p_chat", "p_spoken", "split"])
            mask = pc.and_(
                pc.equal(t["split"], "heldout"),
                pc.greater_equal(t["p_chat"], p_chat_min),
            )
            filtered = t.filter(mask).to_pylist()
            for row in filtered:
                doc_id = str(row["id"])
                if doc_id in seen_docs:
                    continue
                f = jc.features(row["text"])
                if f is None:
                    continue
                seen_docs.add(doc_id)
                rule_score = round(float(f[2]), 3)
                band = jc.band(f[2])
                rec = {
                    "source": "web_colloq",
                    "case_id": f"{doc_id}#{row['para']}",
                    "document_id": doc_id,
                    "group_id": f"web_colloq:{doc_id}",
                    "para": int(row["para"]),
                    "text": row["text"],
                    "rule_band": band,
                    "rule_score": rule_score,
                    "p_chat": float(row["p_chat"]),
                    "p_spoken": float(row["p_spoken"]),
                }
                if per_band_docs <= 0:
                    reservoirs[band].append(rec)
                    if max_pool and sum(len(v) for v in reservoirs.values()) >= max_pool:
                        return [r for b in band_names for r in reservoirs[b]]
                else:
                    seen_band[band] += 1
                    res = reservoirs[band]
                    if len(res) < per_band_docs:
                        res.append(rec)
                    else:
                        slot = int(rng.integers(0, seen_band[band]))
                        if slot < per_band_docs:
                            res[slot] = rec

    records = [r for b in band_names for r in reservoirs[b]]
    return records[:max_pool] if max_pool else records


def build_pool(records, seed=SEED, pool_cap_per_role=2500, per_band_cap=0, earlier_queries=(), analyser=None):
    rng = random.Random(seed)
    ordered = list(records)
    rng.shuffle(ordered)
    earlier_q = set(earlier_queries)
    pool, rejects = [], collections.Counter()
    counts = collections.Counter()
    by_band_role = collections.Counter()
    seen_sigs = set()
    half_band = per_band_cap // 2 if per_band_cap > 0 else 0
    band_names = tuple(b for b, *_ in jc.BANDS)

    for rec in ordered:
        role = assign_role(rec["group_id"], seed=seed)
        band = rec.get("rule_band") or "unknown"
        if half_band > 0:
            if by_band_role[(band, role)] >= half_band:
                if all(by_band_role[(b, r)] >= half_band for b in band_names for r in ROLES):
                    break
                continue
        elif counts[role] >= pool_cap_per_role:
            if all(counts[r] >= pool_cap_per_role for r in ROLES):
                break
            continue
        case, reason = extract_colloquial_case(rec, rng, analyser=analyser)
        if case is None:
            rejects[reason] += 1
            continue
        sig = jc.normalized(case["provenance"]["cleaned_message"])
        qkey = (case["context_text"], case["input"])
        if qkey in earlier_q or sig in seen_sigs or qkey in seen_sigs:
            rejects["duplicate_or_earlier_query"] += 1
            continue
        seen_sigs.update((sig, qkey))
        pool.append(case)
        counts[role] += 1
        by_band_role[(case["provenance"]["rule_band"], role)] += 1
    return pool, {
        "pool_counts": dict(counts),
        "by_band_role": {f"{b}/{r}": n for (b, r), n in sorted(by_band_role.items())},
        "rejections": dict(rejects),
    }


def iter_corpus_texts(path):
    """Yield (doc_id_or_None, text) from .txt, .jsonl, or .parquet training files."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        cols = [c for c in ("id", "text") if c in pf.schema.names]
        for g in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(g, columns=cols).to_pydict()
            ids = t.get("id") or [None] * len(t["text"])
            for doc_id, text in zip(ids, t["text"]):
                yield (None if doc_id is None else str(doc_id).split("#", 1)[0]), text
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                obj = json.loads(line)
                doc_id = obj.get("id")
                yield (None if doc_id is None else str(doc_id).split("#", 1)[0]), obj.get("text", "")
    else:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                yield None, line


def _anchor_matches(normed, anchors, matcher):
    if matcher is not None:
        yield from matcher.iter(normed)
        return
    for pos in range(len(normed) - 11):
        sub = normed[pos:pos + 12]
        ids = anchors.get(sub)
        if ids is not None:
            yield pos + 11, (sub, ids)


def screen_overlap(pool, train_files):
    """12-char normalized anchor containment or SequenceMatcher >= 0.85 against training corpora."""
    texts = [jc.normalized(r["provenance"]["cleaned_message"]) for r in pool]
    own_docs = [str(r["provenance"]["document_id"]) for r in pool]
    raw_anchors = collections.defaultdict(set)
    for idx, text in enumerate(texts):
        if len(text) >= 12:
            raw_anchors[text[:12]].add(idx)
            raw_anchors[text[-12:]].add(idx)
    anchors = {k: tuple(sorted(v)) for k, v in raw_anchors.items()}
    try:
        import ahocorasick

        matcher = ahocorasick.Automaton()
        for anchor, ids in anchors.items():
            matcher.add_word(anchor, (anchor, ids))
        matcher.make_automaton()
    except ImportError:
        matcher = None

    excluded, matches, file_counts = set(), [], collections.Counter()
    for path in train_files:
        name = str(path)
        for row_num, (doc_id, raw) in enumerate(iter_corpus_texts(path), 1):
            file_counts[name] += 1
            normed = jc.normalized(raw)
            if len(normed) < 12:
                continue
            for end, (anchor, ids) in _anchor_matches(normed, anchors, matcher):
                for idx in ids:
                    if idx in excluded:
                        continue
                    if doc_id is not None and doc_id == own_docs[idx]:
                        continue
                    cand = texts[idx]
                    if cand in normed:
                        ratio = 1.0
                    else:
                        loc = end - len(anchor) + 1
                        if anchor == cand[-12:]:
                            loc -= len(cand) - 12
                        window = normed[max(0, loc):max(0, loc) + len(cand)]
                        ratio = difflib.SequenceMatcher(None, cand, window, autojunk=False).ratio()
                    if ratio >= 0.85:
                        excluded.add(idx)
                        matches.append({
                            "pool_position": idx,
                            "document_id": own_docs[idx],
                            "file": name,
                            "row": row_num,
                            "ratio": round(ratio, 3),
                        })
    return excluded, {
        "files": dict(file_counts),
        "excluded": len(excluded),
        "excluded_pool_positions": sorted(excluded),
        "matches": matches,
        "method": "12-character anchors; normalized containment or SequenceMatcher >= 0.85 (skipping own_doc)",
    }


def load_llm_accepted(labels_path, include_label2_dialogue=False):
    """Load DeepSeek confirmation labels and return {case_id: record} for accepted items."""
    accepted = {}
    with Path(labels_path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            r = json.loads(line)
            lbl = int(r["label"])
            kind = str(r.get("kind", ""))
            if lbl == 3 or (include_label2_dialogue and lbl == 2 and kind in ("chat", "fiction")):
                accepted[str(r["id"])] = r
    return accepted


def select_balanced(pool, excluded, per_split=600, seed=SEED + 1):
    """Stratified round-robin selection across rule_band x context x length."""
    rng = random.Random(seed)
    chosen = {}
    for role in ROLES:
        marker_buckets = collections.defaultdict(list)
        plain_buckets = collections.defaultdict(list)
        for idx, row in enumerate(pool):
            if idx in excluded or row["provenance"]["role"] != role:
                continue
            p = row["provenance"]
            key = (p.get("rule_band", "unknown"), bool(row["context_text"]), length_band(row["expected_output"][0]))
            if p.get("colloquial_marker"):
                marker_buckets[key].append(row)
            else:
                plain_buckets[key].append(row)
        for buckets in (marker_buckets, plain_buckets):
            for vals in buckets.values():
                rng.shuffle(vals)

        selected = []
        turn = 0
        while len(selected) < per_split and (any(marker_buckets.values()) or any(plain_buckets.values())):
            primary = marker_buckets if (turn % 2 == 0 or not any(plain_buckets.values())) else plain_buckets
            fallback = plain_buckets if primary is marker_buckets else marker_buckets
            progressed = False
            for key in sorted(set(primary) | set(fallback)):
                if len(selected) >= per_split:
                    break
                bucket = primary.get(key) or fallback.get(key)
                if not bucket:
                    continue
                row = dict(bucket.pop())
                row["index"] = f"{VERSION}-{role}-{len(selected) + 1:04d}"
                selected.append(row)
                progressed = True
            if not progressed:
                break
            turn += 1
        if len(selected) < per_split:
            raise ValueError(f"{role}: only {len(selected)} eligible cases (requested {per_split})")
        chosen[role] = selected
    return chosen


def freeze_benchmark(pool, excluded, out_dir, per_split=600, seed=SEED + 1,
                     llm_labels=None, include_label2_dialogue=False):
    from standard_ime import track_files

    out = Path(out_dir)
    if out.exists():
        raise SystemExit(f"Preserve existing output: {out}; choose a fresh path")

    filtered_pool = pool
    filtered_excluded = set(excluded)
    accepted_map = None
    if llm_labels is not None:
        accepted_map = (
            llm_labels if isinstance(llm_labels, dict)
            else load_llm_accepted(llm_labels, include_label2_dialogue=include_label2_dialogue)
        )
        new_pool, new_excluded = [], set()
        for idx, row in enumerate(pool):
            p = row["provenance"]
            cid = p.get("case_id") or f"{p['document_id']}#0"
            hit = accepted_map.get(cid)
            if hit is None:
                continue
            r_copy = dict(row)
            p_copy = dict(p, llm_label=int(hit["label"]), llm_kind=str(hit.get("kind", "")))
            r_copy["provenance"] = p_copy
            r_copy["review"] = {**row["review"], "status": "llm_confirmed_d4_d5"}
            if idx in filtered_excluded:
                new_excluded.add(len(new_pool))
            new_pool.append(r_copy)
        filtered_pool, filtered_excluded = new_pool, new_excluded

    tracks = select_balanced(filtered_pool, filtered_excluded, per_split=per_split, seed=seed)
    dev_groups = {r["provenance"]["group_id"] for r in tracks["development"]}
    test_groups = {r["provenance"]["group_id"] for r in tracks["test"]}
    if dev_groups & test_groups:
        raise ValueError("Development and test groups overlap")

    out.mkdir(parents=True)
    for role, rows in tracks.items():
        track_files(out / role, rows)

    by_rule_band = {
        role: dict(sorted(collections.Counter(r["provenance"].get("rule_band", "unknown") for r in rows).items()))
        for role, rows in tracks.items()
    }
    strata = {
        role: dict(sorted(collections.Counter(
            f'{r["provenance"].get("rule_band", "unknown")}/'
            f'{"with" if r["context_text"] else "without"}-context/{length_band(r["expected_output"][0])}'
            for r in rows
        ).items()))
        for role, rows in tracks.items()
    }
    manifest = {
        "format": "vimeml_kana_ime_ja_v1",
        "version": VERSION,
        "release_id": uuid.uuid4().hex,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "splits": {role: len(rows) for role, rows in tracks.items()},
        "groups": {"development": len(dev_groups), "test": len(test_groups)},
        "by_rule_band": by_rule_band,
        "strata": strata,
        "llm_confirmed": llm_labels is not None,
        "include_label2_dialogue": bool(include_label2_dialogue),
        "labels_frozen": True,
        "review_completed": False,
        "label_quality": "llm_confirmed_d4_d5" if llm_labels is not None else "d4_d5_dictionary_reading",
        "test_policy": "Sealed: read once, in the preregistered Stage 4 confirmation.",
        "metric": "Top-1 of the displayed list, notation-lenient (vimeml.benchmarks.notation); exact Top-1 also reported",
        "models_scored": False,
        "candidates_exported": False,
    }
    write_json(out / "manifest.json", manifest)
    sums = {
        str(p.relative_to(out)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*")) if p.is_file()
    }
    (out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sums.items()), encoding="utf-8", newline="\n")
    return manifest, sums["manifest.json"]


def cmd_build(args):
    from vimeml.benchmarks.readings import DualReading

    ks._state["reader"] = DualReading()
    ks._state["table"] = load_table(args.table)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = load_heldout_records(
        args.select_dir, p_chat_min=args.p_chat_min, per_band_docs=args.per_band_docs,
        max_pool=args.max_records, seed=args.seed,
    )
    pool, stats = build_pool(
        records, seed=args.seed, pool_cap_per_role=args.pool_cap, per_band_cap=args.per_band,
    )
    write_json(out / "candidate-pool.json", pool)
    llm_cases_path = out / "llm-cases.jsonl"
    part = llm_cases_path.with_name(llm_cases_path.name + ".part")
    with open(part, "w", encoding="utf-8", newline="\n") as f:
        for row in pool:
            p = row["provenance"]
            f.write(json.dumps({
                "id": p["case_id"],
                "band": p["rule_band"],
                "score": p["rule_score"],
                "text": p["cleaned_message"],
            }, ensure_ascii=False) + "\n")
    part.replace(llm_cases_path)
    write_json(out / "build-stats.json", {"records_loaded": len(records), **stats,
                                          "dictionary_versions": ks._state["reader"].versions})
    print(json.dumps({"records_loaded": len(records), "pool": len(pool), **stats}, ensure_ascii=False))


def cmd_screen(args):
    build_dir = Path(args.build_dir)
    pool = json.loads((build_dir / "candidate-pool.json").read_text("utf-8"))
    _, audit = screen_overlap(pool, args.train_files)
    write_json(build_dir / "overlap-audit.json", audit)
    print(json.dumps({"excluded": audit["excluded"], "files": audit["files"]}, ensure_ascii=False))


def cmd_freeze(args):
    build_dir = Path(args.build_dir)
    pool = json.loads((build_dir / "candidate-pool.json").read_text("utf-8"))
    audit_path = build_dir / "overlap-audit.json"
    if not audit_path.is_file() and not args.allow_unscreened:
        raise SystemExit(f"Missing {audit_path}; run `screen` first (or pass --allow-unscreened)")
    excluded = set(json.loads(audit_path.read_text("utf-8"))["excluded_pool_positions"]) if audit_path.is_file() else set()
    manifest, m_sha = freeze_benchmark(
        pool, excluded, args.out, per_split=args.per_split, seed=args.seed + 1,
        llm_labels=args.llm_labels, include_label2_dialogue=args.include_label2_dialogue,
    )
    print(json.dumps({"release_id": manifest["release_id"], "splits": manifest["splits"],
                      "by_rule_band": manifest["by_rule_band"], "strata": manifest["strata"],
                      "manifest_sha256": m_sha}, ensure_ascii=False, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--select-dir", type=Path, required=True)
    b.add_argument("--table", type=Path, default=ROOT / "configs/ja/ja-reading-table.tsv")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--p-chat-min", type=float, default=DEFAULT_P_CHAT)
    b.add_argument("--per-band", type=int, default=1000)
    b.add_argument("--per-band-docs", type=int, default=6000)
    b.add_argument("--pool-cap", type=int, default=2500)
    b.add_argument("--max-records", type=int, default=0)
    b.add_argument("--seed", type=int, default=SEED)
    b.set_defaults(fn=cmd_build)

    s = sub.add_parser("screen")
    s.add_argument("--build-dir", type=Path, required=True)
    s.add_argument("--train-files", nargs="+", required=True)
    s.set_defaults(fn=cmd_screen)

    f = sub.add_parser("freeze")
    f.add_argument("--build-dir", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--per-split", type=int, default=600)
    f.add_argument("--seed", type=int, default=SEED)
    f.add_argument("--llm-labels", type=Path, default=None)
    f.add_argument("--include-label2-dialogue", action="store_true")
    f.add_argument("--allow-unscreened", action="store_true")
    f.set_defaults(fn=cmd_freeze)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
