"""Build an unscored IME benchmark draft with independent and legacy tracks."""
import argparse
import collections
import csv
import difflib
import json
import random
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.ajimee import convert_items
from vimeml.benchmarks.readings import DualReading
from vimeml.training.data import write_json

FORMAT = "vimeml_standard_ime_preparation_v1"
RUN = re.compile(r"[ぁ-ゖァ-ヺー一-龠々〆ヵヶ、。！？「」『』（）・…〜～]+")
KANJI = re.compile(r"[一-龠々]")
CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
CANON = re.compile(r"[^\wぁ-ゖァ-ヺー一-龠々]+")


def normalized(text):
    return CANON.sub("", unicodedata.normalize("NFKC", text)).casefold()


def clean(text):
    # Preserve expressive spelling, kana/kanji choices and punctuation.
    return CONTROLS.sub("", unicodedata.normalize("NFC", text)).strip()


def sources(directory, rng):
    wrime = list(csv.DictReader((directory / "wrime/wrime-ver2.tsv").open(encoding="utf-8", newline=""), delimiter="\t"))
    dialogues = json.loads((directory / "jmultiwoz/dialogues.json").read_text(encoding="utf-8"))
    authors = {"wrime": sorted({r["UserID"] for r in wrime}),
               "jmultiwoz": sorted({r["user_name"] for r in dialogues.values()})}
    roles = {}
    for source, values in authors.items():
        rng.shuffle(values)
        roles[source] = {ident: "development" if i < len(values) // 2 else "blind" for i, ident in enumerate(values)}
    records = []
    for line, row in enumerate(wrime, 2):
        records.append({"source": "wrime", "author_id": row["UserID"], "group_id": f"wrime-author:{row['UserID']}",
                        "document_id": f"wrime-row:{line}", "source_row": line, "text": row["Sentence"],
                        "role": roles["wrime"][row["UserID"]], "original_split": row["Train/Dev/Test"]})
    for ident, dialogue in dialogues.items():
        for turn in dialogue["turns"]:
            if turn["speaker"] != "USER":
                continue
            records.append({"source": "jmultiwoz", "author_id": dialogue["user_name"],
                            "group_id": f"jmultiwoz-user:{dialogue['user_name']}", "document_id": ident,
                            "source_row": turn["turn_id"], "text": turn["utterance"],
                            "role": roles["jmultiwoz"][dialogue["user_name"]],
                            "wizard_id": dialogue["system_name"], "domains": sorted(dialogue["goal"])})
    rng.shuffle(records)
    return records, {name: {role: sum(v == role for v in assignment.values()) for role in ("development", "blind")}
                     for name, assignment in roles.items()}


def extract(record, reader, rng):
    original = record["text"]
    text = clean(original)
    if not 8 <= len(text) <= 512:
        return None, "message_length"
    if re.search(r"https?://|@[A-Za-z0-9_]|<[^>]+>|\[MASK\]|\[PERSON\]", text):
        return None, "url_markup_or_placeholder"
    options = []
    for span in RUN.finditer(text):
        segment = span.group()
        if not 6 <= len(segment) <= 100 or not KANJI.search(segment):
            continue
        words = list(reader.mecab(segment))
        sudachi_starts = {w.begin() for w in reader.sudachi.tokenize(segment, reader.mode)}
        offset = 0
        previous = None
        for word in words:
            compound = previous is not None and (previous.feature.pos1 == "接頭辞" or
                         previous.feature.pos1 == "名詞" and word.feature.pos1 == "名詞")
            if offset in sudachi_starts and not compound and word.feature.pos1 not in ("助詞", "助動詞", "接尾辞", "補助記号"):
                start = span.start() + offset
                target = segment[offset:]
                if start <= 64 and 4 <= len(target) <= 40 and KANJI.search(target):
                    options.append((start, span.end(), segment, offset, target))
            offset += len(word.surface)
            previous = word
    rng.shuffle(options)
    for start, end, segment, offset, target in options[:12]:
        standalone = reader.analyse(target)
        contextual = reader.analyse(segment, offset)
        if (standalone["unknown"] or contextual["unknown"] or not standalone["agreement"] or
                not standalone["kana_only"] or standalone["unidic"] != contextual["unidic"] or
                standalone["sudachi"] != contextual["sudachi"]):
            continue
        query = standalone["unidic"]
        if not 4 <= len(query) <= 64:
            continue
        return {"index": "pending", "context_text": text[:start], "input": query,
                "expected_output": [target], "category": record["source"],
                "provenance": {**{k: v for k, v in record.items() if k != "text"},
                               "original_message": original, "cleaned_message": text,
                               "span_start": start, "span_end": end, "track": "independent",
                               "collection_independence": "Separately collected source; not proof of absence from web training corpora"},
                "review": {"status": "pending", "human_native_review": False, "not_formal_gold": True,
                           "reading_evidence": "Two orthographic dictionaries agree; human confirmation required",
                           "acceptable_forms_exhaustive": False}}, None
    return None, "no_unambiguous_supported_conversion_span"


def build_pool(args, output):
    rng = random.Random(args.seed)
    records, groups = sources(args.sources, rng)
    reader = DualReading()
    quotas = {("wrime", role): 2500 for role in ("development", "blind")}
    quotas.update({("jmultiwoz", role): 1500 for role in ("development", "blind")})
    pool, rejects, counts, documents, signatures = [], collections.Counter(), collections.Counter(), set(), set()
    for position, record in enumerate(records, 1):
        key = record["source"], record["role"]
        if counts[key] >= quotas[key] or (record["source"], record["document_id"]) in documents:
            continue
        case, reason = extract(record, reader, rng)
        if case is None:
            rejects[f"{record['source']}:{reason}"] += 1
            continue
        signature = normalized(case["provenance"]["cleaned_message"])
        query_key = case["context_text"], case["input"]
        if signature in signatures or query_key in signatures:
            rejects["duplicate_message_or_query_context"] += 1
            continue
        signatures.update((signature, query_key))
        documents.add((record["source"], record["document_id"]))
        pool.append(case)
        counts[key] += 1
        if len(pool) % 500 == 0:
            print(f"Prepared {len(pool)} candidate cases", flush=True)
        if all(counts[k] >= target for k, target in quotas.items()):
            break
    stats = {"raw_records": len(records), "pool_counts": {f"{s}/{r}": n for (s,r), n in counts.items()},
             "source_groups": groups, "rejections": dict(rejects), "dictionary_versions": reader.versions,
             "selection": "Seeded source sampling without LM scores or candidate coverage",
             "limits": "Japanese conversion spans only; digits/Latin/emoji may appear in prefix; excluded harder readings require a separate challenge track"}
    write_json(output / "candidate-pool.json", pool)
    write_json(output / "cleaning-report.json", stats)
    return pool


def overlap_audit(pool, output):
    import ahocorasick
    # Two 12-character anchors propose matches; full-text containment or
    # conservative SequenceMatcher confirmation decides the exclusion.
    texts = [normalized(r["provenance"]["cleaned_message"]) for r in pool]
    anchors = collections.defaultdict(set)
    for ident, text in enumerate(texts):
        if len(text) >= 24:
            anchors[text[:12]].add(ident)
            anchors[text[-12:]].add(ident)
    matcher = ahocorasick.Automaton()
    for anchor, ids in anchors.items():
        matcher.add_word(anchor, (anchor, tuple(ids)))
    matcher.make_automaton()
    excluded, matches = set(), []
    files = [ROOT / "outputs/corpus-fast-v1/train.txt", ROOT / "artifacts/token-data/corpus-v3-half/mixed/train.jsonl"]
    scanned = []
    for path in files:
        lines = hit_count = 0
        with path.open(encoding="utf-8") as stream:
            for lines, line in enumerate(stream, 1):
                raw = json.loads(line)["text"] if path.suffix == ".jsonl" else line
                text = normalized(raw)
                checked = set()
                for end, (anchor, ids) in matcher.iter(text):
                    for ident in ids:
                        if ident in excluded or ident in checked:
                            continue
                        checked.add(ident)
                        candidate = texts[ident]
                        reason = None
                        if candidate in text:
                            reason = "normalized_full_message_containment"
                        else:
                            location = end - len(anchor) + 1
                            if anchor == candidate[-12:]:
                                location -= len(candidate) - 12
                            window = text[max(0, location):max(0, location) + len(candidate)]
                            ratio = difflib.SequenceMatcher(None, candidate, window, autojunk=False).ratio()
                            if ratio >= .85:
                                reason = "anchor_proposed_near_duplicate_ratio_ge_0.85"
                        if reason:
                            excluded.add(ident)
                            hit_count += 1
                            matches.append({"pool_position": ident, "source": pool[ident]["provenance"]["source"],
                                            "document_id": pool[ident]["provenance"]["document_id"],
                                            "training_file": str(path.relative_to(ROOT)), "training_line": lines, "reason": reason})
                if lines % 1000000 == 0:
                    print(f"Overlap scan {path.name}: {lines:,} rows; excluded {len(excluded)} candidates", flush=True)
        scanned.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "rows": lines, "matches": hit_count})
    report = {"status": "complete", "files": scanned, "excluded_pool_positions": sorted(excluded), "matches": matches,
              "method": "NFKC/punctuation-stripped full-message containment and >=.85 near match, proposed by first/last 12-character anchors",
              "minimum_checked_message_characters": 24,
              "limits": ["Shorter common messages not classified as contamination", "Not exhaustive semantic/paraphrase detection",
                         "Anchor changes can miss near duplicates", "V3 scan conservatively includes the whole prepared pool, including filtered-out blocks",
                         "No scan of unavailable raw web pages or all possible pretraining sources"],
              "no_full_file_hashes": True}
    write_json(output / "overlap-audit.json", report)
    return excluded


def select_cases(pool, excluded, rng):
    chosen = {}
    used_queries, used_documents, used_messages = set(), set(), set()
    for role in ("development", "blind"):
        result = []
        for source, target in (("wrime", 500), ("jmultiwoz", 250)):
            buckets = collections.defaultdict(list)
            for position, row in enumerate(pool):
                p = row["provenance"]
                if position in excluded or p["source"] != source or p["role"] != role:
                    continue
                length = "short" if len(row["input"]) <= 16 else "medium" if len(row["input"]) <= 32 else "long"
                buckets[(bool(row["context_text"]), length)].append(row)
            for values in buckets.values():
                rng.shuffle(values)
            authors = collections.Counter()
            selected = []
            while len(selected) < target and any(buckets.values()):
                for key in sorted(buckets):
                    if not buckets[key]:
                        continue
                    row = buckets[key].pop()
                    p = row["provenance"]
                    signature = normalized(p["cleaned_message"])
                    query = row["context_text"], row["input"]
                    doc = source, p["document_id"]
                    if authors[p["author_id"]] >= 20 or doc in used_documents or signature in used_messages or query in used_queries:
                        continue
                    row["index"] = f"ime-standard-v1-{role}-{source}-{len(selected)+1:04}"
                    selected.append(row)
                    authors[p["author_id"]] += 1
                    used_documents.add(doc)
                    used_messages.add(signature)
                    used_queries.add(query)
                    if len(selected) == target:
                        break
            if len(selected) != target:
                raise ValueError(f"{source}/{role}: only {len(selected)} eligible cases; no padding or silent quota change.")
            result.extend(selected)
        chosen[role] = result
    return chosen


def legacy_cases(rng):
    path = ROOT / "artifacts/benchmarks/ime-expanded-v21-candidates-v1/development/evaluation_items.json"
    rows = [r for r in json.loads(path.read_text(encoding="utf-8")) if r["provenance"]["source"] == "fineweb"]
    rng.shuffle(rows)
    chosen = rows[:500]
    for row in chosen:
        row["provenance"].update(track="legacy_regression", role="regression", source="fineweb_edu_legacy",
                                 historical_source="ime-expanded-v21-candidates-v1/development")
        row["review"]["historical_labels_unchanged"] = True
    return chosen


def write_track(output, role, rows):
    directory = output / role
    directory.mkdir(exist_ok=True)
    inputs, mapping, stats = convert_items(rows)
    write_json(directory / "evaluation_items.json", rows)
    write_json(directory / "ajimee-input.json", inputs)
    write_json(directory / "case-map.json", mapping)
    write_json(directory / "stats.json", {**stats,
               "sources": dict(collections.Counter(r["provenance"]["source"] for r in rows)),
               "context_reading_strata": dict(collections.Counter(f"{'with' if r['context_text'] else 'without'}-context/{'short' if len(r['input'])<=16 else 'medium' if len(r['input'])<=32 else 'long'}" for r in rows)),
               "unique_authors": len({r["provenance"].get("author_id") for r in rows}) if role != "regression" else None})
    review = [{"id": row["index"], "source": row["provenance"]["source"], "left_context": row["context_text"],
               "target": row["expected_output"][0], "proposed_reading": row["input"],
               "accepted_answers": row["expected_output"], "original_message": row["provenance"].get("cleaned_message"),
               "decision": "pending", "reviewer": "", "native_japanese_review": False, "notes": ""} for row in rows]
    (directory / "review.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in review), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT / "datasets/ime-standard-ja-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmarks/ime-standard-ja-v1-draft")
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        print("Draft already complete; preserve it or choose a new --output.")
        return
    request = {"seed": args.seed, "sources": json.loads((args.sources / "download-manifest.json").read_text(encoding="utf-8")),
               "quotas": {"development": {"wrime": 500, "jmultiwoz": 250}, "blind": {"wrime": 500, "jmultiwoz": 250}, "regression": {"fineweb_edu_legacy": 500}}}
    if (output / "request.json").exists() and json.loads((output / "request.json").read_text(encoding="utf-8")) != request:
        raise ValueError("Preparation inputs changed; choose a new output.")
    write_json(output / "request.json", request)
    pool = json.loads((output / "candidate-pool.json").read_text(encoding="utf-8")) if (output / "candidate-pool.json").exists() else build_pool(args, output)
    excluded = set(json.loads((output / "overlap-audit.json").read_text(encoding="utf-8"))["excluded_pool_positions"]) if (output / "overlap-audit.json").exists() else overlap_audit(pool, output)
    rng = random.Random(args.seed + 1)
    tracks = select_cases(pool, excluded, rng)
    tracks["regression"] = legacy_cases(rng)
    groups = [{r["provenance"]["group_id"] for r in tracks[role]} for role in ("development", "blind")]
    if groups[0] & groups[1]:
        raise ValueError("Independent split groups overlap.")
    for role, rows in tracks.items():
        write_track(output, role, rows)
    manifest = {"format": FORMAT, "version": "ime-standard-ja-v1-draft", "status": "awaiting_label_review_and_candidate_export",
                "created_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
                "splits": {role: len(rows) for role, rows in tracks.items()}, "labels_formal_gold": False,
                "models_scored": False, "blind_lm_scored": False,
                "primary_tracks": ["wrime", "jmultiwoz"], "legacy_track": "fineweb_edu_legacy",
                "aggregate_policy": "Report each source; independent micro and equal-source macro primary; legacy regression separate, all-source aggregate secondary only",
                "split_policy": "WRIME writer-disjoint; JMultiWOZ user-disjoint and dialogue-disjoint; wizard-disjointness not enforced",
                "review_policy": "Human native reading and acceptable-form review before formal release; no model output/candidate order shown",
                "score_policy": "Frozen actual n-best20; suffix full-vocabulary logP sum; no EOS/truncation; stable ties; full denominator including pool misses/fallbacks",
                "blind_policy": "Require a precommitted evaluation plan and frozen reviewed benchmark; a used blind release cannot be used for subsequent model selection",
                "distribution_policy": "Stratified conversion spans, not an unbiased estimate of all keyboard traffic",
                "source_revisions": {name: data["revision"] for name,data in request["sources"]["sources"].items()},
                "licenses": {"wrime": "CC-BY-NC-ND-4.0; local research draft; derivative redistribution not authorized by this preparation",
                             "jmultiwoz": "CC-BY-SA-4.0", "fineweb_edu_legacy": "Reuse existing per-case provenance and upstream notices"},
                "candidate_generation_uses_model_scores": False, "native_review_completed": False}
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
