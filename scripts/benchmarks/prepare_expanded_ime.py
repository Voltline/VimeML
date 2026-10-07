"""Prepare model-independent held-out text and kana for a Mac candidate export."""

import argparse
import collections
import datetime
import json
import random
import re
import sys
import bisect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.ajimee import convert_items
from vimeml.benchmarks.readings import DualReading
from vimeml.training.data import write_json

TEXT = re.compile(r"[ぁ-ゖァ-ヺー一-龠々、。！？]+")
UNSAFE = re.compile(
    r"明日|昨日|一日|二日|何日|何人|生物|人気|市場|上手|下手|大人気|一人|二人|富士山|日本橋|血液|様奥|ひきこさん|接続し|方|型|行っ|行う|行い|痩身|ヤキモキ|お断り|減衰有機|仕事能力に活躍|介護のついて|、、、|^当は|しまかもしれ|髪はたまた"
)
VARIANTS = [
    ("下さい", "ください"),
    ("出来", "でき"),
    ("良い", "よい"),
    ("良く", "よく"),
    ("ご飯", "ごはん"),
    ("眼鏡", "めがね", "メガネ"),
    ("子供", "子ども"),
    ("頑張", "がんば"),
]


def references(text):
    forms = {text}
    for group in VARIANTS:
        for form in list(forms):
            for current in group:
                if current in form:
                    forms.update(form.replace(current, alternative) for alternative in group)
    return [text] + sorted(forms - {text})[:15]


def bucket(context, query):
    ctx = (
        "none"
        if not context
        else ("short" if len(context) <= 16 else "medium" if len(context) <= 32 else "long")
    )
    length = "short" if len(query) <= 16 else "medium" if len(query) <= 32 else "long"
    return ctx + "-" + length


def source_urls(pool):
    """Read only source id/URL columns for the reservoir's selected rows."""
    import pyarrow.parquet as pq

    by_file = collections.defaultdict(list)
    for row in pool:
        if row["source"] == "tatoeba":
            row["source_url"] = (
                "https://tatoeba.org/en/sentences/show/" + row["doc_id"].split(":", 1)[1]
            )
        else:
            by_file[row["source_file"]].append(row)
    for filename, rows in by_file.items():
        by_index = collections.defaultdict(list)
        for row in rows:
            by_index[row["row_index"]].append(row)
        indices = sorted(by_index)
        offset = 0
        with pq.ParquetFile(ROOT / filename) as parquet:
            for batch in parquet.iter_batches(columns=["id", "url"], batch_size=8192):
                lo = bisect.bisect_left(indices, offset)
                hi = bisect.bisect_left(indices, offset + len(batch))
                for idx in indices[lo:hi]:
                    local = idx - offset
                    ident = "fineweb:" + batch.column(0)[local].as_py()
                    for row in by_index[idx]:
                        assert row["doc_id"] == ident
                        row["source_url"] = batch.column(1)[local].as_py()
                offset += len(batch)
        assert all(r.get("source_url") for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "outputs/corpus-fast-v1")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/benchmarks/ime-expanded-v21"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior version; use a fresh output.")
    reader = DualReading()
    rng = random.Random(20261007)
    excluded = set()
    for old in ("ajimee-jwtd-v2-v1", "ime-dev-v2"):
        for r in json.loads(
            (ROOT / "artifacts/benchmarks" / old / "evaluation_items.json").read_text(
                encoding="utf-8"
            )
        ):
            excluded.add((r["context_text"], r["input"]))
    used_groups = set()
    used_sentences = set()
    used_queries = set()
    used_urls = set()
    splits = {}
    rejection = collections.Counter()
    for role, source_split, count in [("development", "validation", 2000), ("blind", "test", 1000)]:
        # A bounded reservoir over structurally eligible corpus rows keeps
        # sampling independent of local model scores and candidate coverage.
        pool = []
        seen = 0
        with (args.corpus / (source_split + ".jsonl")).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                text = row["text"]
                if (
                    not 10 <= len(text) <= 55
                    or not TEXT.fullmatch(text)
                    or not text.endswith(("。", "！", "？"))
                ):
                    continue
                if text.count("。") > 1 or not re.search("[一-龠]", text) or UNSAFE.search(text):
                    continue
                seen += 1
                row["corpus_line"] = line_number
                if len(pool) < 25000:
                    pool.append(row)
                else:
                    position = rng.randrange(seen)
                    if position < len(pool):
                        pool[position] = row
        rng.shuffle(pool)
        source_urls(pool)
        candidates = collections.defaultdict(list)
        for index, row in enumerate(pool):
            text = row["text"]
            group = row["group_id"]
            if group in used_groups or text in used_sentences or row["source_url"] in used_urls:
                rejection["group_or_sentence_reuse"] += 1
                continue
            # Choose a word-boundary suffix, or a short complete sentence.
            words = reader.mecab(text)
            if not words or words[0].feature.pos1 in ("助詞", "助動詞", "接尾辞", "補助記号"):
                rejection["sentence_fragment"] += 1
                continue
            whole_check = reader.analyse(text)
            if (
                not whole_check["agreement"]
                or whole_check["unknown"]
                or not whole_check["kana_only"]
            ):
                rejection["whole_sentence_reading"] += 1
                continue
            sudachi_starts = {word.begin() for word in reader.sudachi.tokenize(text, reader.mode)}
            boundaries = []
            offset = 0
            previous = None
            for word in words:
                compound_boundary = previous is not None and (
                    previous.feature.pos1 == "接頭辞"
                    or previous.feature.pos1 == "名詞"
                    and word.feature.pos1 == "名詞"
                    or previous.feature.pos1 in ("名詞", "動詞")
                    and word.feature.pos1 == "動詞"
                    or re.fullmatch("[ァ-ヺー]+", previous.surface + word.surface)
                )
                if (
                    not compound_boundary
                    and offset in sudachi_starts
                    and word.feature.pos1 not in ("助詞", "助動詞", "接尾辞", "補助記号")
                    and 5 <= offset <= 64
                    and 8 <= len(text) - offset <= 40
                ):
                    boundaries.append(offset)
                offset += len(word.surface)
                previous = word
            with_context = index % 2 == 0 or len(text) > 40
            if with_context:
                if not boundaries:
                    rejection["no_contiguous_suffix_boundary"] += 1
                    continue
                start = rng.choice(boundaries)
            else:
                start = 0
            context, target = text[:start], text[start:]
            check = reader.analyse(target)
            contextual = reader.analyse(text, start)
            if not check["agreement"] or check["unknown"] or not check["kana_only"]:
                rejection["reading_disagreement_or_unknown"] += 1
                continue
            if (
                contextual["unknown"]
                or contextual["unidic"] != check["unidic"]
                or contextual["sudachi"] != check["sudachi"]
            ):
                rejection["contextual_reading_or_boundary"] += 1
                continue
            query = check["unidic"]
            if not 8 <= len(query) <= 80 or not re.search("[一-龠]", target):
                rejection["query_length_or_no_conversion"] += 1
                continue
            if (context, query) in excluded or query in used_queries:
                rejection["old_case_or_duplicate_reading"] += 1
                continue
            answers = []
            for form in references(target):
                alt = reader.analyse(form)
                if alt["unidic"] == query and alt["sudachi"] == query and not alt["unknown"]:
                    answers.append(form)
            if target not in answers:
                rejection["reference_reading_mismatch"] += 1
                continue
            case = {
                "index": "pending",
                "context_text": context,
                "input": query,
                "expected_output": answers,
                "reason_zh": "留出原句的连续词边界转换区间；两套词典的正字法假名读音一致。",
                "category": "natural_heldout_text",
                "provenance": {
                    "kind": "corpus_heldout_suffix",
                    "role": role,
                    "corpus_split": source_split,
                    "corpus_line": row["corpus_line"],
                    "source": row["source"],
                    "source_doc_id": row["doc_id"],
                    "group_id": group,
                    "source_file": row["source_file"],
                    "source_row_index": row["row_index"],
                    "source_url": row["source_url"],
                    "source_text_hash": row["text_hash"],
                    "original_sentence": text,
                    "span_start": start,
                    "span_end": len(text),
                },
                "review": {
                    "status": "two_dictionary_reading_checked",
                    "human_native_review": False,
                    "all_acceptable_forms_exhaustive": False,
                    "not_formal_gold": True,
                },
            }
            candidates[bucket(context, query)].append(case)
            used_groups.add(group)
            used_sentences.add(text)
            used_queries.add(query)
            used_urls.add(row["source_url"])
            if sum(len(v) for v in candidates.values()) >= count * 4:
                break
        # Even round-robin over available context/reading strata; no LM access.
        for values in candidates.values():
            rng.shuffle(values)
        chosen = []
        buckets = sorted(candidates)
        while len(chosen) < count and any(candidates.values()):
            for key in buckets:
                if candidates[key]:
                    chosen.append(candidates[key].pop())
                    if len(chosen) == count:
                        break
        if len(chosen) != count:
            raise ValueError(f"{role}: only {len(chosen)} eligible cases; no draft padding.")
        for i, case in enumerate(chosen, 1):
            case["index"] = f"ime-v21-{role}-{i:04}"
        splits[role] = chosen
        print(
            f"{role}: selected {len(chosen)} unique groups; structurally eligible={seen}",
            flush=True,
        )
    development, blind = splits["development"], splits["blind"]
    assert not {r["provenance"]["group_id"] for r in development} & {
        r["provenance"]["group_id"] for r in blind
    }
    all_rows = development + blind
    assert len({r["input"] for r in all_rows}) == 3000
    assert len({r["provenance"]["source_url"] for r in all_rows}) == 3000
    args.output.mkdir(parents=True)
    for role, rows in splits.items():
        folder = args.output / role
        folder.mkdir()
        converted, mapping, stats = convert_items(rows)
        write_json(folder / "evaluation_items.json", rows)
        write_json(folder / "ajimee-input.json", converted)
        write_json(folder / "case-map.json", mapping)
        write_json(
            folder / "stats.json",
            {
                **stats,
                "context_reading_strata": dict(
                    collections.Counter(bucket(r["context_text"], r["input"]) for r in rows)
                ),
                "sources": dict(collections.Counter(r["provenance"]["source"] for r in rows)),
            },
        )
    converted, mapping, stats = convert_items(all_rows)
    write_json(args.output / "evaluation_items.json", all_rows)
    write_json(args.output / "ajimee-input.json", converted)
    write_json(args.output / "case-map.json", mapping)
    write_json(
        args.output / "manifest.json",
        {
            "format": "ime_expanded_candidate_export_input_v1",
            "status": "prepared_for_candidate_export",
            "stats": stats,
            "splits": {"development": 2000, "blind": 1000},
            "seed": 20261007,
            "dictionary_versions": reader.versions,
            "source_corpus": str(args.corpus.resolve()),
            "source_manifest_policy": "Reuse frozen corpus split and provenance; no corpus/checkpoint SHA256 rescan.",
            "test_access": "Corpus test text read only for blind input preparation; no LM scores or test BPC evaluated.",
            "model_or_candidate_scores_used": False,
            "candidate_export_complete": False,
            "labels_formal_gold": False,
            "review_scope": "Two dictionary orthographic-reading agreement plus structural filtering; not a full native/manual label audit.",
            "known_limitations": [
                "Dictionary consensus can still choose a wrong reading",
                "Reference alternatives are incomplete",
                "Natural web corpus extraction and near-duplicate limits remain",
                "Not a curated homophone/category-balanced final benchmark",
                "Candidate-count/recall strata unavailable until export",
            ],
            "deduplication": "Unique source groups, source URLs, original sentences and full readings across all 3000; exact old query/context excluded. Near-duplicate sources are not fully audited.",
            "prepared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "rejections": dict(rejection),
        },
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
