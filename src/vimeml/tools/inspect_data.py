"""Inspect local raw datasets without cleaning or modifying them."""

import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
DATASETS = ROOT / "datasets"
OUTPUT = ROOT / "outputs" / "inspection"
SEED = 42


def inspect_fineweb():
    files = sorted(
        (DATASETS / "fineweb-2-edu-japanese").glob("*.parquet")
    )
    if not files:
        raise FileNotFoundError("No FineWeb2 parquet files found.")

    rng = random.Random(SEED)
    reports = []

    for path in files:
        with pq.ParquetFile(path) as parquet:
            names = parquet.schema_arrow.names
            if "text" not in names:
                raise ValueError(f"{path.name}: missing text column")

            wanted = [
                "text", "id", "url", "language", "language_score",
                "score", "is_cleaned", "token_count",
            ]
            columns = [name for name in wanted if name in names]
            samples = []
            sampled_group = None

            if parquet.metadata.num_row_groups:
                sampled_group = rng.randrange(
                    parquet.metadata.num_row_groups
                )
                # Read one row group, rather than the entire shard.
                table = parquet.read_row_group(
                    sampled_group, columns=columns
                )
                indices = rng.sample(
                    range(table.num_rows), min(3, table.num_rows)
                )
                samples = table.take(indices).to_pylist()

            reports.append({
                "file": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "documents": parquet.metadata.num_rows,
                "row_groups": parquet.metadata.num_row_groups,
                "schema": str(parquet.schema_arrow),
                "sampled_row_group": sampled_group,
                "samples": samples,
            })

    return reports


def inspect_tatoeba():
    path = DATASETS / "Tatoeba" / "jpn_sentences.tsv"
    rng = random.Random(SEED)
    languages = Counter()
    samples = []
    rows = valid_rows = malformed_rows = characters = 0

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            rows += 1
            parts = line.rstrip("\r\n").split("\t", 2)
            if len(parts) != 3:
                malformed_rows += 1
                continue

            sentence_id, language, text = parts
            valid_rows += 1
            languages[language] += 1
            characters += len(text)
            record = {
                "line": rows,
                "id": sentence_id,
                "language": language,
                "text": text,
            }

            # Reservoir sampling: a uniform sample without loading all rows.
            if len(samples) < 10:
                samples.append(record)
            else:
                index = rng.randrange(valid_rows)
                if index < 10:
                    samples[index] = record

    return {
        "file": path.relative_to(ROOT).as_posix(),
        "rows": rows,
        "malformed_rows": malformed_rows,
        "languages": dict(languages),
        "raw_text_characters": characters,
        "samples": samples,
    }


def main():
    fineweb = inspect_fineweb()
    tatoeba = inspect_tatoeba()
    summary = {
        "fineweb_shards": len(fineweb),
        "fineweb_documents": sum(item["documents"] for item in fineweb),
        "fineweb_bytes": sum(item["bytes"] for item in fineweb),
        "tatoeba_rows": tatoeba["rows"],
        "tatoeba_malformed_rows": tatoeba["malformed_rows"],
        "tatoeba_raw_text_characters": tatoeba["raw_text_characters"],
    }
    report = {
        "seed": SEED,
        "summary": summary,
        "fineweb": fineweb,
        "tatoeba": tatoeba,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "report.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {destination}")


if __name__ == "__main__":
    main()