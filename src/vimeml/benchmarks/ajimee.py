"""Convert a local AJIMEE JSON array to anco input without changing its text."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FORMAT = "ajimee_anco_input_v1"
DEVELOPMENT_FORMAT = "ime_development_anco_input_v1"
SOURCE_URL = "https://github.com/azooKey/AJIMEE-Bench/tree/main/JWTD_v2/v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def convert_items(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty AJIMEE JSON array.")
    converted, mapping, seen = [], [], set()
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {position}: expected an object.")
        index = row.get("index")
        if type(index) not in (str, int) or (isinstance(index, str) and not index.strip()):
            raise ValueError(f"Row {position}: missing/invalid index.")
        case_id = f"ajimee:{index}"
        if case_id in seen:
            raise ValueError(f"Row {position}: duplicate index {index!r}.")
        seen.add(case_id)
        query, context, answers = row.get("input"), row.get("context_text"), row.get("expected_output")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Row {position}: input must be a nonempty string.")
        if not isinstance(context, str):
            raise ValueError(f"Row {position}: context_text must be a string (empty is allowed).")
        if not isinstance(answers, list) or not answers or any(not isinstance(answer, str) or not answer.strip() for answer in answers):
            raise ValueError(f"Row {position}: expected_output must be a nonempty string array.")
        group = "with-context" if context else "without-context"
        # Preserve kana, width, punctuation, whitespace, answer order and the
        # full input exactly. Neither original_text nor split gold is supplied.
        converted.append({"query": query, "answer": list(answers), "tag": [case_id, group],
                          "left_context": context, "right_context": None})
        mapping.append({"position": position, "id": case_id, "source_index": index,
                        "query": query, "left_context": context, "answers": list(answers), "group": group})
    stats = {"cases": len(rows), "with_context": sum(bool(row["context_text"]) for row in rows),
             "without_context": sum(not row["context_text"] for row in rows),
             "multiple_acceptable_outputs": sum(len(row["expected_output"]) > 1 for row in rows),
             "has_split_inputs": sum(bool(row.get("splitted_input_for_limited_input_length")) for row in rows)}
    return converted, mapping, stats


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def prepare(input_path, output):
    input_path, output = Path(input_path).resolve(), Path(output).resolve()
    raw = input_path.read_bytes()
    converted, mapping, stats = convert_items(json.loads(raw.decode("utf-8-sig")))
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved.get("format") != FORMAT or saved.get("source_sha256") != sha(raw):
            raise ValueError("Output belongs to different input/version. Choose a new --output directory.")
        for name, digest in saved["files_sha256"].items():
            if sha((output / name).read_bytes()) != digest:
                raise ValueError(f"Prepared file changed: {name}. Choose a new --output directory.")
        return saved["stats"]
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty. Choose a new --output directory.")
    notice = (
        "# AJIMEE-Bench provenance\n\n"
        f"Source: {SOURCE_URL}\n\n"
        "Dataset: JWTD_v2/v1, based on Japanese Wikipedia Typo Dataset v2.\n"
        "Upstream dataset license: CC-BY-SA 3.0. Preserve attribution and share-alike terms.\n"
        "License: https://creativecommons.org/licenses/by-sa/3.0/\n\n"
        "evaluation_items.json is the supplied local source, preserved byte for byte.\n"
        "Its exact upstream commit has not been verified; source SHA256 is in manifest.json.\n"
        "ajimee-input.json maps input/context_text/expected_output to query/left_context/answer.\n"
        "No text normalization, splitting, filtering or answer injection into context.\n"
        "original_text remains in the source snapshot and is excluded from anco input.\n"
        "For evaluation only; do not add these files to the training corpus.\n"
    ).encode("utf-8")
    files = {"evaluation_items.json": raw, "ajimee-input.json": json_bytes(converted),
             "case-map.json": json_bytes(mapping), "NOTICE.md": notice}
    output.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = output / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    manifest = {"format": FORMAT, "status": "complete", "source_path": str(input_path),
                "source_url": SOURCE_URL, "source_sha256": sha(raw), "upstream_commit_verified": False,
                "license": "CC-BY-SA-3.0", "stats": stats,
                "files_sha256": {name: sha(content) for name, content in files.items()},
                "policy": "Unmodified full queries and all references; supplied left context only; right context null. Mapping positions are zero-based."}
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_bytes(json_bytes(manifest))
    temporary.replace(manifest_path)
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Downloaded evaluation_items.json path.")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmarks/ajimee-jwtd-v2-v1",
                        help="Directory for anco input, source snapshot, case mapping and manifest.")
    args = parser.parse_args(argv)
    try:
        stats = prepare(args.input, args.output)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Anco input: {args.output.resolve() / 'ajimee-input.json'}")
    print(f"Case map: {args.output.resolve() / 'case-map.json'}")


if __name__ == "__main__":
    main()
