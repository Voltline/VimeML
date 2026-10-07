"""Import the real Mac export for the prepared 2000/1000 IME inputs."""

import argparse
import collections
import datetime
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.ajimee import convert_items
from vimeml.benchmarks.expanded_ime import FORMAT, load_expanded_export
from vimeml.benchmarks.evaluate_ajimee import metrics
from vimeml.training.data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument(
        "--prepared", type=Path, default=ROOT / "artifacts/benchmarks/ime-expanded-v21-final"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/benchmarks/ime-expanded-v21-candidates-v1"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior imported versions; choose a fresh output.")
    files = [
        "ajimee-input.json",
        "case-map.json",
        "azookey-candidates.json",
        "converter-version.txt",
        "dictionary-versions.txt",
        "swift-version.txt",
        "export-flags.txt",
        "export.log",
        "completed.txt",
    ]
    # Read only named JSON/metadata files, never execute archive contents or
    # extract arbitrary paths supplied by the attachment.
    with zipfile.ZipFile(args.archive) as package:
        names = package.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member names.")
        expected = {
            role: {name: package.read("azookey-results/" + role + "/" + name) for name in files}
            for role in ("development", "blind")
        }
    args.output.mkdir(parents=True)
    shutil.copyfile(args.archive, args.output / "source-export.zip")
    shutil.copyfile(args.prepared / "manifest.json", args.output / "preparation-manifest.json")
    notice = ROOT / "handoff/ime-v21-3000-20261007/NOTICE.md"
    if notice.exists():
        shutil.copyfile(notice, args.output / "NOTICE.md")
    summary = {}
    for role, count in [("development", 2000), ("blind", 1000)]:
        source = args.prepared / role
        dest = args.output / role
        dest.mkdir()
        exported = dest / "ajimee-results"
        exported.mkdir()
        original = json.loads((source / "evaluation_items.json").read_text(encoding="utf-8"))
        inputs, mapping, stats = convert_items(original)
        if len(original) != count:
            raise ValueError(f"{role}: expected {count} prepared cases, got {len(original)}.")
        for name, value in [("ajimee-input.json", inputs), ("case-map.json", mapping)]:
            if json.loads(expected[role][name]) != value:
                raise ValueError(f"{role}: returned {name} differs from handoff.")
        for name in ("evaluation_items.json", "ajimee-input.json", "case-map.json"):
            shutil.copyfile(source / name, dest / name)
        for name, content in expected[role].items():
            (exported / name).write_bytes(content)
        manifest = {
            "format": FORMAT,
            "status": "complete",
            "split": role,
            "stats": stats,
            "source_label_version": "ime-expanded-v21-final",
            "labels_formal_gold": False,
            "label_status": "provisional_two_dictionary_checked",
            "source_corpus_split": "validation" if role == "development" else "test",
            "role_policy": "Development diagnostics allowed; blind LM scoring only after model selection.",
            "models_scored": False,
            "imported_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        write_json(dest / "manifest.json", manifest)
        _, provenance, rows = load_expanded_export(dest, manifest)
        source_by_id = {f"ajimee:{r['index']}": r for r in original}
        missing = [
            {**r, "source": source_by_id[r["id"]]["provenance"]}
            for r in rows
            if not set(r["answers"]).intersection(r["orders"]["azookey"])
        ]
        write_json(dest / "candidate-missing-exact.json", missing)
        summary[role] = {
            "stats": stats,
            "azookey": metrics(rows, "azookey"),
            "provenance": provenance,
            "candidate_counts": dict(collections.Counter(len(r["candidates"]) for r in rows)),
            "empty_candidates": sum(not r["candidates"] for r in rows),
            "label_status": "provisional; recall is exact-reference coverage",
        }
    write_json(
        args.output / "import-report.json",
        {
            "status": "complete",
            "splits": summary,
            "archive_size": args.archive.stat().st_size,
            "models_scored": False,
            "validation": "All 3000 cases aligned to local handoff; frozen versions/flags; actual candidate counts and CLI rank aggregates.",
            "blind_lm_scored": False,
            "labels_formal_gold": False,
        },
    )
    print(
        json.dumps(
            {role: summary[role]["azookey"] for role in summary}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
