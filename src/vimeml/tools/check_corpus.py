"""Check a completed corpus against its saved integrity report, without merge intermediates."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.data.build import file_sha
from vimeml.data.corpus_parts import text_sha


def check(corpus, full=False):
    corpus = Path(corpus).resolve()
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats = json.loads((corpus / "stats.json").read_text(encoding="utf-8"))
    report = json.loads((corpus / "integrity-check.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or report.get("status") != "passed":
        raise ValueError("Corpus has not completed its export integrity check")
    if file_sha(manifest_path) != report["manifest_sha256"]:
        raise ValueError("Manifest differs from the verified version")
    for relative, expected in manifest["code_sha256"].items():
        if text_sha(ROOT / relative) != expected:
            raise ValueError(f"Recorded corpus core changed: {relative}")
    for relative, expected in manifest.get("merge_code_sha256", {}).items():
        if text_sha(ROOT / relative) != expected:
            raise ValueError(f"Recorded merge implementation changed: {relative}")
    for field, key in (("sentences", "unique_sentences"), ("characters", "final_characters"),
                       ("document_origins", "document_origins")):
        if report[field] != stats[key]:
            raise ValueError(f"Statistics differ from the integrity report: {key}")
    for name, recorded in report["exports"].items():
        if (corpus / name).stat().st_size != recorded["bytes"]:
            raise ValueError(f"Export size differs: {name}")
    if full:
        # Explicit opt-in: full verification reads tens of GB.
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(file_sha, corpus / name): (name, data["sha256"])
                       for name, data in report["exports"].items()}
            for future in as_completed(futures):
                name, expected = futures[future]
                if future.result() != expected:
                    raise ValueError(f"Export SHA256 differs: {name}")
                print(f"SHA256 passed: {name}", flush=True)
    result = {"status": "passed", "mode": "full_export_hashes" if full else "metadata_and_sizes",
              "sentences": stats["unique_sentences"], "characters": stats["final_characters"],
              "core_fingerprints_unchanged": True,
              "ready_for_lm_training": manifest.get("ready_for_lm_training", False)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "outputs/corpus-fast-v1")
    parser.add_argument("--full", action="store_true", help="Recompute recorded export SHA256; reads the full corpus")
    args = parser.parse_args()
    check(args.corpus, args.full)


if __name__ == "__main__":
    main()
