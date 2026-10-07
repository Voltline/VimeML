"""Read an imported expanded IME export without changing references or pools."""

import collections
import json
import math
from pathlib import Path

from vimeml.benchmarks.ajimee import convert_items

FORMAT = "ime_expanded_azookey_export_v1"
CONVERTER = "d59a28e4c7ca049aef04f29a91eae9677a7753f2"
DICTIONARIES = {
    "Sources/KanaKanjiConverterModuleWithDefaultDictionary/azooKey_dictionary_storage": "4d418525b090cf49c219819d05a7e3cc2a4346eb",
    "Sources/KanaKanjiConverterModuleWithDefaultDictionary/azooKey_emoji_dictionary_storage": "67b822603391b01238d7b80b8b61b63f966cf357",
}
FLAGS = ["n_best=20", "typo_mode=off", "Zenzai=disabled (no model)", "stable=off"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_expanded_export(directory, manifest=None):
    directory = Path(directory)
    manifest = manifest or read(directory / "manifest.json")
    if manifest.get("format") != FORMAT or manifest.get("status") != "complete":
        raise ValueError("Expected a complete imported expanded export.")
    inputs, mapping, stats = convert_items(read(directory / "evaluation_items.json"))
    if (
        inputs != read(directory / "ajimee-input.json")
        or mapping != read(directory / "case-map.json")
        or stats != manifest["stats"]
    ):
        raise ValueError("Expanded source/input/mapping mismatch.")
    exported = directory / "ajimee-results"
    if inputs != read(exported / "ajimee-input.json") or mapping != read(
        exported / "case-map.json"
    ):
        raise ValueError("Returned Mac input/mapping differs from prepared source.")
    provenance = {
        name: (exported / name).read_text(encoding="utf-8-sig")
        for name in (
            "converter-version.txt",
            "dictionary-versions.txt",
            "swift-version.txt",
            "export-flags.txt",
            "completed.txt",
        )
    }
    if provenance["converter-version.txt"].strip() != CONVERTER:
        raise ValueError("Unexpected converter revision.")
    actual = {}
    for line in provenance["dictionary-versions.txt"].splitlines():
        if not line.startswith(" "):
            raise ValueError("Dictionary differs from pinned revision or is uninitialized.")
        revision, path, *_ = line.split()
        actual[path] = revision
    if (
        actual != DICTIONARIES
        or provenance["export-flags.txt"].splitlines() != FLAGS
        or provenance["completed.txt"].strip() != "cli_exit=0"
    ):
        raise ValueError("Dictionary versions, export flags or completion marker mismatch.")
    raw = read(exported / "azookey-candidates.json")
    items = raw.get("items")
    if raw.get("n_best") != 20 or not isinstance(items, list) or len(items) != len(inputs):
        raise ValueError("Candidate n-best/count mismatch.")
    rows = []
    ranks = collections.Counter()
    for position, (expected, item, case) in enumerate(zip(inputs, items, mapping)):
        if any(
            item.get(key) != expected[value]
            for key, value in (
                ("query", "query"),
                ("answers", "answer"),
                ("left_context", "left_context"),
            )
        ):
            raise ValueError(f"Row {position}: exported case mismatch.")
        if item.get("right_context") not in ("", None):
            raise ValueError(f"Row {position}: unexpected right context.")
        candidates = item.get("outputs")
        if not isinstance(candidates, list) or len(candidates) > 20:
            raise ValueError(f"Row {position}: invalid candidate count.")
        texts = []
        for candidate in candidates:
            text, score = candidate.get("text"), candidate.get("score")
            if (
                not isinstance(text, str)
                or not text
                or type(score) not in (int, float)
                or not math.isfinite(score)
            ):
                raise ValueError(f"Row {position}: invalid candidate.")
            texts.append(text)
        if len(texts) != len(set(texts)):
            raise ValueError(f"Row {position}: duplicate candidates.")
        rank = next((i for i, text in enumerate(texts) if text in case["answers"]), -1)
        if item.get("max_rank") != rank:
            raise ValueError(f"Row {position}: CLI reference rank mismatch.")
        ranks[str(rank)] += 1
        rows.append(
            {**case, "candidates": candidates, "orders": {"azookey": texts}, "fallbacks": {}}
        )
    if raw.get("stat", {}).get("query_count") != len(rows) or raw.get("stat", {}).get(
        "ranks"
    ) != dict(ranks):
        raise ValueError("CLI aggregate statistics differ from actual rows.")
    provenance.update(
        {
            "n_best": 20,
            "cli_execution_seconds": raw.get("execution_time"),
            "validation": "Exact decoded JSON identity, versions, flags, candidate shape/rank; no repeated SHA256.",
            "flag_provenance": "Recorded exporter arguments; CLI JSON does not independently embed every flag.",
        }
    )
    return manifest, provenance, rows
