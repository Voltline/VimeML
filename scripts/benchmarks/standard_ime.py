"""Review, freeze, package and import the reusable standard IME benchmark."""
import argparse
import collections
import copy
import json
import re
import shutil
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.benchmarks.ajimee import convert_items
from vimeml.benchmarks.expanded_ime import CONVERTER
from vimeml.benchmarks.standard_ime import FORMAT, read, load_standard_export, reviewed_and_frozen
from vimeml.training.data import write_json

DRAFT = ROOT / "artifacts/benchmarks/ime-standard-ja-v1-draft"
PREPARED_ROLES = ("development", "blind", "regression")


def fresh(path):
    if path.exists():
        raise ValueError(f"Preserve existing output: {path}; choose a fresh path.")
    path.mkdir(parents=True)


def track_files(path, items):
    path.mkdir(parents=True, exist_ok=True)
    cli, mapping, stats = convert_items(items)
    for name, value in (("evaluation_items.json", items), ("ajimee-input.json", cli), ("case-map.json", mapping), ("stats.json", stats)):
        write_json(path / name, value)


def notice(output):
    text = """# Source attribution and use scope

Local research benchmark. WRIME conversion spans and readings are derived
from WRIME v2: https://github.com/ids-cv/wrime (CC BY-NC-ND 4.0).
This preparation does not grant permission to redistribute an adapted WRIME
benchmark. Restricted data and review packages remain outside Git. Public code
and methodology may be shared separately; data publication requires permission.

JMultiWOZ: https://github.com/nu-dialogue/jmultiwoz (CC BY-SA 4.0).
Authors: Atsumoto Ohashi, Ryu Hirai, Shinya Iizuka, Ryuichiro Higashinaka (2024).
WRIME: cite Suzuki et al., LREC 2022 and Kajiwara et al., NAACL 2021.
Per-case IDs/rows and pinned revisions are preserved in provenance/manifest.

Legacy regression: previously exposed FineWeb2 Edu Japanese development cases,
not independent or blind. Existing per-case web URLs/provenance remain intact.
FineWeb2 packaging: ODC-By 1.0 and upstream Common Crawl terms; underlying text
retains its original rights. Original frozen references are unchanged.
Source: https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese

Candidate collection performs no LM scoring. Human review pages show no model
outputs or candidate ordering. AI expert review and dictionary checks are
separately recorded from human native gold. No credentials or trained weights are bundled.
"""
    (output / "NOTICE.md").write_text(text, encoding="utf-8")
    for source in ("wrime", "jmultiwoz"):
        shutil.copyfile(ROOT / f"datasets/ime-standard-ja-v1/{source}/LICENSE", output / f"LICENSE-{source}.txt")


def make_review(args):
    if args.refresh:
        if read(args.output / "manifest.json")["version"] != read(args.prepared / "manifest.json")["version"]:
            raise ValueError("Review directory belongs to a different draft.")
        for role in ("development", "blind"):
            existing = [json.loads(line) for line in (args.output / f"{role}-review.jsonl").read_text(encoding="utf-8").splitlines()]
            if any(r["decision"] != "pending" for r in existing):
                raise ValueError("Preserve edited reviews; choose a new output.")
    else:
        fresh(args.output)
    notice(args.output)
    manifest = read(args.prepared / "manifest.json")
    write_json(args.output / "manifest.json", manifest)
    template = (ROOT / "templates/benchmarks/standard-ime-review.html").read_text(encoding="utf-8")
    from vimeml.benchmarks.readings import DualReading
    reader = DualReading()
    pairs = [("分か", "わか"), ("可愛い", "かわいい"), ("可愛く", "かわいく"),
             ("下さい", "ください"), ("出来", "でき"), ("良い", "よい"), ("良く", "よく"),
             ("ご飯", "ごはん"), ("眼鏡", "めがね", "メガネ"), ("子供", "子ども"),
             ("頑張", "がんば"), ("事", "こと"), ("為", "ため"), ("無い", "ない"),
             ("無く", "なく"), ("買い取り", "買取", "買取り"), ("かもしれ", "かも知れ"),
             ("我々", "われわれ"), ("言え", "いえ"), ("有り", "あり"), ("無理", "むり")]
    for role in ("development", "blind"):
        reviews = [json.loads(line) for line in (args.prepared / role / "review.jsonl").read_text(encoding="utf-8").splitlines()]
        for row in reviews:
            forms = {row["target"]}
            for group in pairs:
                for form in sorted(forms)[:16]:
                    for current in group:
                        if current in form:
                            forms.update(form.replace(current, alt) for alt in group)
                forms = set(sorted(forms)[:32]) | {row["target"]}
            suggestions = []
            for form in sorted(forms - {row["target"]})[:32]:
                check = reader.analyse(form)
                if (not check["unknown"] and check["unidic"] == row["proposed_reading"] and
                        check["sudachi"] == row["proposed_reading"]):
                    suggestions.append(form)
            row["spelling_suggestions"] = suggestions[:15]
            row["suggestion_policy"] = "Pre-score orthographic proposals, dictionary-matched; not accepted answers until reviewer confirms"
        blob = json.dumps({"version": manifest["version"], "role": role, "cases": reviews}, ensure_ascii=False).replace("<", "\\u003c")
        (args.output / f"{role}-review.html").write_text(template.replace("__PAYLOAD__", blob), encoding="utf-8")
        (args.output / f"{role}-review.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in reviews), encoding="utf-8")
    (args.output / "README.md").write_text(
        "# 标签审核\n\n打开development-review.html或blind-review.html即可离线审核。"
        "核对给定左文下的读音、转换边界和全部合理表记；信息不足则标为exclude并写原因。"
        "不要根据任何模型结果改答案。审核者和母语审核声明必须如实填写，AI初审保留false。"
        "下载的两个JSONL交给freeze入口；未审核项目不会进入正式发布。"
        "原文存在右侧文字时，只用于追溯，不把未提供给模型的右文作为唯一消歧依据。\n",
        encoding="utf-8")
    with zipfile.ZipFile(args.output.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as package:
        for path in sorted(args.output.iterdir()):
            package.write(path, args.output.name + "/" + path.name)


def freeze(args):
    if args.output.exists():
        raise ValueError("Preserve prior releases.")
    manifest = read(args.prepared / "manifest.json")
    if reviewed_and_frozen(manifest) and (args.version == manifest["version"] or not args.change_note.strip()):
        raise ValueError("Corrections to a reviewed release require a new version and --change-note.")
    tracks, quarantine, decisions = {}, [], {}
    for role in ("development", "blind"):
        original = read(args.prepared / role / "evaluation_items.json")
        reviews = [json.loads(line) for line in (args.reviews / f"{role}-review.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {r["id"]: r for r in reviews}
        if len(by_id) != len(reviews) or set(by_id) != {r["index"] for r in original}:
            raise ValueError("Review IDs do not match the prepared draft.")
        selected = []
        for row in original:
            review = by_id[row["index"]]
            if (review["left_context"] != row["context_text"] or
                    review["target"] != row["expected_output"][0] or review["source"] != row["provenance"]["source"]):
                raise ValueError("Review context/source changed; prepare a new version.")
            if review["decision"] == "exclude" and review["notes"].strip() and review["reviewer"].strip():
                quarantine.append({"case": row, "review": review})
                continue
            if review["decision"] != "accept" or not review["reviewer"].strip() or not review["notes"].strip():
                raise ValueError(f"Pending review not complete: {row['index']}")
            native = args.review_mode == "human-native"
            if native and review["native_japanese_review"] is not True:
                raise ValueError(f"Native review not complete: {row['index']}")
            if not native and (review.get("review_type") != "ai_expert" or review["native_japanese_review"] is not False):
                raise ValueError("AI expert review must be identified explicitly, without a native-human claim.")
            answers = review["accepted_answers"]
            if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers) or len(set(answers)) != len(answers):
                raise ValueError("Accepted forms must be a nonempty unique string list.")
            if not isinstance(review["proposed_reading"], str) or not re.fullmatch(r"[ァ-ヺー。、！？「」『』（）・：，,.!?]{4,64}", review["proposed_reading"]):
                raise ValueError("Reviewed input must follow the declared 4-64-character orthographic katakana protocol.")
            frozen = copy.deepcopy(row)
            frozen["input"] = review["proposed_reading"]
            frozen["expected_output"] = answers
            frozen["review"] = {"status": "native_reviewed" if native else "ai_expert_reviewed",
                                "human_native_review": native, "not_formal_gold": not native,
                                "reviewer": review["reviewer"], "notes": review["notes"], "acceptable_forms_exhaustive": False}
            selected.append(frozen)
        if {r["provenance"]["source"] for r in selected} != {"wrime", "jmultiwoz"}:
            raise ValueError("Both sources must remain in each external split.")
        tracks[role] = selected
        decisions[role] = reviews
    tracks["regression"] = read(args.prepared / "regression/evaluation_items.json")
    # Edited readings must not create exact input/context leakage across splits.
    dev_keys = {(r['context_text'], r['input']) for r in tracks['development']}
    if dev_keys & {(r['context_text'], r['input']) for r in tracks['blind']}:
        raise ValueError("Reviewed development/blind input/context pairs overlap.")
    fresh(args.output)
    for role, rows in tracks.items():
        track_files(args.output / role, rows)
    manifest.update(parent_release_id=manifest.get("release_id"), change_note=args.change_note,
                    frozen_utc=datetime.now(timezone.utc).isoformat(),
                    version=args.version, release_id=uuid.uuid4().hex, status="labels_frozen_awaiting_candidates",
                    labels_formal_gold=args.review_mode == "human-native",
                    native_review_completed=args.review_mode == "human-native",
                    labels_frozen=True, review_completed=True,
                    label_quality="human_native_reviewed" if args.review_mode == "human-native" else "ai_expert_reviewed",
                    review_policy="Frozen pre-score AI expert references; not human native gold" if args.review_mode == "ai-expert" else "Frozen native-human-reviewed references",
                    legacy_labels_formal_gold=False, splits={r: len(v) for r,v in tracks.items()},
                    label_change_policy="Immutable published release; corrections require a new release ID and an explicit changelog")
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "review-decisions.json", decisions)
    write_json(args.output / "quarantine.json", quarantine)
    notice(args.output)


def package(args):
    manifest = read(args.prepared / "manifest.json")
    fresh(args.output)
    for role in ("development", "blind"):
        dest = args.output / role
        dest.mkdir()
        for filename in ("evaluation_items.json", "ajimee-input.json", "case-map.json", "stats.json"):
            shutil.copyfile(args.prepared / role / filename, dest / filename)
    write_json(args.output / "manifest.json", manifest)
    notice(args.output)
    shutil.copyfile(ROOT / "scripts/benchmarks/export_azookey_v21.sh", args.output / "export_azookey.sh")
    shutil.copyfile(ROOT / "scripts/benchmarks/run_standard_ime_mac.sh", args.output / "run_mac.sh")
    handoff_note = (ROOT / "templates/benchmarks/standard-ime-mac.md").read_text(encoding="utf-8")
    for key, value in {"VERSION": manifest["version"], "DEVELOPMENT": manifest["splits"]["development"],
                       "BLIND": manifest["splits"]["blind"], "TOTAL": manifest["splits"]["development"] + manifest["splits"]["blind"],
                       "LABEL_QUALITY": manifest.get("label_quality", "unreviewed_draft"),
                       "FORMAL_GOLD": manifest["labels_formal_gold"]}.items():
        handoff_note = handoff_note.replace(f"__{key}__", str(value))
    (args.output / "README_CN.md").write_text(handoff_note, encoding="utf-8")
    (args.output / "README.md").write_text(
        f"# Standard IME candidate collection\n\n{manifest['splits']['development']} development / {manifest['splits']['blind']} blind.\n\n"
        "```bash\nbash export_azookey.sh /path/to/AzooKeyKanaKanjiConverter\nzip -r azookey-results.zip azookey-results\n```\n\n"
        f"Converter: {CONVERTER}; existing n-best20 dictionary/flags unchanged. "
        "Legacy FineWeb candidates are reused locally, so this package exports only new sources. "
        "Exporting a blind candidate pool is permitted; no LM scoring is performed. "
        "Reviewed label-only changes can reuse candidates when query/context/IDs are identical; original outputs remain preserved. "
        "Reading, case exclusion or context changes require a fresh package/export. "
        f"Label quality: {manifest.get('label_quality', 'unreviewed_draft')}; formal human gold: {manifest['labels_formal_gold']}.\n", encoding="utf-8")
    with zipfile.ZipFile(args.output.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.output.rglob("*")):
            if path.is_file():
                archive.write(path, str(path.relative_to(args.output.parent)))


def import_candidates(args):
    manifest = read(args.prepared / "manifest.json")
    if args.output.exists():
        raise ValueError("Preserve prior imported versions.")
    payload = {}
    files = ("ajimee-input.json", "case-map.json", "azookey-candidates.json", "converter-version.txt",
             "dictionary-versions.txt", "swift-version.txt", "export-flags.txt", "completed.txt", "export.log")
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        for role in ("development", "blind"):
            payload[role] = {}
            for filename in files:
                suffix = f"azookey-results/{role}/{filename}"
                matches = [n for n in names if n == suffix or n.endswith("/"+suffix)]
                if len(matches) != 1:
                    raise ValueError(f"Expected exactly one {suffix}.")
                payload[role][filename] = archive.read(matches[0])
            label_changed = False
            for filename, answer_key in (("ajimee-input.json", "answer"), ("case-map.json", "answers")):
                returned = json.loads(payload[role][filename])
                expected = read(args.prepared / role / filename)
                if (len(returned) != len(expected) or any(
                        {k:v for k,v in a.items() if k != answer_key} != {k:v for k,v in b.items() if k != answer_key}
                        for a,b in zip(returned, expected))):
                    raise ValueError("Query/context/IDs changed; create a fresh candidate export.")
                label_changed |= returned != expected
            if label_changed:
                if not reviewed_and_frozen(manifest):
                    raise ValueError("Label-only rebasing requires a reviewed frozen release.")
                original_inputs = json.loads(payload[role]["ajimee-input.json"])
                inputs = read(args.prepared / role / "ajimee-input.json")
                raw = json.loads(payload[role]["azookey-candidates.json"])
                if len(raw["items"]) != len(inputs):
                    raise ValueError("Candidate count differs from the handoff.")
                for item, old_input, current in zip(raw["items"], original_inputs, inputs):
                    if (item["query"] != old_input["query"] or item["left_context"] != old_input["left_context"] or
                            item["answers"] != old_input["answer"]):
                        raise ValueError("Original candidate output differs from the original CLI input.")
                    item["answers"] = current["answer"]
                    item["max_rank"] = next((i for i,c in enumerate(item["outputs"]) if c["text"] in current["answer"]), -1)
                raw["stat"]["ranks"] = dict(collections.Counter(str(r["max_rank"]) for r in raw["items"]))
                for filename in ("ajimee-input.json", "case-map.json", "azookey-candidates.json"):
                    payload[role][filename.replace(".json", "-original.json")] = payload[role][filename]
                payload[role]["azookey-candidates.json"] = (json.dumps(raw,ensure_ascii=False,indent=2)+"\n").encode("utf-8")
                for filename in ("ajimee-input.json", "case-map.json"):
                    payload[role][filename] = (args.prepared / role / filename).read_bytes()
    fresh(args.output)
    manifest.setdefault("release_id", "draft-" + uuid.uuid4().hex)
    for role in PREPARED_ROLES:
        rows = read(args.prepared / role / "evaluation_items.json")
        dest = args.output / role
        track_files(dest, rows)
        exported = dest / "ajimee-results"
        exported.mkdir()
        if role != "regression":
            for filename, content in payload[role].items():
                (exported / filename).write_bytes(content)
        else:
            old = ROOT / "artifacts/benchmarks/ime-expanded-v21-candidates-v1/development"
            old_items = read(old / "evaluation_items.json")
            positions = {row["index"]: i for i,row in enumerate(old_items)}
            raw = read(old / "ajimee-results/azookey-candidates.json")
            raw["items"] = [raw["items"][positions[row["index"]]] for row in rows]
            raw["stat"]["query_count"] = len(rows)
            raw["stat"]["ranks"] = dict(collections.Counter(str(item["max_rank"]) for item in raw["items"]))
            write_json(exported / "azookey-candidates.json", raw)
            for filename in files:
                if filename not in ("ajimee-input.json", "case-map.json", "azookey-candidates.json"):
                    shutil.copyfile(old / "ajimee-results" / filename, exported / filename)
            for filename in ("ajimee-input.json", "case-map.json"):
                shutil.copyfile(dest / filename, exported / filename)
        stats = read(dest / "stats.json")
        imported = {**manifest, "format": FORMAT, "status": "complete", "split": role, "stats": stats,
                    "labels_formal_gold": bool(manifest["labels_formal_gold"] and role != "regression"),
                    "label_quality": "legacy_historical_draft" if role == "regression" else manifest.get("label_quality", "unreviewed_draft"),
                    "labels_frozen": bool(manifest.get("labels_frozen") and role != "regression"),
                    "review_completed": bool(manifest.get("review_completed") and role != "regression"),
                    "native_review_completed": bool(manifest.get("native_review_completed") and role != "regression"),
                    "source_corpus_split": "independent_source" if role != "regression" else "historical_development",
                    "candidate_provenance": "New Mac export" if role != "regression" else "Subset of frozen original 2000-case export; aggregate subset ranks recomputed, original files unchanged"}
        imported["reviewed_labels_rebased"] = "azookey-candidates-original.json" in payload.get(role, {})
        write_json(dest / "manifest.json", imported)
        load_standard_export(dest)
    manifest.update(status="candidates_imported", blind_lm_scored=False)
    write_json(args.output / "manifest.json", manifest)
    notice(args.output)


def plan(args):
    manifest = read(args.benchmark / "blind/manifest.json")
    if not reviewed_and_frozen(manifest):
        raise ValueError("Freeze reviewed labels before committing a blind plan.")
    consumed = (args.benchmark / "blind-consumption.json").exists()
    if args.output.exists() or consumed and not args.reference_test:
        raise ValueError("Preserve the existing plan/consumed blind release.")
    if args.reference_test and not consumed:
        raise ValueError("Reference-test mode is for a previously opened blind release.")
    import torch
    models = []
    for path in args.checkpoint:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        models.append({"checkpoint": str(path.resolve()), "bytes": path.stat().st_size,
                       "mtime_ns": path.stat().st_mtime_ns, "step": saved["step"],
                       "training_config": saved["config"], "signatures": saved["signatures"]})
        del saved
    write_json(args.output, {"plan_id": uuid.uuid4().hex, "benchmark_release_id": manifest["release_id"],
                            "created_utc": datetime.now(timezone.utc).isoformat(), "models": models,
                            "primary_metric": "FP32 suffix logP sum: per-source top1, independent micro and equal-source macro",
                            "selection_completed_before_blind": not args.reference_test,
                            "fresh_blind_evidence": not args.reference_test,
                            "role": "reference_test_after_consumption" if args.reference_test else "blind",
                            "label_quality": manifest.get("label_quality", "human_native_reviewed"),
                            "checkpoint_identity_policy": "Path/bytes/step/config/signatures; no repeated weight-file SHA256"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("review-ui", "freeze", "package", "import"):
        sub = subs.add_parser(name)
        sub.add_argument("--prepared", type=Path, default=DRAFT)
        sub.add_argument("--output", type=Path, required=True)
        if name == "freeze":
            sub.add_argument("--reviews", type=Path, required=True)
            sub.add_argument("--version", default="ime-standard-ja-v1")
            sub.add_argument("--change-note", default="Initial native review of the unscored draft")
            sub.add_argument("--review-mode", choices=("human-native", "ai-expert"), default="human-native")
        if name == "import":
            sub.add_argument("--archive", type=Path, required=True)
        if name == "review-ui":
            sub.add_argument("--refresh", action="store_true", help="Refresh generated templates only while all on-disk reviews remain pending")
    sub = subs.add_parser("plan")
    sub.add_argument("--benchmark", type=Path, required=True)
    sub.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--reference-test", action="store_true", help="Reuse an opened release as a fixed reference test, not as fresh blind evidence")
    args = parser.parse_args()
    {"review-ui": make_review, "freeze": freeze, "package": package, "import": import_candidates, "plan": plan}[args.command](args)
    print(f"Completed {args.command}: {args.output}")


if __name__ == "__main__":
    main()
