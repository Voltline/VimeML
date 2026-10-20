"""Score frozen Chinese IME outputs against independent native-speaker labels.

Legacy predictions are reused. New inference receives input/context/candidates only;
human answers never enter candidate generation or reranker inference.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from .chinese_review import digest


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value, frozen=False):
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"Frozen input differs: {path}")
    path.write_text(content, encoding="utf-8")


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values), encoding="utf-8")


def prepare(root, annotation_path, artifacts, output):
    packet = read(annotation_path)
    original = read(root / "outputs/chinese-human-review-20261017/dist/blind-inputs.json")
    mapping = read(root / "artifacts/chinese-human-review-20261017/mapping-private.json")
    if packet.get("schema_version") != 1 or packet.get("dataset_id") != original["dataset_id"] or packet.get("cases") != original["cases"]:
        raise ValueError("Annotation packet does not match the frozen blinded inputs")
    labels = packet["annotations"]
    if len(labels) != 100 or len({r["id"] for r in labels}) != 100 or {r["id"] for r in labels} != {c["id"] for c in original["cases"]}:
        raise ValueError("Annotation IDs or count differ")
    for label in labels:
        answers = label.get("answers")
        if not label.get("reviewed_at") or not isinstance(answers, list) or not all(isinstance(a, str) and a.strip() for a in answers):
            raise ValueError(f"Unconfirmed or malformed annotation: {label['id']}")
        if label["decision"] not in {"single", "multiple", "uncertain", "invalid"}:
            raise ValueError("Unknown annotation decision")
        if label["decision"] == "single" and len(answers) != 1 or label["decision"] == "multiple" and len(set(answers)) < 2:
            raise ValueError(f"Inconsistent answer count: {label['id']}")
    artifacts.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    target = artifacts / "native-annotations-original.json"
    if target.exists():
        if target.read_bytes() != Path(annotation_path).read_bytes():
            raise FileExistsError("An annotation revision must use a new experiment directory")
    else:
        shutil.copyfile(annotation_path, target)
    write(artifacts / "blind-inputs.json", original, True)
    write(artifacts / "mapping-private.json", mapping, True)
    manifests = read(root / "artifacts/chinese-ime-research-20261017/models/manifest.json")
    dependencies = [root / "outputs/chinese-human-review-20261017/dist/blind-inputs.json",
                    root / "artifacts/chinese-human-review-20261017/mapping-private.json",
                    root / "artifacts/chinese-ime-research-20261017/probe/target/release/vimeml-chinese-probe.exe",
                    root / "artifacts/chinese-ime-research-20261017/capture.exe"]
    prior = root / "outputs/chinese-ime-research-20261017"
    for mode in ("grammar", "nbest-grammar"):
        dependencies.append(prior / f"rime-{mode}-cases.jsonl")
        for size in ("4M", "25M"):
            dependencies.append(prior / f"pinyin-ime-reranker-{size}-{mode}-scores.jsonl")
    dependencies += [prior / "pinyingpt/pinyingpt-scores.jsonl", prior / "pinyingpt/pinyingpt-summary.json"]
    manifest = {"dataset_id": original["dataset_id"], "input_sha256": digest(original["cases"]),
        "native_export_sha256": sha(target), "native_exported_at": packet["exported_at"],
        "confirmed": 100, "decisions": dict(Counter(r["decision"] for r in labels)),
        "models": manifests, "cached_files": {str(p.relative_to(root)): sha(p) for p in dependencies},
        "policy": "Same fixed models and grammar/nbest-grammar configurations as the pre-label experiment. Exact listed-answer membership, whitespace trimming only; first listed answer separately. Exclude uncertain/invalid from scorable denominators; disclose their counts. No tuning on native annotations."}
    write(artifacts / "evaluation-manifest.json", manifest, True)
    lookup = {r["blind_id"]: r for r in mapping["mapping"]}
    fresh = [c for c in original["cases"] if lookup[c["id"]]["suite"] == "new_diagnostic"]
    if len(fresh) != 40:
        raise ValueError("Expected exactly 40 new cases")
    (artifacts / "new-inputs.tsv").write_text("".join(c["id"] + "\t" + c["input"] + "\n" for c in fresh), encoding="utf-8")
    write_rows(artifacts / "new-model-inputs.jsonl", fresh)
    print("Validated and froze all 100 native annotations; 40 model inputs contain no human answers.", flush=True)


def capture_and_score(root, artifacts, output):
    upstream = root / "artifacts/chinese-ime-research-20261017"
    prior = root / "outputs/chinese-ime-research-20261017"
    packet = read(artifacts / "blind-inputs.json")
    mapping = {m["blind_id"]: m for m in read(artifacts / "mapping-private.json")["mapping"]}
    labels = {l["id"]: l for l in read(artifacts / "native-annotations-original.json")["annotations"]}
    frozen = read(artifacts / "evaluation-manifest.json")
    for relative, expected in frozen["cached_files"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"Cached artifact changed: {relative}")
    dictionary = set()
    for line in (upstream / "rime/shared/pinyin_simp.dict.yaml").read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 2:
            dictionary.add((fields[0], fields[1].replace(" ", "").replace("'", "")))
    fresh = rows(artifacts / "new-model-inputs.jsonl")
    for mode in ("grammar", "nbest-grammar"):
        is_nbest = mode == "nbest-grammar"
        shared = upstream / ("rime-nbest/shared" if is_nbest else "rime/shared")
        schema = "research_multi_grammar" if is_nbest else "research_grammar"
        user = artifacts / f"rime-user-{mode}"
        user.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["PATH"] = str(upstream / "librime/dist/lib") + os.pathsep + env["PATH"]
        proc = subprocess.run([str(upstream / "capture.exe"), str(shared), str(user), schema,
            str(artifacts / "new-inputs.tsv")], capture_output=True, check=True, env=env)
        (output / f"new-{mode}-capture.tsv").write_bytes(proc.stdout)
        (output / f"new-{mode}-capture-stderr.txt").write_bytes(proc.stderr)
        pools = {c["id"]: [] for c in fresh}
        for line in proc.stdout.decode("utf-8").splitlines():
            ident, index, text, complete = line.split("\t")
            if int(index) != len(pools[ident]):
                raise ValueError("Candidate ordering is inconsistent")
            key = next(c["input"] for c in fresh if c["id"] == ident).replace("'", "")
            trusted = complete == "1" and (text, key) in dictionary
            pools[ident].append({"text": text, "answers_key": complete == "1", "trusted_dictionary_hit": trusted})
        new_cases = [dict(c, candidates=pools[c["id"]]) for c in fresh]
        write_rows(output / f"new-{mode}-cases.jsonl", new_cases)
        nonempty = output / f"new-{mode}-nonempty.jsonl"
        write_rows(nonempty, [c for c in new_cases if c["candidates"]])
        for item in frozen["models"]:
            name = item["id"].split("/")[-1]
            weight = upstream / f"models/{name}/sentence-model.safetensors"
            if sha(weight) != item["sha256"]:
                raise ValueError(f"Model weight changed: {name}")
            result = subprocess.run([str(upstream / "probe/target/release/vimeml-chinese-probe.exe"),
                str(weight), str(nonempty)], input="continue\n", capture_output=True, check=True,
                text=True, encoding="utf-8")
            scored = [json.loads(line) for line in result.stdout.splitlines()]
            write(output / f"{name}-{mode}-load.json", scored[0])
            write_rows(output / f"{name}-{mode}-new-scores.jsonl", scored[1:])
        old = {c["id"]: c for c in rows(prior / f"rime-{mode}-cases.jsonl")}
        new = {c["id"]: c for c in new_cases}
        old_scores = {size: {r["id"]: r for r in rows(prior / f"pinyin-ime-reranker-{size}-{mode}-scores.jsonl")} for size in ("4M", "25M")}
        new_scores = {size: {r["id"]: r for r in rows(output / f"pinyin-ime-reranker-{size}-{mode}-new-scores.jsonl")} for size in ("4M", "25M")}
        combined = []
        for case in packet["cases"]:
            ident = case["id"]
            m = mapping[ident]
            legacy = m["suite"] == "legacy_audit"
            pool = old[m["source_id"]] if legacy else new[ident]
            if pool["input"] != case["input"] or pool["context"] != case["context"]:
                raise ValueError("Prediction input/context identity mismatch")
            cs = pool["candidates"]
            predictions = {"rime": {"index": 0 if cs else None, "text": cs[0]["text"] if cs else None,
                "full_key": cs[0]["answers_key"] if cs else False}}
            for size in ("4M", "25M"):
                s = (old_scores[size][m["source_id"]] if legacy else new_scores[size].get(ident))
                for policy, field in (("gated", "chosen"), ("ungated", "ungated"), ("no_context", "no_context")):
                    i = s[field] if s else None
                    predictions[f"{size}/{policy}"] = {"index": i, "text": cs[i]["text"] if i is not None else None,
                        "full_key": cs[i]["answers_key"] if i is not None else False}
            combined.append(dict(case, suite=m["suite"], tags=m["tags"], source_id=m["source_id"],
                original_reference_only=m["original_reference_only"], annotation=labels[ident],
                candidates=cs, predictions=predictions, prediction_source="cached_pre_label" if legacy else "new_fixed_model_inference"))
        write_rows(output / f"evaluated-{mode}.jsonl", combined)
        print(mode, "100 joined cases; legacy inference reused, new cases scored.", flush=True)


def summarize(selected):
    eligible = [c for c in selected if c["annotation"]["decision"] in {"single", "multiple"}]
    methods = list(selected[0]["predictions"]) if selected else []
    result = {"total": len(selected), "scorable": len(eligible),
        "decisions": dict(Counter(c["annotation"]["decision"] for c in selected)), "methods": {}}
    result["candidate_ceiling"] = sum(any(ca["answers_key"] and ca["text"] in {a.strip() for a in c["annotation"]["answers"]} for ca in c["candidates"]) for c in eligible)
    for name in methods:
        hits = preferred = rescued = broken = unlisted_full = incomplete = 0
        for case in eligible:
            answers = [a.strip() for a in case["annotation"]["answers"]]
            p, baseline = case["predictions"][name], case["predictions"]["rime"]
            before = baseline["full_key"] and baseline["text"] in answers
            after = p["full_key"] and p["text"] in answers
            hits += after
            preferred += p["full_key"] and p["text"] == answers[0]
            rescued += after and not before
            broken += before and not after
            unlisted_full += p["full_key"] and p["text"] not in answers
            incomplete += not p["full_key"]
        result["methods"][name] = {"accepted_set_hits": hits, "first_listed_hits": preferred,
            "accepted_rate": hits / len(eligible) if eligible else None,
            "first_listed_rate": preferred / len(eligible) if eligible else None,
            "rescued": rescued, "broken": broken, "full_output_unlisted": unlisted_full,
            "no_full_output": incomplete}
    return result


def report(root, artifacts, output):
    summaries, adjudication = {}, []
    for mode in ("grammar", "nbest-grammar"):
        cases = rows(output / f"evaluated-{mode}.jsonl")
        groups = {"legacy_audit": [c for c in cases if c["suite"] == "legacy_audit"],
                  "new_diagnostic": [c for c in cases if c["suite"] == "new_diagnostic"]}
        groups.update({f"legacy_{d}": [c for c in groups["legacy_audit"] if c["annotation"]["decision"] == d] for d in ("single", "multiple")})
        groups.update({f"new/{tag}": [c for c in groups["new_diagnostic"] if c["tags"] == tag] for tag in sorted({c["tags"] for c in groups["new_diagnostic"]})})
        summaries[mode] = {group: summarize(selected) for group, selected in groups.items()}
        for case in cases:
            if case["annotation"]["decision"] not in {"single", "multiple"}:
                continue
            unlisted = {}
            for name in ("rime", "4M/gated", "25M/gated"):
                p = case["predictions"][name]
                if p["full_key"] and p["text"] not in case["annotation"]["answers"]:
                    unlisted.setdefault(p["text"], []).append(name)
            if unlisted:
                adjudication.append({"mode": mode, "id": case["id"], "input": case["input"],
                    "context": case["context"], "listed_answers": case["annotation"]["answers"],
                    "unlisted_full_outputs": unlisted, "note": "Unlisted exact strings need adjudication; not automatically confirmed semantic errors."})
    write(output / "summary.json", summaries)
    write_rows(output / "unlisted-outputs-for-adjudication.jsonl", adjudication)
    primary = rows(output / "evaluated-nbest-grammar.jsonl")
    changes = [{"id": c["id"], "source_id": c["source_id"], "input": c["input"], "old_reference": c["original_reference_only"],
        "native_answers": c["annotation"]["answers"], "notes": c["annotation"]["notes"]} for c in primary
        if c["suite"] == "legacy_audit" and c["original_reference_only"] not in c["annotation"]["answers"]]
    write(output / "old-reference-disagreements.json", changes)
    cached_pg = rows(root / "outputs/chinese-ime-research-20261017/pinyingpt/pinyingpt-scores.jsonl")
    by_source = {c["source_id"]: c for c in primary if c["suite"] == "legacy_audit"}
    pg = {}
    for mode in ("full", "abbr"):
        group = [r for r in cached_pg if r["suite"] == "upstream_sentences" and r["mode"] == mode]
        pg[mode] = {"n": len(group), "accepted_top1": sum(r["predictions"][0] in by_source[r["id"]]["annotation"]["answers"] for r in group),
            "accepted_top10": sum(any(p in by_source[r["id"]]["annotation"]["answers"] for p in r["predictions"]) for r in group),
            "first_listed_top1": sum(r["predictions"][0] == by_source[r["id"]]["annotation"]["answers"][0] for r in group),
            "raw_reading_mismatch": sum(not r["matches_raw_input"] for r in group),
            "task_boundary": "Prior segmented, reference-derived syllable inputs; not raw-keystroke comparison. No new PinyinGPT inference using human labels."}
    write(output / "pinyingpt-cached-rejudgment.json", pg)
    for group in ("legacy_audit", "legacy_single", "legacy_multiple", "new_diagnostic"):
        s = summaries["nbest-grammar"][group]
        print(group, "n/scorable", s["total"], s["scorable"], "ceiling", s["candidate_ceiling"],
              {m: s["methods"][m] for m in ("rime", "4M/gated", "25M/gated")}, flush=True)
