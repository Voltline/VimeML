"""Publish a score-hidden development label audit and rescore frozen rankings."""

import copy
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.training.data import write_json
from vimeml.benchmarks.ajimee import convert_items
from vimeml.benchmarks.evaluate_ajimee import metrics
from vimeml.benchmarks.readings import DualReading
from vimeml.benchmarks.comparison import paired


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    audit = ROOT / "outputs/ime-eval/dev-label-audit-20261007"
    output = ROOT / "artifacts/benchmarks/ime-dev-label-reviewed-v3"
    if output.exists():
        raise ValueError("Preserve prior audit; use a new version.")
    original = load(ROOT / "artifacts/benchmarks/ime-dev-v2/evaluation_items.json")
    source = {row["index"]: row for row in original}
    mapping = load(audit / "sealed-mapping.json")
    blind = load(audit / "frozen-blind-review.json")
    reader = DualReading()
    accepted, quarantine, decisions = [], [], []
    for line in (audit / "blind-decisions.txt").read_text(encoding="utf-8").splitlines():
        ident, *forms = line.split("|")
        original_row = source[mapping[ident]]
        row = copy.deepcopy(original_row)
        reason = "同读音同词义表记；先重建全部输入，再检查原标签，不读取模型候选或得分决定标签。"
        if forms[0].startswith("QUARANTINE:"):
            decision = {
                "index": row["index"],
                "review_id": ident,
                "action": "quarantine",
                "reason_zh": forms[0].split(":", 1)[1],
            }
            quarantine.append({"case": row, "decision": decision})
        else:
            # All old forms were separately reviewed after blind reconstruction;
            # no old reference was found to require deletion among kept cases.
            answers = list(dict.fromkeys(row["expected_output"] + forms))
            decision = {
                "index": row["index"],
                "review_id": ident,
                "action": "extend" if answers != row["expected_output"] else "keep",
                "original_answers": row["expected_output"],
                "reviewed_answers": answers,
                "added_answers": [a for a in answers if a not in row["expected_output"]],
                "reason_zh": reason,
                "dictionary_readings": [{"form": a, **reader.analyse(a)} for a in answers],
            }
            row["expected_output"] = answers
            row["review_v3"] = decision
            accepted.append(row)
        decisions.append(decision)
    accepted.sort(
        key=lambda row: next(i for i, r in enumerate(original) if r["index"] == row["index"])
    )
    converted, mapped, stats = convert_items(accepted)
    output.mkdir(parents=True)
    for name, data in [
        ("evaluation_items.json", accepted),
        ("ajimee-input.json", converted),
        ("case-map.json", mapped),
        ("review-decisions.json", decisions),
        ("quarantine.json", quarantine),
    ]:
        write_json(output / name, data)
    manifest = {
        "format": "ime_development_label_audit_v3",
        "status": "complete",
        "role": "development_diagnostic",
        "source": "artifacts/benchmarks/ime-dev-v2",
        "stats": stats,
        "reviewed_cases": 137,
        "quarantined_cases": len(quarantine),
        "changed_kept_cases": sum(d["action"] == "extend" for d in decisions),
        "added_reference_forms": sum(len(d.get("added_answers", [])) for d in decisions),
        "model_order_hidden_during_review": True,
        "external_native_review": False,
        "prior_exposure": blind["prior_exposure"],
        "dictionary_versions": reader.versions,
        "reading_policy": "Dictionary outputs are supporting evidence; manually inspect alternate readings and numeric composition. Dictionaries agree incorrectly on some known readings, so agreement alone is not gold.",
        "manual_reading_exceptions": {
            "B007": "富士山 permits フジサン; parser chose フジヤマ",
            "B010": "明日 permits アシタ; parser chose アス",
            "B027": "洗濯物=センタクモノ; UniDic chose ブツ",
            "B059": "水曜日=スイヨウビ; 十二=ジュウニ",
            "B065": "1250日間=センニヒャクゴジュウニチカン; numeric composition",
            "B070": "売場 permits ウリバ; parser chose バイジョウ",
            "B089": "日本語=ニホンゴ",
            "B096": "三十日後=サンジュウニチゴ; numeric composition",
            "B098": "何を=ナニヲ; parser chose ナンヲ",
            "B106": "値切り=ネギリ; parser chose ネキリ",
            "B121": "十五分=ジュウゴフン; numeric composition",
            "B136": "三十日間=サンジュウニチカン; numeric composition",
        },
        "blind_reconstruction_amendment": {
            "B073": "Corrected our reconstruction 調整(チョウセイ) to 調節(チョウセツ) after dictionary check, before model-score joining; original gold was correct."
        },
        "candidate_policy": "Reuse exactly the original query/context/pools and precomputed model orders; revised answers only. No model training, inference or candidate export.",
        "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    write_json(output / "manifest.json", manifest)
    # Only after publishing the label version, join precomputed scores.
    gold = {f"ajimee:{r['index']}": r for r in accepted}
    reports = {}
    scores = {}
    for model, directory in [
        ("v1", "tiny-ja-v1-dev-v2-scores"),
        ("v2", "tiny-ja-v2.0-best-development"),
    ]:
        source_rows = [
            json.loads(s)
            for s in (ROOT / "outputs/ime-eval" / directory / "scores.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        reviewed = []
        for r in source_rows:
            if r["id"] not in gold:
                continue
            g = gold[r["id"]]
            assert r["query"] == g["input"] and r["left_context"] == g["context_text"]
            changed = copy.deepcopy(r)
            changed["prior_answers"] = changed["answers"]
            changed["answers"] = g["expected_output"]
            reviewed.append(changed)
        destination = audit / model
        destination.mkdir(parents=True)
        (destination / "scores.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reviewed), encoding="utf-8"
        )
        scores[model] = reviewed
        reports[model] = {name: metrics(reviewed, name) for name in reviewed[0]["orders"]}
    result = {
        "source_label_version": "ime-dev-v2 137 cases",
        "new_label_version": "ime-dev-label-reviewed-v3 135 cases",
        "inference_repeated": False,
        "frozen_original_scores_preserved": True,
        "metrics": reports,
        "paired": paired(scores["v1"], scores["v2"]),
        "note": "Score-hidden AI audit with prior exposure, not independent blind/native evaluation. Absolute percentages have a different denominator and label policy.",
    }
    write_json(audit / "comparison-reviewed.json", result)
    print(
        json.dumps(
            {
                "manifest": {
                    k: manifest[k]
                    for k in [
                        "reviewed_cases",
                        "quarantined_cases",
                        "changed_kept_cases",
                        "added_reference_forms",
                    ]
                },
                "metrics": {m: report["lm_context_sum"] for m, report in reports.items()},
                "paired": result["paired"]["paired"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
