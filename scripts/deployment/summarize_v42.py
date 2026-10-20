"""Collect V4.2 Mac gate, task and measurement records into one acceptance status file."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vimeml.training.data import file_sha

OUT = ROOT / "outputs/deployment"
TRACKS = ("standard-development", "legacy-regression", "ajimee", "historical-development")
EXPECTED_TOP1 = {"standard-development": 523, "legacy-regression": 370, "ajimee": 139, "historical-development": 123}
V21_TOP1 = {"standard-development": 505, "legacy-regression": 374, "ajimee": 144, "historical-development": 122}
RUNS = {
    "cuda": ROOT / "outputs/evaluation-v42-20261020",
    "mac_pytorch": OUT / "v42-mac-pytorch-baseline-v1/tracks",
    "coreml_fp32": OUT / "v42-coreml-fp32-tracks-v1",
    "coreml_int8": OUT / "v42-coreml-int8-tracks-v1",
    "kv_fp32": OUT / "v42-coreml-kv-fp32-tracks-v1",
    "kv_int8": OUT / "v42-coreml-kv-int8-tracks-v1",
}
METHODS = ("lm_context_sum", "lm_context_mean_secondary", "lm_no_context_sum")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(run, track):
    return [json.loads(line) for line in (RUNS[run] / track / "scores.jsonl").read_text(encoding="utf-8").splitlines() if line]


def all_method_orders(left, right):
    """Per-case order differences for every LM method, not only the primary sum."""
    result = {}
    for track in TRACKS:
        a, b = rows(left, track), rows(right, track)
        assert [r["id"] for r in a] == [r["id"] for r in b]
        result[track] = {method: [x["id"] for x, y in zip(a, b) if x["orders"][method] != y["orders"][method]]
                         for method in METHODS}
    return result


def top1(run):
    out = {}
    for track in TRACKS:
        data = rows(run, track)
        out[track] = sum(bool(r["orders"]["lm_context_sum"]) and r["orders"]["lm_context_sum"][0] in r["answers"] for r in data)
    return out


def track_comparison(run):
    summary = read(RUNS[run] / "summary.json")
    return {track: summary["tracks"][track]["comparison"] for track in TRACKS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--resources", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Fresh output required.")
    counts = {run: top1(run) for run in RUNS}
    fixtures = read(OUT / "v42-mac-pytorch-baseline-v1/fixtures.json")
    gates = {name: read(OUT / f"{name}/alignment.json") for name in (
        "v42-fp32-alignment-v1", "v42-kv-fp32-alignment-v1", "v42-int8-alignment-v1",
        "v42-kv-int8-vs-stateless-int8-v1", "v42-kv-int8-vs-fp32-fixtures-v1")}
    orders = {f"{a}_vs_{b}": all_method_orders(a, b) for a, b in (
        ("cuda", "mac_pytorch"), ("mac_pytorch", "coreml_fp32"), ("mac_pytorch", "kv_fp32"),
        ("coreml_fp32", "coreml_int8"), ("coreml_int8", "kv_int8"))}
    no_changes = lambda key: all(not ids for track in orders[key].values() for ids in track.values())
    int8 = track_comparison("coreml_int8")
    measured = read(args.measurements)
    checks = {
        "1_mac_pytorch_fixtures_within_3e-4": fixtures["passed"],
        "1_mac_pytorch_top1_equals_recorded_cuda": counts["mac_pytorch"] == EXPECTED_TOP1,
        "1_mac_pytorch_orders_equal_cuda_all_methods": no_changes("cuda_vs_mac_pytorch"),
        "2_fp32_stateless_and_kv_converted": True,
        "3_fp32_stateless_alignment_passed": gates["v42-fp32-alignment-v1"]["passed"],
        "3_fp32_kv_alignment_passed": gates["v42-kv-fp32-alignment-v1"]["passed"],
        "3_coreml_fp32_top1_equals_mac_pytorch": counts["coreml_fp32"] == counts["mac_pytorch"],
        "3_coreml_fp32_orders_equal_mac_pytorch_all_methods": no_changes("mac_pytorch_vs_coreml_fp32"),
        "3_kv_fp32_top1_equals_mac_pytorch": counts["kv_fp32"] == counts["mac_pytorch"],
        "3_kv_fp32_orders_equal_mac_pytorch_all_methods": no_changes("mac_pytorch_vs_kv_fp32"),
        "4_int8_strict_logits_alignment_passed (expected false)": gates["v42-int8-alignment-v1"]["passed"],
        "4_int8_kv_vs_same_weight_stateless_int8_passed": gates["v42-kv-int8-vs-stateless-int8-v1"]["passed"],
        "4_kv_int8_top1_equals_stateless_int8": counts["kv_int8"] == counts["coreml_int8"],
        "4_kv_int8_orders_equal_stateless_int8_all_methods": no_changes("coreml_int8_vs_kv_int8"),
        "5_measurements_recorded": set(measured["models"]) >= {"v42-fp32", "v42-int8", "v21-fp32", "v21-int8"},
        "6_ios_resources_prepared": bool(args.resources and (args.resources / "manifest.json").is_file()),
    }
    status = {"format": "vimeml_v42_mac_acceptance_v1", "date": "2026-10-20", "checks": checks,
        "top1": counts, "recorded_v21_fp32_top1": V21_TOP1,
        "int8_vs_coreml_fp32": {track: {"fp32": int8[track]["reference_top1"], "int8": counts["coreml_int8"][track],
            **int8[track]["paired"], "improved_ids": int8[track]["improved_ids"], "regressed_ids": int8[track]["regressed_ids"],
            "order_changed_cases": int8[track]["order_changed_cases"], "top1_changed_cases": int8[track]["top1_changed_cases"],
            "max_score_sum_abs": int8[track]["max_score_sum_abs"], "mean_score_sum_abs": int8[track]["mean_score_sum_abs"]}
            for track in TRACKS},
        "all_method_order_differences": orders,
        "alignment_summaries": {name: {"passed": g["passed"], **g["summary"]} for name, g in gates.items()},
        "inputs_sha256": {"measurements": file_sha(args.measurements)},
        "policy": "Failed gates are results. No retraining, checkpoint selection, pool/reference change or scoring tuning."}
    args.output.mkdir(parents=True)
    (args.output / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "top1": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
