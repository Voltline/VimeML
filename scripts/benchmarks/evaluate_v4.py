"""Local FP32 V4 comparison on explicitly frozen development/regression inputs.

No blind discovery, training, label editing, export, or scoring-rule tuning.
Candidate snapshots are recovered from historical completed evaluation records
when the original Standard export manifest is unavailable, with that limitation
recorded explicitly. Original files remain unchanged.
"""
import argparse
import copy
import gc
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vimeml.benchmarks.ajimee import convert_items
from vimeml.benchmarks.comparison import paired, rows as read_rows
from vimeml.benchmarks.evaluate_ajimee import load_export, metrics, rerank, summarize
from vimeml.benchmarks.standard_ime import source_metrics
from vimeml.training.data import SentenceWindowDataset, file_sha, make_loader, write_json

V4 = ROOT / "artifacts/models/tiny-ja-v4-b-e8k-d384-l7"
BASELINE = ROOT / "artifacts/models/tiny-ja-v2.1-e16k-d320-l6-extend5/best.pt"
TRACKS = {
    "standard-development": ("outputs/training-v3-b/reused-development/epoch-2",
        "artifacts/benchmarks/ime-standard-ja-v1-ai-expert-r1/development", "development", 728),
    "legacy-regression": ("outputs/training-v3-b/legacy500-selected",
        "artifacts/benchmarks/ime-standard-ja-v1-ai-expert-r1/regression", "regression", 500),
    "ajimee": ("outputs/ime-eval/tiny-ja-v2.1-extend5-best-ajimee",
        "artifacts/benchmarks/ajimee-jwtd-v2-v1", "regression", 200),
    "historical-development": ("outputs/ime-eval/tiny-ja-v2.1-extend5-best-development",
        "artifacts/benchmarks/ime-dev-reviewed-v2", "development", 137),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def frozen_row(row):
    keep = ("position", "id", "source_index", "query", "left_context", "answers", "group",
            "candidates", "source", "provenance")
    result = {k: copy.deepcopy(row[k]) for k in keep if k in row}
    result.update(orders={"azookey": list(row["orders"]["azookey"])}, fallbacks={})
    if result["orders"]["azookey"] != [c["text"] for c in result["candidates"]]:
        raise ValueError("Historical candidate order differs from candidate payload")
    if len(set(result["orders"]["azookey"])) != len(result["candidates"]):
        raise ValueError("Duplicate frozen candidate")
    if any(not math.isfinite(c["score"]) for c in result["candidates"]):
        raise ValueError("Nonfinite frozen AzooKey score")
    return result


def prepare(output, model_directory=None, inventory_path=None):
    if (output / "plan.json").exists():
        plan = read(output / "plan.json")
        if set(plan["tracks"]) != set(TRACKS) or plan.get("blind_used") is not False:
            raise ValueError("Expected the explicit non-blind V4 evaluation plan")
        if model_directory is not None and any(
            (ROOT / model["path"]).parent != Path(model_directory).resolve()
            for name, model in plan["models"].items() if name != "v21"
        ):
            raise ValueError("Existing plan belongs to another model experiment")
        for name, track in plan["tracks"].items():
            cache, source, role, count = TRACKS[name]
            allowed = {ROOT / cache / "scores.jsonl", ROOT / cache / "metrics.json",
                       ROOT / source / "evaluation_items.json", output / "inputs" / (name + ".json")}
            if track["role"] != role or track["cases"] != count or {ROOT / p for p in track["input_hashes"]} != allowed:
                raise ValueError("Plan differs from the allowlisted development/regression inputs")
            for path, digest in track["input_hashes"].items():
                if file_sha(ROOT / path) != digest:
                    raise ValueError(f"Frozen input changed: {path}")
        return plan
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, (cache, source, role, count) in TRACKS.items():
        cache, source = ROOT / cache, ROOT / source
        report = read(cache / "metrics.json")
        if report["status"] != "complete":
            raise ValueError("Incomplete historical source")
        original = read_rows(cache / "scores.jsonl")
        _, mapping, _ = convert_items(read(source / "evaluation_items.json"))
        if len(original) != count or len(mapping) != count:
            raise ValueError("Frozen sample count changed")
        for row, expected in zip(original, mapping, strict=True):
            if any(row[k] != expected[k] for k in expected):
                raise ValueError("Historical candidate snapshot does not match retained labels/context")
        if name in ("standard-development", "legacy-regression"):
            if report["split"] != role or report["result_role"] != role:
                raise ValueError("Only explicit development/regression records allowed")
            for row, source_row in zip(original, read(source / "evaluation_items.json"), strict=True):
                if row["source"] != source_row["provenance"]["source"] or row["provenance"] != source_row["provenance"]:
                    raise ValueError("Frozen source provenance differs")
            provenance = "Exact historical candidate snapshot; original Standard export manifest unavailable locally"
        elif name == "ajimee":
            _, _, exported = load_export(source)
            if [frozen_row(r) for r in exported] != [frozen_row(r) for r in original]:
                raise ValueError("Cached pool differs from original retained AzooKey export")
            provenance = "Original retained export verified by existing AJIMEE loader"
        else:
            if file_sha(source / "evaluation_items.json") != report["benchmark_manifest"]["source_sha256"]:
                raise ValueError("Historical development label hash differs")
            provenance = "Exact historical candidate snapshot verified against original report's retained reference SHA256; original candidate export unavailable locally"
        snapshot = output / "inputs" / (name + ".json")
        snapshot.parent.mkdir(exist_ok=True)
        write_json(snapshot, [frozen_row(r) for r in original])
        paths = [cache / "scores.jsonl", cache / "metrics.json", source / "evaluation_items.json", snapshot]
        records[name] = {"cases": count, "role": role, "provenance": provenance,
            "label_quality": report.get("label_quality", report["benchmark_manifest"].get("label_status", "historical_public_references")),
            "labels_formal_gold": report.get("labels_formal_gold", False),
            "input_hashes": {p.relative_to(ROOT).as_posix(): file_sha(p) for p in paths}}
    models = {"v21": {"path": BASELINE.relative_to(ROOT).as_posix(), "sha256": file_sha(BASELINE)}}
    model_directory = V4 if model_directory is None else Path(model_directory).resolve()
    inventory_path = ROOT / "outputs/training-v4-20261015/remote-inventory.json" if inventory_path is None else Path(inventory_path).resolve()
    inventory = read(inventory_path)
    prefix = model_directory.relative_to(ROOT).as_posix() + "/"
    epoch_names = sorted({Path(r["relative"]).stem for r in inventory
        if r["relative"].startswith(prefix) and "/" not in r["relative"][len(prefix):]
        and re.fullmatch(r"epoch-\d+\.pt", Path(r["relative"]).name)}, key=lambda n: int(n.split("-")[1]))
    if not epoch_names:
        raise ValueError("Expected recorded complete-epoch checkpoint identities")
    for name in ["best", "last", "best-epoch", *epoch_names]:
        relative = (model_directory / (name + ".pt")).relative_to(ROOT).as_posix()
        record = next(r for r in inventory if r["relative"] == relative)
        models[name] = {"path": relative, "sha256": record["sha256"]}
    tokens = [read(ROOT / f"artifacts/token-data/{name}/manifest.json") for name in ("corpus-v2-16k", "corpus-v4-8k")]
    if [(t["splits"]["validation"]["sentences"], t["splits"]["validation"]["characters"]) for t in tokens] != [(260337, 11110489)] * 2:
        raise ValueError("Validation character/sentence populations differ")
    # V2 rows.bin addresses JSONL; V4 rows.bin addresses the same frozen TXT.
    # Their byte offsets differ by design. Compare original TXT identities instead.
    validation_text_hashes = [{v for p, v in t["input_sha256"].items() if p.replace("\\", "/").endswith("/validation.txt")} for t in tokens]
    if len(validation_text_hashes[0]) != 1 or validation_text_hashes[0] != validation_text_hashes[1]:
        raise ValueError("Original validation TXT identities differ")
    for suffix in ("sources.bin",):
        if len({file_sha(ROOT / f"artifacts/token-data/{n}/validation.{suffix}") for n in ("corpus-v2-16k", "corpus-v4-8k")}) != 1:
            raise ValueError("Validation row/source alignment differs")
    plan = {"format": "vimeml_v4_local_evaluation_plan_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "tracks": records, "models": models, "blind_used": False, "test_split_used": False,
        "epoch_checkpoints": epoch_names, "inventory_sha256": file_sha(inventory_path),
        "primary_model": "best", "primary_comparator": "v21", "precision": "fp32; TF32 disabled",
        "scoring": "Existing rerank(): joint-tokenized suffix full-vocabulary logP sum; no EOS/truncation; stable ties; whole-case fallback; full denominator",
        "secondary": "context-free sum and length mean are diagnostics; no score coefficient tuning",
        "development_selection": f"Highest Standard development contextual-sum Top-1 among best and {len(epoch_names)} complete epochs; ties prefer best, then earliest epoch. Exploratory selection, not held-out confirmation.",
        "validation": "Same original 260337 sentences / 11110489 Unicode characters; all windows and prediction targets including EOS; compare BPC, not token PPL",
        "generation": "Existing synthetic 6-token greedy/sample/beam settings; fixed 12 prompts with 16 generated tokens; character length reported because token budgets differ across vocabularies",
        "statistics": "Exact paired McNemar descriptive p-values; multiple checkpoint/source comparisons exploratory, no unadjusted significance declarations"}
    write_json(output / "plan.json", plan)
    return plan


def load_model(name, plan, device):
    from vimeml.training.infer import JapaneseLM
    from vimeml.training.infer_v4 import JapaneseLMV4
    path = ROOT / plan["models"][name]["path"]
    if path.exists():
        if file_sha(path) != plan["models"][name]["sha256"]:
            raise ValueError("Original checkpoint hash changed")
    else:
        path = path.parent / "inference" / path.name
        import torch
        packed = torch.load(path, map_location="cpu", weights_only=True)
        if packed.get("source_checkpoint_sha256") != plan["models"][name]["sha256"] or not packed.get("inference_only"):
            raise ValueError("Inference checkpoint source binding differs")
        del packed
    tokenizer = ROOT / ("artifacts/tokenizers/ja-unigram-16k-v2" if name == "v21" else "artifacts/tokenizers/v4-a/ja-unigram-8192")
    lm = (JapaneseLM if name == "v21" else JapaneseLMV4)(path, tokenizer, device)
    lm.metadata.update(checkpoint=str(path), original_checkpoint_sha256=plan["models"][name]["sha256"],
        device=str(device), tf32=False, parameters=lm.model.parameter_count())
    return lm


def validation(lm, name, output):
    from vimeml.training.train import evaluate
    token = ROOT / "artifacts/token-data" / ("corpus-v2-16k" if name == "v21" else "corpus-v4-8k")
    index = ROOT / "artifacts/training-data" / ("corpus-v2-c128" if name == "v21" else "corpus-v4-c128")
    meta = read(token / "manifest.json")
    for suffix in ("tokens.bin", "offsets.bin", "sources.bin"):
        filename = f"validation.{suffix}"
        if file_sha(token / filename) != meta["output_sha256"][filename]:
            raise ValueError("Validation token file changed")
    dataset = SentenceWindowDataset(token, index, "validation")
    loader = make_loader(dataset, batch_size=128, num_workers=2, shuffle=False, pin_memory=lm.device.type == "cuda")
    started = time.perf_counter()
    result = evaluate(lm.model, loader, lm.device, "fp32", meta["splits"]["validation"]["characters"])
    if result["prediction_pairs"] != meta["splits"]["validation"]["prediction_pairs"]:
        raise ValueError("Full validation target coverage differs")
    result.update(model=lm.metadata, sentences=meta["splits"]["validation"]["sentences"],
                  windows=len(dataset), token_manifest_sha256=file_sha(token / "manifest.json"),
                  elapsed_seconds=time.perf_counter() - started)
    write_json(output / "validation.json", result)
    dataset.close()
    del loader
    lm.model.eval()
    print(name, "FULL_VALIDATION_BPC", result["bpc"], flush=True)


def generation(lm, output):
    from vimeml.training import evaluate_ime as diagnostic
    previous = diagnostic.JapaneseLM
    try:
        diagnostic.JapaneseLM = lambda *args, **kwargs: lm
        diagnostic.main(["--device", str(lm.device), "--output", str(output / "synthetic")])
    finally:
        diagnostic.JapaneseLM = previous
    samples = []
    for prompt in read(ROOT / "configs/inference-prompts.json"):
        results = [lm.generate(prompt["prompt"], max_new_tokens=16, temperature=temp, seed=42) for temp in (0, .8)]
        for result in results:
            result["new_characters"] = len(result["text"]) - len(prompt["prompt"])
        samples.append({**prompt, "results": results})
    write_json(output / "generation.json", {"model": lm.metadata, "samples": samples,
        "qualitative_only": True, "prompt_sha256": file_sha(ROOT / "configs/inference-prompts.json")})


def run(name, output, plan, device, full=False, broad=False):
    dest = output / name
    dest.mkdir(exist_ok=True)
    lm = load_model(name, plan, device)
    for track in TRACKS if broad else ("standard-development",):
        target = dest / track
        if (target / "metrics.json").exists():
            if read(target / "metrics.json")["model"]["original_checkpoint_sha256"] != lm.metadata["original_checkpoint_sha256"]:
                raise ValueError("Existing evaluation model differs")
            continue
        inputs = read(output / "inputs" / (track + ".json"))
        started = time.perf_counter()
        rerank(lm, inputs)
        target.mkdir(exist_ok=True)
        (target / "scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in inputs), encoding="utf-8")
        result = {"status": "complete", "model": lm.metadata, "input": plan["tracks"][track],
            "metrics": summarize(inputs), "elapsed_seconds": time.perf_counter() - started}
        if "source" in inputs[0]:
            result["source_metrics"] = source_metrics(inputs, metrics)
        baseline = output / "v21" / track / "scores.jsonl"
        if name != "v21" and baseline.exists():
            comparison = paired(read_rows(baseline), inputs)
            comparison["paired"].pop("significant_at_0_05")
            write_json(target / "paired-v21.json", comparison)
        write_json(target / "metrics.json", result)
        print(name, track, result["metrics"]["all"]["lm_context_sum"]["top1_correct"], flush=True)
    if full and not (dest / "validation.json").exists():
        validation(lm, name, dest)
    if broad and not (dest / "generation.json").exists():
        generation(lm, dest)
    del lm
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation-v4-20261015")
    parser.add_argument("--name", default="plan", help="plan or a checkpoint name in the frozen inventory")
    parser.add_argument("--model-directory", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--broad", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    args.output = args.output.resolve()
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = prepare(args.output, args.model_directory, args.inventory)
    if args.name != "plan":
        if args.name not in plan["models"]:
            parser.error("Checkpoint name is absent from the frozen inventory")
        run(args.name, args.output, plan, args.device, args.full_validation, args.broad)


if __name__ == "__main__":
    main()
