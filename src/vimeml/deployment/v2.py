"""V2-only inference bundles, FP32 Core ML and gated INT8 block32 experiments.

The V1 deployment entry points and precision recipes remain frozen.
"""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

from vimeml.deployment.bundle import environment, fresh_directory, read_json, write_json
from vimeml.deployment.coreml import CoreMLForward, coremltools, inspect_spec
from vimeml.deployment.graph import trace_graph
from vimeml.deployment.validation import compare_rows, numeric_error, SCORING_CASES
from vimeml.training.data import file_sha
from vimeml.training.infer import JapaneseLM, ROOT
from vimeml.training.model_factory import configuration_for, create_model, model_from_checkpoint
from vimeml.training.evaluate_ime import score_candidates

BUNDLE_FORMAT = "vimeml_inference_bundle_v2"
PACKAGE_FORMAT = "vimeml_coreml_v2"
ARCHITECTURE = "tiny_gpt_v2"


def inventory(path, hashes=False):
    return {p.relative_to(path).as_posix(): {"bytes": p.stat().st_size,
            **({"sha256": file_sha(p)} if hashes else {})}
            for p in sorted(Path(path).rglob("*")) if p.is_file()}


def export_bundle(checkpoint, tokenizer, token_manifest, output):
    checkpoint, tokenizer, token_manifest, output = map(Path, (checkpoint, tokenizer, token_manifest, output))
    if output.exists():
        raise ValueError("Bundle output already exists.")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format") != "vimeml_tiny_gpt_v2" or saved.get("architecture") != ARCHITECTURE:
        raise ValueError("Expected an explicitly declared V2 checkpoint.")
    tokens = read_json(token_manifest)
    if file_sha(token_manifest) != saved["signatures"]["tokens"]:
        raise ValueError("Token manifest does not match checkpoint.")
    tokenizer_sha = file_sha(tokenizer)
    expected = [value for name, value in tokens["input_sha256"].items()
                if name.replace("\\", "/").rsplit("/", 1)[-1] == "tokenizer.model"]
    if expected != [tokenizer_sha]:
        raise ValueError("Tokenizer does not match checkpoint vocabulary.")
    # The transferred deployment checkpoint already binds the original best.pt
    # identity. Keep it for score comparisons; do not confuse it with this file.
    origin_sha = saved.get("source_checkpoint_sha256")
    if saved.get("deployment_only") and not origin_sha:
        raise ValueError("Deployment checkpoint is missing original checkpoint provenance.")
    checkpoint_sha = file_sha(checkpoint)
    if not origin_sha:
        origin_sha = checkpoint_sha
    source_audit = {}
    for name, digest in saved["signatures"]["training_code"].items():
        current = file_sha(ROOT / "src/vimeml/training" / name)
        source_audit[name] = {"checkpoint_sha256": digest, "current_sha256": current,
                              "matches": current == digest}
        # Training orchestration may change after training (e.g. local monitor
        # link removal); inference uses only these architecture definitions.
        if name in {"model.py", "model_v2.py", "model_factory.py"} and current != digest:
            raise ValueError(f"Frozen training source changed: {name}")
    state = saved["model"]
    if not torch.equal(state["lm_head.weight"], state["token_embedding.weight"]):
        raise ValueError("Expected identical tied embedding/head tensors.")
    model = model_from_checkpoint(saved).eval()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite learned weight.")
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer))
    special = {name: getattr(processor, f"{name}_id")() for name in ("pad", "unk", "bos", "eos")}
    if special != tokens["special_ids"] or processor.vocab_size() != model.config.vocab_size:
        raise ValueError("Tokenizer contract mismatch.")
    weights = {name: value.detach().cpu().float().contiguous() for name, value in state.items()
               if name != "lm_head.weight"}
    fresh_directory(output)
    torch.save(weights, output / "weights.pt")
    write_json(output / "config.json", model.configuration())
    shutil.copyfile(tokenizer, output / "tokenizer.model")
    shutil.copyfile(token_manifest, output / "token-manifest.json")
    result = {"format": BUNDLE_FORMAT, "status": "complete", "architecture": ARCHITECTURE,
              "precision": "fp32", "parameter_count": model.parameter_count(), "special_ids": special,
              "checkpoint_sha256": origin_sha, "deployment_checkpoint_sha256": checkpoint_sha,
              "checkpoint_step": saved["step"], "training_signatures": saved["signatures"],
              "source_audit": source_audit,
              "tied_lm_head": True, "files": inventory(output, hashes=True), "environment": environment(),
              "policy": "Unique FP32 inference weights; no optimizer/RNG. Joint suffix logP sum; no EOS/truncation/KV cache."}
    write_json(output / "manifest.json", result)
    return result


def verify_bundle(path, hashes=True):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if (manifest.get("format") != BUNDLE_FORMAT or manifest.get("status") != "complete" or
            manifest.get("architecture") != ARCHITECTURE or not manifest.get("tied_lm_head")):
        raise ValueError("Unsupported/incomplete V2 bundle.")
    if set(manifest["files"]) != {"weights.pt", "config.json", "tokenizer.model", "token-manifest.json"}:
        raise ValueError("Unexpected inference file list.")
    for name, entry in manifest["files"].items():
        if (path / name).stat().st_size != entry["bytes"] or (hashes and file_sha(path / name) != entry["sha256"]):
            raise ValueError(f"Bundle checksum mismatch: {name}")
    return manifest


class BundleLM(JapaneseLM):
    def __init__(self, path):
        path = Path(path)
        # Export binds immutable weights/tokenizer once. Later stages reuse
        # that record and check the file sizes, schema and loaded contract.
        # Call verify_bundle(path) explicitly when resources have changed.
        manifest = verify_bundle(path, hashes=False)
        self.device = torch.device("cpu")
        self.processor = spm.SentencePieceProcessor(model_file=str(path / "tokenizer.model"))
        self.special = {name: getattr(self.processor, f"{name}_id")() for name in ("pad", "unk", "bos", "eos")}
        self.model = create_model(manifest["architecture"], configuration_for(manifest["architecture"],
                                  read_json(path / "config.json"))).eval()
        state = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        if "lm_head.weight" in state:
            raise ValueError("Shared embedding must be stored only once.")
        loaded = self.model.load_state_dict(state, strict=False)
        if loaded.missing_keys != ["lm_head.weight"] or loaded.unexpected_keys:
            raise ValueError(f"Unexpected inference weights: {loaded}")
        if (self.special != manifest["special_ids"] or self.processor.vocab_size() != self.model.config.vocab_size
                or self.model.parameter_count() != manifest["parameter_count"]):
            raise ValueError("Bundle model/tokenizer contract mismatch.")
        self.forbidden = tuple(self.special[name] for name in ("pad", "unk", "bos"))
        self.metadata = {"architecture": ARCHITECTURE, "checkpoint_sha256": manifest["checkpoint_sha256"],
                         "checkpoint_step": manifest["checkpoint_step"],
                         "deployment_checkpoint_sha256": manifest["deployment_checkpoint_sha256"],
                         "tokenizer_sha256": manifest["files"]["tokenizer.model"]["sha256"],
                         "bundle_manifest_sha256": file_sha(path / "manifest.json"),
                         "model": self.model.configuration(), "precision": "fp32", "device": "cpu",
                         "environment": environment()}


def finish(output, model, metadata):
    output = Path(output)
    fresh_directory(output)
    model.save(str(output / "model.mlpackage"))
    files = inventory(output / "model.mlpackage")
    write_json(output / "manifest.json", {"format": PACKAGE_FORMAT, "status": "complete",
        "architecture": ARCHITECTURE, **metadata, "environment": environment(),
        "spec": inspect_spec(model), "package_files": files,
        "package_bytes": sum(entry["bytes"] for entry in files.values()),
        "note": "Logical package bytes; compiled size, resident memory and device footprint are separate."})


def verify_package(path):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if (manifest.get("format") != PACKAGE_FORMAT or manifest.get("status") != "complete" or
            manifest.get("architecture") != ARCHITECTURE):
        raise ValueError("Incomplete/unsupported V2 Core ML package.")
    if inventory(path / "model.mlpackage") != manifest["package_files"]:
        raise ValueError("Package file inventory changed.")
    return manifest


def convert(bundle, output):
    ct = coremltools()
    if Path(output).exists():
        raise ValueError("Conversion output exists.")
    lm = BundleLM(bundle)
    model = ct.convert(trace_graph(lm.model), source="pytorch", convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT32,
        inputs=[ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(lower_bound=1,
            upper_bound=lm.model.config.context_length, default=16)), dtype=np.int32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)], skip_model_load=True)
    model.short_description = "VimeML V2.1 RMSNorm/SwiGLU; FP32 computation; no KV cache."
    model.user_defined_metadata["bundle_manifest_sha256"] = lm.metadata["bundle_manifest_sha256"]
    finish(output, model, {"kind": "fp32", "minimum_ios": 18, "bundle": lm.metadata,
        "interface": {"input_ids": "int32 [1,T], 1<=T<=128; right PAD",
                      "logits": "float32 [1,T,16384]; full vocabulary; no softmax"},
        "precision_policy": "All learned parameters and computation FP32; explicit causal attention.",
        "conversion_source_sha256": file_sha(Path(__file__))})


def parameter_selection(opt, model, config):
    selected, excluded, configs = [], [], {}
    for name, entry in opt.get_weights_metadata(model, weight_threshold=config.weight_threshold).items():
        value = entry.val
        normalized = name.replace(".", "_")
        embedding = normalized.startswith(("model_token_embedding_weight", "model_position_embedding_weight"))
        consumers = [{"name": child.name, "type": child.op_type,
                      "inputs": dict(child.params_name_mapping)} for child in entry.child_ops]
        parameter = bool(consumers) and all(
            (child.op_type == "linear" and child.params_name_mapping.get("weight") == name) or
            (embedding and child.op_type == "gather" and child.params_name_mapping.get("x") == name)
            for child in entry.child_ops)
        record = {"name": name, "shape": list(value.shape), "elements": int(value.size),
                  "dtype": str(value.dtype), "consumers": consumers}
        if not parameter or value.ndim != 2 or not np.issubdtype(value.dtype, np.floating):
            record.update(reason="Structural constant, mask or non-matrix; left unchanged",
                          nonfinite_elements=int((~np.isfinite(value)).sum()))
            excluded.append(record)
        elif not np.isfinite(value).all():
            raise ValueError(f"Nonfinite learned parameter: {name}")
        elif value.size > config.weight_threshold:
            selected.append(record)
            configs[name] = config
    if not selected:
        raise ValueError("No eligible parameter matrices.")
    return opt.OptimizationConfig(global_config=None, op_name_configs=configs), {
        "policy": "V2 finite linear matrices and named embedding tables only; norms/masks unchanged",
        "selected_constants": selected, "excluded_constants": excluded,
        "selected_elements": sum(item["elements"] for item in selected)}


def compress(source, output, alignment):
    ct = coremltools()
    source, output, alignment = map(Path, (source, output, alignment))
    if output.exists():
        raise ValueError("Compression output exists.")
    original = verify_package(source)
    gate = read_json(alignment)
    if (original["kind"] != "fp32" or gate.get("format") != "vimeml_alignment_v2" or
            not gate.get("passed") or gate.get("coreml_manifest_sha256") != file_sha(source / "manifest.json")):
        raise ValueError("First pass alignment for this exact uncompressed V2 FP32 package.")
    from coremltools.optimize import coreml as opt
    model = ct.models.MLModel(str(source / "model.mlpackage"), skip_model_load=True)
    config = opt.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                        granularity="per_block", block_size=32, weight_threshold=2048)
    optimization, selection = parameter_selection(opt, model, config)
    print(f"Selected {len(selection['selected_constants'])} parameter matrices; "
          f"excluded {len(selection['excluded_constants'])} structural constants.", flush=True)
    result = opt.linear_quantize_weights(model, config=optimization)
    if not any(name.startswith("constexpr_") for name in inspect_spec(result)["operations"]):
        raise ValueError("No compressed weights found.")
    finish(output, result, {"kind": "linear8_fp32_compute", "minimum_ios": 18,
        "bundle": original["bundle"], "interface": original["interface"],
        "source_manifest_sha256": file_sha(source / "manifest.json"),
        "fp32_alignment_sha256": file_sha(alignment), "conversion_source_sha256": file_sha(Path(__file__)),
        "compression": {"method": "linear_symmetric", "bits": 8, "block_size": 32,
                        "granularity": "per_block", "selection": selection},
        "precision_policy": "INT8 parameter matrices, FP32 compute; norms/masks unchanged; no activation quantization."})


class CoreMLLM(BundleLM):
    def __init__(self, bundle, directory):
        super().__init__(bundle)
        manifest = verify_package(directory)
        if manifest["bundle"]["bundle_manifest_sha256"] != self.metadata["bundle_manifest_sha256"]:
            raise ValueError("Core ML package belongs to another bundle.")
        self.model = CoreMLForward(directory, self.model.config, "CPU_ONLY")
        self.metadata.update(precision=manifest["kind"], compute_units="CPU_ONLY",
                             coreml_manifest_sha256=file_sha(Path(directory) / "manifest.json"))


@torch.inference_mode()
def align(lm, reference, output, atol=3e-4, rtol=3e-4):
    reference, output = map(Path, (reference, output))
    if output.exists():
        raise ValueError("Alignment output exists.")
    info = read_json(reference / "reference.json")
    if (info.get("format") != "vimeml_v2_mac_preparation_reference_v1" or
            info["checkpoint_step"] != lm.metadata["checkpoint_step"]):
        raise ValueError("Invalid V2 preparation reference.")
    errors, actuals = [], {}
    with np.load(reference / "logits.npz", allow_pickle=False) as arrays:
        for name, case in info["cases"].items():
            ids = arrays[name + "_input_ids"]
            actual = lm.model(torch.from_numpy(ids.copy())).numpy()
            actuals[name] = actual
            valid = case["valid_prefix"]
            error = numeric_error(arrays[name + "_logits"][:, :valid], actual[:, :valid], atol, rtol)
            error.update(id=name, length=case["length"], valid_prefix=valid,
                         last_top1_equal=int(actual[0, valid-1].argmax()) == int(arrays[name + "_logits"][0, valid-1].argmax()))
            errors.append(error)
    invariants = {name: numeric_error(actuals["short"], actuals[name][:, :16], atol, rtol)
                  for name in ("right_pad", "changed_future")}
    report = {"format": "vimeml_alignment_v2", "passed": all(
        item["outside_tolerance"] == 0 for item in [*errors, *invariants.values()]),
        "model": lm.metadata, "coreml_manifest_sha256": lm.metadata.get("coreml_manifest_sha256"),
        "reference": info, "tolerances": {"atol": atol, "rtol": rtol},
        "logits": errors, "invariants": invariants,
        "note": "Strict Windows FP32 fixture comparison. Quantized ranking quality is evaluated separately; failures are preserved."}
    fresh_directory(output)
    write_json(output / "alignment.json", report)
    return report


def evaluate(lm, benchmark, baseline, output):
    from vimeml.benchmarks.evaluate_ajimee import load_export, rerank, summarize, write_summary
    benchmark, baseline, output = map(Path, (benchmark, baseline, output))
    if output.exists():
        raise ValueError("Evaluation output exists.")
    manifest, provenance, rows = load_export(benchmark)
    if manifest.get("split") == "blind":
        raise ValueError("Blind data is outside this deployment experiment.")
    prior = read_json(baseline / "metrics.json")
    for key in ("checkpoint_sha256", "tokenizer_sha256", "model"):
        if prior["model"][key] != lm.metadata[key]:
            raise ValueError(f"Frozen baseline identity mismatch: {key}")
    if prior["benchmark_manifest"] != manifest or prior["export_provenance"] != provenance:
        raise ValueError("Frozen baseline candidate provenance differs.")
    if file_sha(baseline / "scores.jsonl") != prior["files_sha256"]["scores.jsonl"]:
        raise ValueError("Frozen baseline scores changed.")
    before = [json.loads(line) for line in (baseline / "scores.jsonl").read_text().splitlines()]
    started = time.perf_counter()
    rerank(lm, rows)
    comparison = compare_rows(before, rows)
    fresh_directory(output)
    (output / "scores.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                               for row in rows), encoding="utf-8")
    report = {"format": "vimeml_deployment_metrics_v2", "status": "complete", "model": lm.metadata,
        "benchmark_manifest": manifest, "export_provenance": provenance, "metrics": summarize(rows),
        "baseline_metrics": prior["metrics"], "comparison": comparison,
        "labels_formal_gold": manifest.get("labels_formal_gold"),
        "elapsed_seconds": time.perf_counter() - started,
        "files_sha256": {"scores.jsonl": file_sha(output / "scores.jsonl")},
        "policy": "Frozen original candidates/labels; LM-only joint suffix logP sum, stable ties, whole-case fallback; no EOS/truncation; no blind selection."}
    write_json(output / "metrics.json", report)
    write_summary(output / "results.md", report, rows)
    return report


@torch.inference_mode()
def samples(lm, baseline, output):
    baseline, output = map(Path, (baseline, output))
    if output.exists():
        raise ValueError("Samples output exists.")
    prior = read_json(baseline)
    for key in ("checkpoint_sha256", "tokenizer_sha256", "model"):
        if prior["metadata"][key] != lm.metadata[key]:
            raise ValueError(f"Sample baseline identity mismatch: {key}")
    cases = []
    for case in prior["cases"]:
        greedy = lm.generate(case["prompt"], prior["max_new_tokens"], temperature=0,
                             seed=case["greedy"]["seed"])
        pieces = lm.next_pieces(case["prompt"])
        cases.append({"kind": case["kind"], "prompt": case["prompt"], "greedy": greedy,
                      "baseline_greedy": case["greedy"], "next_pieces": pieces,
                      "greedy_tokens_equal": greedy["new_token_ids"] == case["greedy"]["new_token_ids"],
                      "next_top1_equal": pieces[0]["id"] == case["next_pieces"][0]["id"]})
    fresh_directory(output)
    write_json(output / "samples.json", {"metadata": lm.metadata, "cases": cases,
        "greedy_equal_cases": sum(case["greedy_tokens_equal"] for case in cases),
        "note": "Same 12 fixed prompts and greedy settings. Stochastic CPU/CUDA samples are not a deterministic alignment gate."})


@torch.inference_mode()
def device_fixtures(lm, output):
    """Bind tokenization and quantized scores to the new client resource set."""
    output = Path(output)
    if output.exists():
        raise ValueError("Fixture output exists.")
    candidates = []
    for case in SCORING_CASES:
        scored = score_candidates(lm, case["context"], case["candidates"])
        for result in scored["candidates"]:
            ids = [lm.special["bos"], *lm.processor.encode(case["context"] + result["text"], out_type=int)]
            candidates.append({"context": case["context"], "text": result["text"],
                "input_ids": ids[:-1], "targets": ids[1:],
                "score_start": scored["common_prefix_tokens_including_bos"] - 1,
                "log_probability_sum": result["log_probability_sum"]})
    write_json(output, {"format": "vimeml_device_fixtures_v2", "model": lm.metadata,
                        "bundle_manifest_sha256": lm.metadata["bundle_manifest_sha256"],
                        "candidates": candidates})
