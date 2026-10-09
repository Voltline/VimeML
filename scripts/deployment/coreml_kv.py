"""Convert, gate, compare and package the independent V2 KV-cache experiment."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from vimeml.deployment import kv_cache as kv, v2
from vimeml.deployment.bundle import fresh_directory, read_json, write_json
from vimeml.deployment.validation import numeric_error
from vimeml.training.data import file_sha
from vimeml.training.evaluate_ime import common_prefix_length


def align(bundle, directory, reference, output, baseline=None):
    runtime = kv.Runtime(bundle, directory)
    old = v2.CoreMLLM(bundle, baseline) if baseline else None
    errors, invariants, saved = [], [], {}
    info = read_json(reference / "reference.json")
    if info.get("format") != "vimeml_v2_mac_preparation_reference_v1" or info["checkpoint_step"] != runtime.metadata["checkpoint_step"]:
        raise ValueError("Invalid frozen reference identity.")
    if old and any(old.metadata[key] != runtime.metadata[key] for key in ("checkpoint_sha256", "tokenizer_sha256", "model")):
        raise ValueError("Stateless baseline model/tokenizer identity differs.")
    with np.load(reference / "logits.npz", allow_pickle=False) as arrays:
        for name, case in info["cases"].items():
            ids = arrays[name + "_input_ids"][0]
            expected = old.model(torch.tensor([ids.tolist()])).numpy() if old else arrays[name + "_logits"]
            patterns = [[len(ids)], [1] * len(ids)]
            if len(ids) > 3: patterns.append([3, len(ids)-3])
            for chunks in patterns:
                parts, past, state = [], 0, None
                for size in chunks:
                    logits, state = runtime.step(ids[past:past+size], state, past)
                    parts.append(logits); past += size
                actual = np.concatenate(parts, axis=1)
                valid = case["valid_prefix"]
                error = numeric_error(expected[:, :valid], actual[:, :valid], 3e-4, 3e-4)
                errors.append({"case": name, "chunks": chunks, **error})
                if chunks == [len(ids)]: saved[name] = actual
            last, _ = runtime.step(ids, last_only=True)
            errors.append({"case": name, "last_only": True, **numeric_error(expected[:, -1:], last, 3e-4, 3e-4)})
        for name in ("right_pad", "changed_future"):
            invariants.append({"case": name, **numeric_error(saved["short"], saved[name][:, :16], 3e-4, 3e-4)})
        # Immutable shared snapshots, stale suffix overwrite, and A/B/A branch
        # order catch errors that fresh-prefill comparisons cannot detect.
        prefix = arrays["short_input_ids"][0][:7]
        _, state = runtime.step(prefix, last_only=True)
        original = tuple(a.copy() for a in state)
        for suffix in ([9], [47, 81, 102], [9]):
            actual, _ = runtime.step(suffix, state, len(prefix))
            expected = (old.model(torch.tensor([[*prefix.tolist(), *suffix]])).numpy() if old else
                        v2.BundleLM(bundle).model(torch.tensor([[*prefix.tolist(), *suffix]])).detach().numpy())[:, len(prefix):]
            errors.append({"branch": suffix, **numeric_error(expected, actual, 3e-4, 3e-4)})
        immutable = all(np.array_equal(a, b) for a, b in zip(state, original))
        dirty = tuple(a.copy() for a in state)
        for a in dirty: a[:, :, :, len(prefix):, :] = 999
        a, _ = runtime.step([9], state, len(prefix))
        b, _ = runtime.step([9], dirty, len(prefix))
        invariants.append({"case": "unused_tail", **numeric_error(a, b, 0, 0)})
    report = {"format": "vimeml_kv_alignment_v1", "passed": immutable and all(e["outside_tolerance"] == 0 for e in errors+invariants),
        "coreml_manifest_sha256": runtime.metadata["coreml_manifest_sha256"], "model": runtime.metadata,
        "comparison": "same-weight stateless Core ML" if old else "frozen Windows FP32",
        "baseline_manifest_sha256": old.metadata["coreml_manifest_sha256"] if old else None,
        "tolerances": {"atol": 3e-4, "rtol": 3e-4}, "immutable_branch_inputs": immutable,
        "errors": errors, "invariants": invariants}
    fresh_directory(output); write_json(output / "alignment.json", report)
    return report


class CachedForward:
    """Evaluation adapter: share exact batch token prefixes within one call."""
    def __init__(self, runtime):
        self.runtime, self.config = runtime, runtime.config

    def __call__(self, inputs):
        ids = inputs.numpy().astype(np.int32)
        common = common_prefix_length(ids.tolist())
        if common:
            prefix, state = self.runtime.step(ids[0, :common])
        result = []
        for row in ids:
            if common == len(row): logits = prefix
            elif common:
                suffix, _ = self.runtime.step(row[common:], state, common)
                logits = np.concatenate((prefix, suffix), axis=1)
            else: logits, _ = self.runtime.step(row)
            result.append(logits)
        return torch.from_numpy(np.concatenate(result, axis=0))


class CachedLM(v2.BundleLM):
    def __init__(self, bundle, directory):
        super().__init__(bundle)
        self.runtime = kv.Runtime(bundle, directory)
        self.model, self.metadata = CachedForward(self.runtime), self.runtime.metadata
        self._generating = False
        self._ids = self._state = None

    @torch.inference_mode()
    def next_logits(self, ids):
        if self._generating and self._ids and ids[:len(self._ids)] == self._ids and len(ids) > len(self._ids):
            logits, state = self.runtime.step(ids[len(self._ids):], self._state, len(self._ids), last_only=True)
        else:
            logits, state = self.runtime.step(ids, last_only=True)
        if self._generating: self._ids, self._state = list(ids), state
        return torch.from_numpy(logits[0,-1])

    def generate(self, *args, **kwargs):
        self._generating, self._ids, self._state = True, None, None
        try: return super().generate(*args, **kwargs)
        finally: self._generating, self._ids, self._state = False, None, None


def timing(work, repeats=20):
    for _ in range(3): work()
    times = []
    for _ in range(repeats):
        start = time.perf_counter(); work(); times.append((time.perf_counter()-start)*1000)
    return {"n": len(times), "p50_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)), "max_ms": max(times)}


def benchmark(bundle, directory, baseline, output):
    runtime, old = kv.Runtime(bundle, directory), v2.CoreMLLM(bundle, baseline)
    rng = np.random.default_rng(716)
    rows = []
    for length in (8, 32, 64, 120):
        ids = rng.integers(4, 16384, length).tolist(); ids[0] = 2
        _, state = runtime.step(ids[:-1], last_only=True)
        rows.append({"prefix_tokens": length-1,
            "stateless": timing(lambda: old.model(torch.tensor([ids]))),
            "cached_decode": timing(lambda: runtime.step(ids[-1:], state, length-1, last_only=True)),
            "prefill_last_only": timing(lambda: runtime.step(ids, last_only=True))})
    def generated(cached, ids, tokens=8):
        sequence = list(ids)
        if cached: logits, state = runtime.step(sequence, last_only=True)
        else: logits = old.model(torch.tensor([sequence])).numpy()[:, -1:]
        for i in range(tokens):
            token = int(logits[0,-1].argmax()); sequence.append(token)
            if i == tokens-1: break
            if cached: logits, state = runtime.step([token], state, len(sequence)-1, last_only=True)
            else: logits = old.model(torch.tensor([sequence])).numpy()[:, -1:]
        return sequence
    generation = []
    for length in (8, 32, 64):
        ids = [2, *rng.integers(4, 16384, length-1).tolist()]
        generation.append({"prefix_tokens": length, "new_tokens": 8,
            "tokens_equal": generated(True,ids) == generated(False,ids),
            "cached_total": timing(lambda: generated(True,ids), 10),
            "stateless_total": timing(lambda: generated(False,ids), 10)})
    fresh_directory(output)
    write_json(output / "benchmark.json", {"scope": "Mac CPU_ONLY Python/Core ML; includes Python and array-validation overhead, not iPhone or keyboard extension",
        "model": runtime.metadata, "decode": rows, "generation": generation,
        "cache_bytes_per_snapshot": runtime.manifest["cache_bytes_per_snapshot"]})


def package(bundle, source, compiled, comparison, output):
    m = kv.verify_package(source)
    gate = read_json(comparison)
    if m["kind"] != "linear8_fp32_compute" or not (compiled / "coremldata.bin").is_file():
        raise ValueError("Expected compiled INT8 KV resources.")
    if (gate.get("format") != "vimeml_kv_alignment_v1" or not gate.get("passed")
            or gate.get("comparison") != "same-weight stateless Core ML"
            or gate["coreml_manifest_sha256"] != file_sha(source / "manifest.json")):
        raise ValueError("Passed same-weight stateless comparison required before packaging.")
    fresh_directory(output)
    resources = output / "Resources"; resources.mkdir()
    shutil.copytree(compiled, resources / "TinyJapaneseV21KVINT8.mlmodelc")
    # Reuses the exact V2 tokenizer already shipped with the baseline, not a
    # second tokenizer copy. Installer checks this existing identity.
    write_json(resources / "VimeLMManifestV21KV.json", {"format": "vime_ios_lm_v2_kv_v1",
        "architecture": v2.ARCHITECTURE, "model_version": "2.1-extend5-step40000-int8-b32-kv-v1",
        "minimum_ios": 18, "vocab_size": 16384, "context_length": 128, "compute_units": "CPU_ONLY",
        "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3},
        "tokenizer_sha256": m["bundle"]["tokenizer_sha256"], "checkpoint_sha256": m["bundle"]["checkpoint_sha256"],
        "cache_shape": m["cache_shape"], "cache_dtype": "float32",
        "source_manifest_sha256": file_sha(source / "manifest.json"),
        "stateless_alignment_sha256": file_sha(comparison),
        "compiled_files_sha256": {name: item["sha256"] for name,item in v2.inventory(compiled,hashes=True).items()},
        "policy": "Experimental opt-in; explicit FP32 request-local immutable caches; baseline remains default."})
    write_json(output / "manifest.json", {"format": "vimeml_kv_client_resources_v1", "model": m,
        "files": v2.inventory(output)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=4)
    stages = p.add_subparsers(dest="stage", required=True)
    for name in ("convert", "align", "compress", "evaluate", "samples", "benchmark", "package"):
        s = stages.add_parser(name); s.add_argument("--output", type=Path, required=True)
        if name != "compress": s.add_argument("--bundle", type=Path, required=True)
        if name in ("align", "evaluate", "samples", "benchmark"): s.add_argument("--model", type=Path, required=True)
        if name in ("compress", "package"): s.add_argument("--source", type=Path, required=True)
        if name == "compress": s.add_argument("--alignment", type=Path, required=True)
        if name == "align":
            s.add_argument("--reference", type=Path, required=True); s.add_argument("--baseline", type=Path)
        if name in ("evaluate", "samples", "benchmark"): s.add_argument("--baseline", type=Path, required=True)
        if name == "evaluate": s.add_argument("--benchmark", type=Path, required=True)
        if name == "package":
            s.add_argument("--compiled", type=Path, required=True)
            s.add_argument("--comparison", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.threads < 1: p.error("Fresh output and positive thread count required.")
    torch.set_num_threads(args.threads)
    if args.stage == "convert": kv.convert(args.bundle,args.output)
    elif args.stage == "compress": kv.compress(args.source,args.output,args.alignment)
    elif args.stage == "align":
        report = align(args.bundle,args.model,args.reference,args.output,args.baseline)
        print(json.dumps({"passed": report["passed"], "max_abs": max(e["max_abs"] for e in report["errors"])}, indent=2))
        if not report["passed"]: raise SystemExit("Alignment failed; evidence preserved.")
    elif args.stage == "benchmark": benchmark(args.bundle,args.model,args.baseline,args.output)
    elif args.stage == "package": package(args.bundle,args.source,args.compiled,args.comparison,args.output)
    else:
        lm = CachedLM(args.bundle,args.model)
        if args.stage == "evaluate": v2.evaluate(lm,args.benchmark,args.baseline,args.output)
        else: v2.samples(lm,args.baseline,args.output)
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__": main()
