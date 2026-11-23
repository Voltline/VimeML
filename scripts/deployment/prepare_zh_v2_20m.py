"""vimeml-zh-v2-20m handoff for the Mac (Core ML) and Vime (iOS): weights, fixtures and one ZIP.

Brief: docs/reference/mac-zh-v2-20m.md (kd-cont) and docs/reference/mac-zh-v2-20m-colloq.md (colloq-c40, which
replaced kd-cont as vimeml-zh-v2-20m on 2026-11-23). Both have v1's shape (12K, d384, nine layers, context
128), so the zh-v1 Core ML code is reused; only the weights and the decoding contract (task mode with
self rescoring) change.

  weights   deployment.pt (FP32 tensors, provenance) and PyTorch FP32 logits fixtures, as for v1
  bundle    inference bundle from deployment.pt (zh_v1.export_bundle)
  fixture   Vime generation fixture: task mode, beam 16 per key position, top 10, self rescoring at
            lambda 1.6; the 125 cases (keys, contexts) of the v1 fixture zh-generation-v1-lm-r2
  package   ZIP of bundle, FP32 fixtures, generation fixture and reading dictionary, with SHA256SUMS

usage: python -X utf8 scripts/deployment/prepare_zh_v2_20m.py <stage> [--model kd-cont|colloq-c40] [--output ZIP]
--model selects the checkpoint and its own output directories (kd-cont, the default, keeps the original paths).
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/deployment"))

RELEASE = "vimeml-zh-v2-20m"
MODEL_DIR = "artifacts/models/tiny-zh13-12k-pinyin-kd-cont"
CHECKPOINT = f"{MODEL_DIR}/final.pt"
CHECKPOINT_SHA256 = "412aee1d1c2d73c9f2ff52533188055e4655575472fe3be136e048fd240268a4"
DEPLOYMENT = f"{MODEL_DIR}/deployment.pt"
TOKENIZER_DIR = "artifacts/tokenizers/zh-v13/zh-unigram-12288"
REFERENCE = "outputs/deployment/vimeml-zh-v2-20m-fp32-input"
BUNDLE = f"artifacts/releases/{RELEASE}/inference"
READINGS = "outputs/zh-readings/dict/vime_pinyin.dict.yaml"
READINGS_SHA256 = "3a1654cd5ce20ecacd88046186c34ad87e748a494f7078f23c1ad15492eb0c8a"
V1_CASES = "outputs/vime-fixtures/zh-generation-v1-lm-r2/cases.jsonl"
FIXTURE = "outputs/vime-fixtures/zh-generation-v2-20m-task-r1"
MODE, BEAM, TOPK, LAMBDA = "task", 16, 10, 1.6
ARM = "pinyin-kd-cont"
PHASE = "v2-20m: task mode with self rescoring (replaces v1 lm mode on the phone)"
MODELS = {
    "kd-cont": {},
    "colloq-c40": {
        "MODEL_DIR": "artifacts/models/tiny-zh13-12k-pinyin-colloq-c40",
        "CHECKPOINT_SHA256": "0ec5ffb110fc3959534858fe1a931f4c177878daa8db33bea861c843ca1c1027",
        "REFERENCE": "outputs/deployment/vimeml-zh-v2-20m-c40-fp32-input",
        "BUNDLE": f"artifacts/releases/{RELEASE}-c40/inference",
        "FIXTURE": "outputs/vime-fixtures/zh-generation-v2-20m-c40-task-r1",
        "ARM": "pinyin-colloq-c40",
        "PHASE": "v2-20m colloq-c40 (replaces kd-cont): same task mode with self rescoring, same cases"},
}


def select(model):
    """Point the module constants at one model's checkpoint and output directories."""
    g = globals()
    g.update(MODELS[model])
    g["CHECKPOINT"] = f"{g['MODEL_DIR']}/final.pt"
    g["DEPLOYMENT"] = f"{g['MODEL_DIR']}/deployment.pt"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stage_weights():
    import prepare_mac_zh_v1 as P
    P.CHECKPOINT, P.CHECKPOINT_SHA256, P.DEPLOYMENT, P.REFERENCE = CHECKPOINT, CHECKPOINT_SHA256, DEPLOYMENT, REFERENCE
    P.RELEASE_NAME, P.ARM = RELEASE, ARM
    if (ROOT / DEPLOYMENT).exists() or (ROOT / REFERENCE).exists():
        raise SystemExit("deployment.pt or the FP32 fixtures exist already")
    print(json.dumps(P.prepare_weights(ROOT), ensure_ascii=False, indent=1))


def stage_bundle():
    from vimeml.deployment import zh_v1
    result = zh_v1.export_bundle(ROOT / DEPLOYMENT, ROOT / TOKENIZER_DIR, ROOT / BUNDLE)
    print(json.dumps({k: result[k] for k in ("release_name", "checkpoint_sha256", "checkpoint_step", "parameter_count")}))


def stage_fixture():
    import torch
    from vimeml.benchmarks import pinyin_gen_zh as pg
    from vimeml.deployment import zh_search as zs, zh_v1
    from vimeml.deployment.zh_torch_backend import ModelBatchBackend
    from vimeml.posttraining.zh_readings import ReadingTable

    out = ROOT / FIXTURE
    if out.exists():
        raise SystemExit(f"{FIXTURE} exists")
    if sha(ROOT / READINGS) != READINGS_SHA256:
        raise SystemExit("reading dictionary differs")
    torch.set_num_threads(4)
    lm = zh_v1.BundleLMZh(ROOT / BUNDLE)
    if lm.metadata["release_name"] != RELEASE or lm.metadata["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise SystemExit("bundle is not vimeml-zh-v2-20m")
    model, sp = lm.model.float().eval(), lm.processor
    backend = ModelBatchBackend(model, "cpu")
    index = zs.Index(sp, ReadingTable(str(ROOT / READINGS)))
    cases = []
    for line in open(ROOT / V1_CASES, encoding="utf-8"):
        c = json.loads(line)
        ranked, stats = zs.search(backend, sp, index, c["keys"], c["context"], MODE, BEAM, TOPK)
        texts = [t for t, _, _ in ranked]
        lm_scores = pg.prior_scores(model, sp, c["context"], texts, "cpu") if texts else None
        top = [{"text": t, "ids": ids, "gen_score": round(s, 5),
                "lm_score": None if lm_scores is None else round(lm_scores[i], 5),
                "final_score": None if lm_scores is None else round(s + LAMBDA * lm_scores[i], 5)}
               for i, (t, s, ids) in enumerate(ranked)]
        order = sorted(range(len(top)), key=lambda i: (-(top[i]["final_score"] if lm_scores else top[i]["gen_score"]), i))
        cases.append({"id": c["id"], "kind": c["kind"], "keys": c["keys"], "context": c["context"],
                      "prompt_ids": zs.prompt_ids(sp, c["keys"], c["context"], MODE),
                      "top": top, "rescored": [top[i]["text"] for i in order],
                      "decode_calls": stats.calls, "decode_rows": stats.rows})
    out.mkdir(parents=True)
    with open(out / "cases.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    kinds = {}
    for c in cases:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    write_json(out / "meta.json", {
        "format": "vime_zh_generation_fixture_v2",
        "phase": PHASE,
        "model": {"release": RELEASE, "checkpoint_sha256": CHECKPOINT_SHA256,
                  "bundle_manifest_sha256": lm.metadata["bundle_manifest_sha256"], "precision": "fp32 (PyTorch, CPU)"},
        "tokenizer_sha256": lm.metadata["tokenizer_sha256"],
        "readings_dict_sha256": READINGS_SHA256,
        "search": {
            "mode": MODE, "beam": BEAM, "topk": TOPK,
            "prompt": "[2 BOS, 4 CTX] + encode(context) + [5 READ] + encode(keys) + [6 OUT]; the keys are one "
                      "SentencePiece encode of the raw key string, so the prompt changes on every key",
            "beam_per": "key position, as v1 (zh-generation-v1-lm-r2): hypotheses compete only with others that "
                        "consumed the same number of keys; beam 16 kept per position after dedup by token ids",
            "finish": "keys consumed exactly, then + log P(EOS=3) after the last token (gen_score includes it)",
            "token_limit": "prompt + output < 128",
            "spellings": "each char: any full reading, or its initial (first letter, or zh/ch/sh)"},
        "rescoring": {
            "lambda": LAMBDA,
            "lm_score": "the same model in lm mode: BOS + context + candidate tokenized jointly, sum of log-probs "
                        "from the first token where the candidates differ from the context alone, no EOS; "
                        "= VimeLanguageModel.scores(context:candidates:) on the top texts, one joint call",
            "final_score": "gen_score + lambda * lm_score; shown order = final_score descending, ties keep generation order",
            "rescored": "the shown order"},
        "context_rule": "VimeLanguageModel.sentenceContext(left text, limit 64)",
        "source": "the 125 cases (keys, contexts) of zh-generation-v1-lm-r2: ime-pinyin-zh-v1-r1 development "
                  "(test never read) and five hand-written keys",
        "cases": len(cases), "kinds": kinds,
        "tolerance": "FP32 here; on-device INT8 may differ by up to ~0.4 in logits: expect the same text set, "
                     "scores within ~0.5, rare swaps of near-tied neighbours"})
    print(json.dumps({"cases": len(cases), "kinds": kinds}))


def stage_package(output):
    output = Path(output)
    if output.exists():
        raise SystemExit("output exists")
    files = [p for tree in (BUNDLE, REFERENCE, FIXTURE) for p in sorted((ROOT / tree).rglob("*")) if p.is_file()]
    files.append(ROOT / READINGS)
    files.append(ROOT / READINGS.replace("vime_pinyin.dict.yaml", "SOURCE.txt"))
    sums = "".join(f"{sha(p)}  {p.relative_to(ROOT).as_posix()}\n" for p in files)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr("SHA256SUMS", sums)
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:
            raise SystemExit("ZIP CRC check failed")
    print(json.dumps({"zip": str(output), "bytes": output.stat().st_size, "sha256": sha(output), "files": len(files) + 1}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["weights", "bundle", "fixture", "package"])
    parser.add_argument("--model", choices=sorted(MODELS), default="kd-cont")
    parser.add_argument("--output")
    args = parser.parse_args()
    select(args.model)
    if args.stage == "package":
        if not args.output:
            parser.error("package needs --output")
        stage_package(args.output)
    else:
        {"weights": stage_weights, "bundle": stage_bundle, "fixture": stage_fixture}[args.stage]()


if __name__ == "__main__":
    main()
