# V4.2 Core ML and quantization handoff

This is the task brief for the macOS session. It covers converting V4.2 epoch 20 (`best-lm.pt`, step 1,081,942) to Core ML, quantizing it with the same recipe as the deployed V2.1, and measuring it against V2.1. It does not authorize replacing the deployed model, retraining, or changing benchmark inputs. [Cross-platform exchange](artifact-exchange.md) describes the general Git/artifact boundary.

## Setup

```bash
unzip vimeml-v42-mac-<date>.zip -d vimeml-v42-mac
python3 vimeml-v42-mac/setup_mac.py            # default ~/Documents/Sources/VimeML-v42-coreml
cd ~/Documents/Sources/VimeML-v42-coreml
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt && python -m pip install -e .
```

`setup_mac.py` clones `repository.bundle`, checks the recorded commit, sets `origin`, creates `claude/mac-v42-coreml`, and restores the payload into ignored `artifacts/` and `outputs/` after size and SHA256 checks. The macOS pins are Python 3.11, PyTorch 2.7.1, coremltools 9.0, NumPy 2.3.5 and scikit-learn 1.5.1. Record the actual versions, the macOS build and the hardware.

## Payload

| Path | Contents |
| --- | --- |
| `artifacts/models/tiny-ja-v42-e8k-d384-l7/deployment.pt` | V4.2 epoch 20 FP32 tensors; no optimizer or RNG state; source checkpoint SHA256 recorded inside |
| `artifacts/tokenizers/v4-a/ja-unigram-8192/`, `artifacts/token-data/corpus-v4-8k/manifest.json` | Matching 8K tokenizer and the identity manifest that `JapaneseLMV4` checks |
| `outputs/deployment/tiny-ja-v42-epoch20-fp32-input/` | PyTorch CPU FP32 logits for `bos`, `short`, `long`, `right_pad` and `changed_future`, plus `long_kv_step_logits` from single-token KV steps |
| `outputs/evaluation-v4-20261015/inputs/` | Four frozen candidate pools; hashes pinned in `vimeml.posttraining.infer.TRACKS` |
| `outputs/evaluation-v42-20261020/` | V4.2 FP32 CUDA scores, metrics and paired comparisons: the task-level reference |
| `outputs/evaluation-v4-20261015/v21/` | V2.1 per-case scores on the same four pools |
| `outputs/training-v42-20261020/` | Download receipt and training records, for identity only |
| `artifacts/deployment/tiny-ja-v2.1-extend5-{bundle,fp32,int8-b32}-v1/` | Deployed V2.1 bundle and Core ML packages, the baseline for same-harness measurements |

Blind and test data, training corpora, token stores, optimizer state and credentials are not included.

## Reference numbers

V4.2 FP32 Top-1 on the frozen pools: Standard development **523/728**, Legacy regression **370/500**, AJIMEE **139/200**, Historical development **123/137**. V2.1 FP32 on the same pools: 505, 374, 144, 122. Scoring uses joint context+candidate tokenization, the full-vocabulary suffix logP sum, no final EOS, no truncation and stable ties (`vimeml.benchmarks.evaluate_ajimee.rerank`).

## Tasks

1. **Mac PyTorch baseline.** Load `deployment.pt` with `JapaneseLMV4` on CPU FP32. Reproduce the fixture logits within atol/rtol 3e-4, and the four Top-1 counts. Report any per-case order difference from the CUDA scores, not just the totals.
2. **FP32 Core ML conversion.** V4 has no converter yet. Adapt the V2.1 path (`vimeml.deployment.v2.export_bundle/convert`, `scripts/deployment/coreml_v2.py`) for the fixed V4 contract: `int32[1,T]` with 1 ≤ T ≤ 128, raw `float32[1,T,8192]` logits, iOS 18, `compute_precision=FLOAT32`, `CPU_ONLY`. Also build the single-token KV graph from `vimeml.deployment.v4.KVStepGraphV4`, following `vimeml.deployment.kv_cache` and `scripts/deployment/coreml_kv.py`; it has cache shape `(7,1,6,128,64)`. Keep the V1/V2 formats and code paths unchanged.
3. **FP32 alignment gates.** Stateless: all fixture cases within 3e-4, plus the `right_pad`/`changed_future` causality invariants. KV: one-shot, one-token and split-chunk prefill against `long_kv_step_logits` and the full logits, plus snapshot/rollback checks equivalent to the V2.1 KV checks. Then run the four task tracks; Top-1 must match the PyTorch baseline.
4. **INT8 quantization.** Use the V2.1 recipe: `linear_symmetric` int8, `per_block` with block size 32, `weight_threshold=2048`, FP32 compute, `CPU_ONLY`. Strict logits alignment is expected to fail, as it did for V2.1 (max error about 1.0). Report the error distribution instead of loosening the gate. The task-level check is Top-1 on all four tracks against FP32 Core ML, with paired improvements and regressions. Report every regressed case ID.
5. **Measurements.** For V4.2 FP32 and INT8 and for V2.1 FP32 and INT8 under the same harness, record package size, compiled `.mlmodelc` size, load time, process memory after load and at peak, and stateless and KV per-call latency (p50/p95, warm and cold). Use the same machine and workload, and state the thermal and power conditions. Mac timings do not substitute for iPhone measurements.
6. **Optional, only if steps 1–5 pass.** Prepare iOS client resources in the style of `package_v2_ios.py`, with a fixtures JSON for the Vime client. Device runs in the separate Vime repository are a later step.

## Rules

- Do not read blind or test data. Do not retrain, select another checkpoint, change candidate pools or references, or tune any scoring setting from these results.
- Write each run to a new output directory. Never overwrite existing records or the V2.1 release assets.
- Record failures, tolerance misses and skipped steps as they are; a failed gate is a result, not something to work around.
- Commit source changes on `claude/mac-v42-coreml`. Write a dated report under `docs/reports/mac-<date>/` and add it to the index.
- Return large assets as a results ZIP in the style of `package_mac_results.py`, with paths, sizes, environment and pass/fail checks. Source changes go through Git and a pull request.
