# V4.3 Core ML, quantization and device handoff

This is the task brief for the macOS session. It covers converting V4.3 epochs 30 and 27 to Core ML, quantizing them with the deployed V2.1 recipe, and measuring them against V4.2 epoch 20 and V2.1 on the Mac and on the iPhone 16 Pro Max test host. It does not authorize replacing the deployed model, retraining, or changing benchmark inputs. The V4.2 run is the template: [V4.2 brief](mac-v42-coreml.md), [V4.2 Core ML report](../reports/mac-20261020/v42-coreml.md).

## Model

V4.3 has the same contract as V4.2: 8K tokenizer (SHA256 `296a2bdf…81b5`), d384, seven layers, six heads, SwiGLU 960, context 128, tied head, 15,070,848 parameters, ordinary LM with no conversion markers. It was trained from random weights for 30 epochs on the V4.2 pools plus nine deduplicated FineWeb2-Edu shards (web roughly doubled per epoch).

| Variant | Checkpoint | Step | Selection |
| --- | --- | ---: | --- |
| `v43-epoch30` | `epoch-30.pt` | 3,225,888 | Best complete LM BPC (3.043831) |
| `v43-epoch27` | `epoch-27.pt` | 2,903,298 | Requested comparison epoch |
| `v42-epoch20` | V4.2 `best-lm.pt` | 1,081,942 | Baseline, packages already built |

Because the graphs and weight shapes are identical to V4.2, package size, load time, memory and latency should match V4.2 within measurement noise. A difference beyond noise is a finding to investigate, not an expected gain. Quality (Top-1, INT8 drift) is where the variants differ.

## Setup

```bash
unzip vimeml-v43-mac-<date>.zip -d vimeml-v43-mac
python3 vimeml-v43-mac/setup_mac.py            # default ~/Documents/Sources/VimeML-v43-coreml
cd ~/Documents/Sources/VimeML-v43-coreml
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt && python -m pip install -e .
```

`setup_mac.py` clones `repository.bundle`, checks the recorded commit, sets `origin`, creates `claude/mac-v43-coreml`, and restores the payload after size and SHA256 checks. Use the V4.2 pins (Python 3.11, PyTorch 2.7.1, coremltools 9.0, NumPy 2.3.5, scikit-learn 1.5.1) and record the actual versions, macOS build and hardware.

## Payload

| Path | Contents |
| --- | --- |
| `artifacts/models/tiny-ja-v43-e8k-d384-l7/epoch{30,27}-deployment.pt` | FP32 tensors only; source checkpoint SHA256 recorded inside |
| `outputs/deployment/tiny-ja-v43-epoch{30,27}-fp32-input/` | Windows PyTorch CPU FP32 fixtures (`bos`, `short`, `long`, `right_pad`, `changed_future`, `long_kv_step_logits`) |
| `outputs/evaluation-v43-20261020/epoch{30,27}/` | V4.3 FP32 CUDA scores and paired comparisons: the task-level reference |
| `outputs/evaluation-v42-20261020/`, `outputs/evaluation-v4-20261015/{inputs,v21}/` | V4.2 and V2.1 per-case scores; the four frozen pools |
| `artifacts/tokenizers/v4-a/ja-unigram-8192/`, `artifacts/token-data/corpus-v4-8k/manifest.json` | Matching tokenizer and identity manifest |
| `outputs/training-v43-20261020/` | Download receipt and training records, for identity only |
| `artifacts/deployment/tiny-ja-v4.2-epoch20-{bundle,fp32,int8-b32,kvchunk-fp32,kvchunk-int8-b32,client-resources-v1}-v1/` | Accepted V4.2 packages; do not rebuild |
| `artifacts/deployment/tiny-ja-v2.1-extend5-{bundle,fp32,int8-b32}-v1/`, `…-kv-{fp32,int8-b32}-mac-v42-measure-v1/` | V2.1 deployed packages and measurement-only KV packages |

Blind, test and fixed-reference data, training corpora, optimizer state and credentials are not included.

## Reference numbers (FP32 CUDA, frozen pools)

| Model | Standard dev /728 | Legacy /500 | AJIMEE /200 | Historical dev /137 |
| --- | ---: | ---: | ---: | ---: |
| V2.1 | 505 | 374 | 144 | 122 |
| V4.2 epoch 20 | 523 | 370 | 139 | 123 |
| V4.3 epoch 30 | 519 | 367 | 143 | 123 |
| V4.3 epoch 27 | 521 | 372 | 138 | 124 |

V4.2 Core ML INT8 gave 522/371/139/123.

## Tasks

All `coreml_v4.py` stages take `--variant v43-epoch30` or `--variant v43-epoch27`; the default remains V4.2. Package directories follow `tiny-ja-v4.3-epoch{30,27}-<kind>-v1`, and reports go under `outputs/deployment/v43-e{30,27}-*`. Run every step for both epochs.

1. **Mac PyTorch baseline.** Run `coreml_v4.py --variant … baseline`. Fixtures must be within 3e-4. Top-1 must equal the CUDA reference on all four tracks; report per-case order differences.
2. **Bundle and FP32 Core ML.** Run `export`, `convert` and `convert-kvchunk`. The single-token KV graph is superseded by the chunk graph and is skipped unless time allows.
3. **FP32 gates.** Run `align` (stateless) and `align-kvchunk` against the fixtures. Evaluate the four tracks with `evaluate` and `evaluate-kvchunk`. Top-1 must equal the PyTorch baseline.
4. **INT8.** Use the unchanged recipe via `compress` and `compress-kvchunk`: linear_symmetric int8, per_block 32, weight_threshold 2048, FP32 compute, CPU_ONLY, iOS 18. Strict logits alignment is expected to fail. Report the error distribution and compare it with V4.2 (max abs, quantiles). Check the kvchunk INT8 against stateless INT8 of the same weights. Evaluate the four tracks against FP32 Core ML and list every regressed case ID.
5. **Mac measurements.** Run `measure_v42.py run --set v43 --output outputs/deployment/v43-mac-measurements-v1`. It interleaves V4.3 e30 FP32/INT8, e27 INT8, V4.2 FP32/INT8 and V2.1 FP32/INT8 stateless packages, and the chunk-KV packages of each, in fresh processes, all in the same session. Record the thermal and power conditions as for V4.2. Report V4.3 against V4.2 epoch 20 first, then against V2.1.
6. **iOS resources.** Run `package_v4_ios.py --variant v43-epoch{30,27} --chunk` with the INT8 stateless and INT8 chunk-KV packages and the same Vime fixture template as V4.2. Resources are named `TinyJapaneseV43E30INT8.mlmodelc`, `VimeLMManifestV43E30.json` and so on, so V4.2 resources are unaffected.
7. **iPhone test host.** In the Vime worktree used for V4.2 (`claude/v42-lm-measure`, or a new branch from it), add opt-in variants for V4.3 e30 and e27 next to `.v42`. Keep `.v21` the default at every production call site. Run the model functional tests on the simulator and on the iPhone 16 Pro Max. Then run `VimeLanguageModelBenchmarkTests` with V2.1, V4.2 and both V4.3 variants alternating over four rounds, one variant per process (`Scripts/benchmark_lm_variants.py`). Record the thermal state and Low Power Mode at the start and end of each run. If the phone is not `nominal`, say so and repeat once it has cooled if possible. Report load and reload time, first predict, `predict` T=1/8/32/64/128 p50 and p95, rerank, beam and next-word p50 and p95, and footprint before load, after load and at peak. Keyboard-extension measurements remain out of scope unless they are cheap to add.

## Rules

- Do not read blind, test or fixed-reference data. Do not retrain, select other checkpoints, change candidate pools or references, or tune scoring from these results.
- Write each run to a new output directory. Never overwrite V4.2 or V2.1 packages or records.
- Record failures, tolerance misses and skipped steps as they are.
- Do not quote benchmark text (WRIME, JMultiWOZ, AJIMEE sentences) in committed docs; case IDs and aggregates only.
- Commit source changes on `claude/mac-v43-coreml`, without Claude co-author trailers. Write a dated report under `docs/reports/mac-<date>/v43-coreml.md` and add it to the index.
- Return large assets as a results ZIP in the style of `package_v42_mac_results.py`, with paths, sizes, environment and pass/fail checks. Source changes go through Git and a pull request.
