# vimeml-tiny-zh-v1 Core ML, quantization and device handoff

This is the task brief for the macOS session. It covers converting `vimeml-tiny-zh-v1`, the first Chinese pinyin model, to Core ML, quantizing it with the deployed V4.2 recipe, and measuring it on the Mac and on the iPhone 16 Pro Max test host. It does not authorize shipping the model in Vime, retraining, choosing another checkpoint, or changing benchmark inputs. The V4.2 and V4.3 runs are the templates: [V4.2 brief](mac-v42-coreml.md), [V4.3 brief](mac-v43-coreml.md), [V4.2 Core ML report](../reports/mac-20261020/v42-coreml.md). Background on the model: [blind confirmation and baselines (Chinese)](../zh-CN/reports/zh-v1-20261111/blind-and-baselines.md).

## Model

| Item | Value |
| --- | --- |
| Release name | `vimeml-tiny-zh-v1` (internal codename `zh13`, arm `12k-fresh-s1`) |
| Source checkpoint | `artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/final.pt`, step 844,730, SHA256 `44c423a1…7d4` (not in the package) |
| Packaged weights | `deployment.pt` in the same directory: FP32 tensors only, source SHA256 recorded inside |
| Tokenizer | `artifacts/tokenizers/zh-v13/zh-unigram-12288/tokenizer.model`, SentencePiece Unigram, SHA256 `68bd37c4…7e5a`; pad 0, unk 1, bos 2, eos 3 |
| Shape | vocabulary 12,288, d384, **nine** layers, six heads, SwiGLU 960, RMSNorm, context 128, learned positions, tied head, no bias |
| Parameters | 20,036,736 |
| Format / class | `vimeml_tiny_gpt_zh13`, `vimeml.training.model_zh13.TinyGPTZh13` (a `TinyGPTV2` subclass with the V4 block family); load with `vimeml.benchmarks.zh13_lm_scores.load` |
| KV cache | `(9, 1, 6, 128, 64)` float32 per key and value |

The block is the same as V4.2's, but the vocabulary is 12,288 instead of 8,192 and there are nine layers instead of seven. Expect roughly 1.33× V4.2's size: FP32 about 80 MB, INT8 block-32 about 23 MB (V4.2 INT8 is 17.16 MB). Latency and memory should rise by a similar factor. Being larger than V4.2 is expected; whether the size is acceptable is a question for the report, not a reason to change the recipe.

## Setup

```bash
unzip vimeml-zh-v1-mac-<date>.zip -d vimeml-zh-v1-mac
python3 vimeml-zh-v1-mac/setup_mac.py            # default ~/Documents/Sources/VimeML-zh-v1-coreml
cd ~/Documents/Sources/VimeML-zh-v1-coreml
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt && python -m pip install -e .
```

`setup_mac.py` clones `repository.bundle` (the pushed `main`), checks the recorded commit, sets `origin`, stays on `main`, and restores the payload after size and SHA256 checks. Use the V4.2 pins (Python 3.11, PyTorch 2.7.1, coremltools 9.0, NumPy 2.3.5, scikit-learn 1.5.1) and record the actual versions, macOS build and hardware.

## Payload

| Path | Contents |
| --- | --- |
| `artifacts/models/tiny-zh13-12k-fresh-s1-d384-l9/` | `deployment.pt`, plus `config.json`, `summary.json`, `environment.json` from training (identity only) |
| `artifacts/tokenizers/zh-v13/zh-unigram-12288/` | `tokenizer.model`, `tokenizer.vocab` |
| `outputs/deployment/vimeml-tiny-zh-v1-fp32-input/` | Windows PyTorch CPU FP32 fixtures: `bos`, `short`, `long`, `right_pad`, `changed_future` logits and `long_kv_step_logits` |
| `outputs/benchmarks/ime-standard-zh-v1/frozen-r1/` | **Development split only** (1,456 pinyin items, 990 abbreviated): items, frozen 20-candidate pools, Rime outputs, overlap list |
| `outputs/benchmarks/zh13-20m/runs/12k-fresh-s1/` | v1 reference per-candidate scores (`full.jsonl`, `abbrev.jsonl`) with meta |
| `outputs/benchmarks/zh13-20m/judgement/scores/{12k-fresh-s1,rime}-{full,abbrev}/` | Scored results (`items.jsonl`, `summary.json`) for v1 and the Rime pool order |

Blind items, training data, the full checkpoint with optimizer state, and credentials are not included. V4.2 and V2.1 packages are not included either; reuse the accepted ones from the existing V4.3 checkout (`~/Documents/Sources/VimeML-v43-coreml/artifacts/deployment/`) by copying, never by rebuilding.

## Reference numbers (development, exact Top-1)

| System | Pinyin / 1,456 | Abbreviated / 990 |
| --- | ---: | ---: |
| v1, FP32 (Windows CUDA reference scores) | 1,363 (93.6%) | 337 (34.0%) |
| Rime pool order (librime 1.17.0 + rime-ice) | 1,001 (68.8%) | 142 (14.3%) |

Scoring: BOS + left context + candidate tokenized jointly; the score is the sum of FP32 full-vocabulary log-probabilities from the first token where any candidate differs from the context alone; no EOS; ties keep pool order. Implementation: `vimeml.benchmarks.zh13_lm_scores.score_item`. Score files feed `scripts/benchmarks/score_standard_ime_zh.py score … --scores`, and `… compare` pairs two results.

## Tasks

1. **Mac PyTorch baseline.**
   - Load `deployment.pt` and check the fixtures stay within 3e-4 (stateless and `forward_cached` steps).
   - Rescore development on the Mac CPU with `scripts/benchmarks/score_zh13_lm.py` on both tracks, then run `score_standard_ime_zh.py score`.
   - Top-1 must be 1,363 and 337. `compare` against `judgement/scores/12k-fresh-s1-*` should show no gains or losses; report any rank differences by item id.
2. **zh deployment code.** The V4 Core ML code hard-codes the Japanese contract: `v4_coreml.VOCAB = 8192`, `JapaneseLMV4` and its token-data checks, `TinyGPTV4`, and a seven-layer cache in `v4_coreml`, `v4_chunk`, `package_v4_ios.py` and `measure_v42.py`.
   - Generalize them so the vocabulary, layer count and loader come from the bundle, or add a thin zh module beside them.
   - Do not change V4.2 outputs: the existing V4 tests and the V4.2 package manifests must stay valid.
   - Packages are named `tiny-zh-v1-<kind>-v1` under `artifacts/deployment/`, and reports go under `outputs/deployment/zh-v1-*`.
3. **Bundle and FP32 Core ML.** Build the inference bundle, the stateless FP32 package (int32 `[1,T]`, 1≤T≤128 → float32 `[1,T,12288]`), and the chunk-KV FP32 package (cache `(9,1,6,128,64)`).
   - Run stateless and chunk-KV alignment against the fixtures at 3e-4.
   - Evaluate development on both tracks through Core ML with the same scoring. Top-1 must equal the Mac PyTorch baseline.
4. **INT8.** Use the unchanged recipe: `linear_symmetric` int8, `per_block` 32, `weight_threshold` 2048, FP32 compute, CPU_ONLY, iOS 18, for both stateless and chunk-KV.
   - Strict logits alignment is expected to fail. Report the error distribution (max abs, quantiles) next to V4.2's.
   - Check chunk-KV INT8 against stateless INT8 of the same weights.
   - Evaluate both development tracks against FP32 Core ML and list every regressed or improved item id.
5. **Quantization survey (exploratory).** Compare package size and development Top-1 on both tracks for:
   - 4-bit `linear_symmetric` per-block 32;
   - 6-bit and 4-bit k-means palettization (per grouped channel if coremltools 9 supports it, otherwise per tensor);
   - INT8 with the tied embedding/head kept in FP16.

   Same compute settings as task 4, stateless only. This is a size/quality map for a later decision. Do not pick a winner and do not change the task-4 recipe from these results.
6. **Mac measurements.** Extend `measure_v42.py` with a `zh-v1` model set.
   - Interleave v1 FP32/INT8 (stateless and chunk-KV) with V4.2 epoch 20 FP32/INT8 in fresh processes, in the same session.
   - Report package size, load time, first predict, T=1/8/32/64/128 p50 and p95, and memory.
   - Record thermal and power conditions as for V4.2.
7. **iOS resources and iPhone test host.**
   - Package the INT8 stateless and INT8 chunk-KV models with the tokenizer into resources named `TinyChineseV1INT8.mlmodelc`, `VimeLMManifestZhV1.json` and so on, so Japanese resources are unaffected.
   - In the Vime worktree used for V4.2/V4.3, add an opt-in `.zhV1` variant for the model functional tests and `VimeLanguageModelBenchmarkTests`.
   - Keep every production call site unchanged; Chinese input and Rime integration in the client are out of scope.
   - Run on the simulator and on the iPhone 16 Pro Max, alternating with V4.2 over four rounds, one variant per process.
   - Report load and reload time, `predict` T=1/8/32/64/128 p50 and p95, and footprint before load, after load and at peak, with thermal state and Low Power Mode at the start and end.

## Rules

- Only development data is in the package; do not try to obtain blind. Do not retrain, select other checkpoints, change candidate pools or references, or tune scoring from these results.
- Write each run to a new output directory. Never overwrite V4.2, V4.3 or V2.1 packages or records.
- Record failures, tolerance misses and skipped steps as they are.
- Do not quote benchmark text (CrossWOZ, RiSAWOZ, KdConv turns, targets or candidates) in committed docs; item ids and aggregates only.
- Commit source changes directly on `main` and push to `origin/main`; no feature branch or pull request. No Claude co-author trailers.
- Write a dated report under `docs/reports/mac-<date>/zh-v1-coreml.md` and add it to the index.
- Return large assets (packages, measurement records, regressed-id lists) as a results ZIP in the style of `package_v42_mac_results.py`, with paths, sizes, SHA256, environment and pass/fail checks.
