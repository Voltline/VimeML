# vimeml-ja-v2-20m: Core ML build (Mac) and device test (Vime)

> [Chinese version](../../zh-CN/ja/mac/mac-ja-v2-20m.md) (reference text)

A self-contained task for the Mac session and Vime: stage 5 of the [ja-gen plan](../training/training-ja-gen-plan.md). The stage 4 [test confirmation](../reports/ja-gen-confirmation-20261128/report.md) passed, so `colloq-c40` is released as `vimeml-ja-v2-20m` and replaces the deployed V4.2 (AzooKey N-best + V4.2 reranking). Kana test lenient Top-1 rose from 510 to 528 (p 0.0024); colloquial test from 509 to 537.

What was confirmed is a whole setup, not just a model:

- AzooKey N-best 20 merged with the model's generated top 10;
- every merged candidate scored twice by the same model and ordered by `gen_score + 1.6 × lm_score`.

This task reproduces that setup on the phone and measures its latency and memory.

The model block is the one of `vimeml-zh-v2-20m` (12K vocabulary, d384, nine layers, context 128), so the Core ML flow is the Chinese v2 one:

- the stateless INT8 package of [mac-zh-v1-coreml.md](../../zh/mac/mac-zh-v1-coreml.md);
- the batched KV v4 package (`fp32kv`, with `prefix`) of [mac-zh-kvbatch-v4.md](../../zh/mac/mac-zh-kvbatch-v4.md).

Three things differ: Japanese weights and tokenizer; search bucketed by hiragana reading position over a reading index; a display list that merges AzooKey candidates.

## Model

| Item | Value |
| --- | --- |
| Release | `vimeml-ja-v2-20m` (arm `20m-colloq-c40`) |
| Source checkpoint | `artifacts/models/ja-kana-20m-colloq-c40/final.pt`, SHA256 `19b04ddf…d15f` (not in the package) |
| Inference bundle | `artifacts/releases/vimeml-ja-v2-20m/inference/` (`weights.pt`, `config.json`, `tokenizer.model`, `manifest.json`) |
| Tokenizer | `ja-unigram-12288`, SHA256 `61d71460…aa53`; pad 0, unk 1, bos 2, eos 3; task ids CTX 4, READ 5, OUT 6; byte fallback |
| Architecture | vocabulary 12,288, d384, nine layers, six heads, SwiGLU 960, context 128 |
| Reading index | `outputs/ja-gen-stage2/kana-index.json`, SHA256 `b60c485d…de5b`: hiragana reading → allowed token ids |
| FP32 fixtures | `outputs/deployment/vimeml-ja-v2-20m-c40-fp32-input/` (logits and KV steps) |
| Generation fixture | `outputs/vime-fixtures/ja-generation-v2-20m-c40-task-r1/` (`cases.jsonl` + `meta.json`, 125 cases) |

The 125 generation fixture cases:

- 120 drawn from `ime-kana-ja-v1` development, stratified by left context × length band, each with its exported AzooKey N-best 20;
- 5 hand-written conversational sentences, generation only, with no AzooKey list;
- test was not read.

Each case records:

- the generated top 10 (`top`, with `beam_score`) and the generation-only display order `rescored`;
- the merged list `merged`: for every candidate `gen_score`, `lm_score`, `final_score` and all the token ids those scores use (`gen_prompt_ids`, `gen_output_ids`, `lm_ids`, `lm_from`);
- the display order `shown`.

`meta.json` states the search and merge contract.

The handoff ZIP holds these files and the reading table at their repository paths, with a `SHA256SUMS` file (name, size and SHA256 under "Handoff package" below). Code is from `main`; `scripts/ja/deployment/prepare_ja_v2_20m.py` rebuilds every file in the ZIP.

## Mac task

Work in the existing kvbatch checkout:

1. `git pull` on `main`.
2. Unzip the handoff package at the repository root and run `shasum -a 256 -c SHA256SUMS`.
3. Use the Chinese v4 environment and set `B=artifacts/releases/vimeml-ja-v2-20m/inference`.

All package names start with `tiny-ja-v2-20m-`, so nothing Chinese is overwritten.

1. **Stateless package** (`scripts/zh/deployment/coreml_zh_v1.py`):
   - `convert --bundle $B --output artifacts/deployment/tiny-ja-v2-20m-fp32-v1`.
   - `align --bundle $B --model artifacts/deployment/tiny-ja-v2-20m-fp32-v1 --reference-fixtures outputs/deployment/vimeml-ja-v2-20m-c40-fp32-input --output outputs/deployment/ja-v2-20m-align-fp32`. Must pass at 3e-4.
   - `compress --source artifacts/deployment/tiny-ja-v2-20m-fp32-v1 --alignment outputs/deployment/ja-v2-20m-align-fp32/alignment.json --output artifacts/deployment/tiny-ja-v2-20m-int8-b32-v1`, deployed recipe `int8-b32` unchanged.
   - Skip `evaluate`: it is the Chinese Standard evaluation.
2. **Batched KV v4** (`scripts/zh/deployment/coreml_zh_kvbatch4.py`, every stage with `--bundle $B`):
   - `check`, then `build --variant fp32kv --stateless artifacts/deployment/tiny-ja-v2-20m-int8-b32-v1` with output `artifacts/deployment/tiny-ja-v2-20m-kvbatch-v4`. Source alignment and INT8-vs-stateless alignment must pass as in the Chinese v4 report.
   - `fixture <int8-merged> <out> --stateless artifacts/deployment/tiny-ja-v2-20m-int8-b32-v1`. This is the generic KV plumbing fixture (`prefix` in one and two calls, `logits`, `scored`, `score`) on the Chinese v4 sentences. The Japanese tokenizer has byte fallback and round-trips all of them; it checks the KV plumbing, not Japanese quality.
3. **Japanese search check** (`scripts/ja/deployment/coreml_ja_v2_20m.py`, replacing the Chinese pinyin `search` stage):

   ```bash
   python scripts/ja/deployment/coreml_ja_v2_20m.py search <int8-merged> outputs/deployment/ja-v2-20m-search-int8 \
       --bundle $B --reference <src-merged>
   ```

   It makes three checks per case:
   - **generation**: bucketed constrained beam 16, top 10, with the package's `logits`, against the fixture's `top`.
   - **scores**: the fixture's merged list scored with the package's `score` from the fixture's own token ids, against `gen_score` / `lm_score`; the display order from those scores against `shown`.
   - **pipeline**: the package's own generated top 10 merged with the case's AzooKey list and scored by the package, as on the phone; display top 1 against `shown[0]`, generation-only top 1 against `rescored[0]`.

   Gates (INT8 against the FP32 fixture, judged as the Chinese v2 run finally judged them, fixed before any result):
   - scores: display top 1 equal to `shown[0]` in at least 123/125 cases; common-candidate scores within about 0.6.
   - pipeline: display top 1 equal to `shown[0]` in at least 123/125 cases.
   - generation is reported, not gated: users see the merged list, and near-tie swaps in generation order and tenth-place set differences were already seen in Chinese v2.

   List every differing case (`differing_cases` in `summary.json`). The same script was run locally on the PyTorch FP32 KV graphs (`--torch`): 125/125 on all three, largest score difference 1.9e-4 (see "Handoff package").
4. **iOS resource**: run `coreml_zh_kvbatch4.py ios` with the alignment, Japanese search and fixture outputs:

   ```bash
   python scripts/zh/deployment/coreml_zh_kvbatch4.py ios <int8-merged> <out> --bundle $B \
       --alignment <int8-merged-align.json> --search outputs/deployment/ja-v2-20m-search-int8 --fixture <fixture out>
   ```

   The resource is `TinyJapaneseV2KVBatchV4FP32KVINT8.mlmodelc` with manifest `VimeLMManifestJaKVBatchV4-fp32kv.json`, so it sits beside the Chinese resources.
5. **Latency**: in one session run `measure` on the Chinese `vimeml-zh-v2-20m` (c40) v4 INT8 package and the Japanese one, `--prompt 48` and `--prompt 80`. Same block, so latency should match; report any difference beyond noise. Also report `p50_search_ms` / `p50_score_ms` from the Japanese search check (Mac Python, including input conversion).
6. **Report** in the v4 format: sizes, alignment, search check, latency and any failure. Do not change recipe, variant, search, λ or candidate counts to pass a gate.

## Vime: replace V4.2 with vimeml-ja-v2-20m

Today's Japanese flow is AzooKey N-best plus V4.2 reranking. The new flow uses one model for two jobs: generating candidates, and scoring AzooKey and generated candidates alike. V4.2 resources stay as the fallback.

1. **Resources**:
   - `TinyJapaneseV2KVBatchV4FP32KVINT8.mlmodelc` and its manifest;
   - the bundle's `tokenizer.model` under a resource name that does not clash with the V4.2 tokenizer;
   - `kana-index.json`, reading → token ids; it may be converted to a more compact format at build time, but the content must be identical.
2. **Inputs**:
   - `reading`: the uncommitted kana in hiragana (katakana to hiragana, long-vowel mark and punctuation kept);
   - `context`: left context, at most 64 characters, as Chinese `sentenceContext`.
3. **Generation** (`vimeml.deployment.ja_search.search_constrained`, `search` section of `meta.json`):
   - Prompt: `[2 BOS, 4 CTX] + encode(context[-64:]) + [5 READ] + encode(reading) + [6 OUT]`. Left-context tokens are dropped from the left so the prompt leaves at least 2 positions of 128.
   - The context part does not depend on the reading: build its KV once per context with `prefix`, then extend with `[READ] + encode(reading) + [OUT]` whenever the reading changes.
   - The search is bucketed by hiragana characters consumed. In each bucket:
     - only tokens the reading index allows (`reading[pos:end]` in the index, and the reading after `end` still coverable);
     - dedup by token ids, keep beam 16;
     - one batched decode.
   - Scores are log-probs normalized over the whole vocabulary; `scored` with the moves' tokens and EOS in `allowed` works.
   - When the reading is consumed, add log P(EOS = 3). Dedup by decoded text keeping the best score; keep the top 10.
   - Check `prompt_ids` exactly against the fixture.
4. **Merge and score** (`kana_gen_ja.merge_and_rank`, `merge` section of `meta.json`):
   - Candidate set: the first 20 candidates of AzooKey's whole-reading conversion in AzooKey order, then the generated top 10 in generation order; exact-string dedup.
   - `gen_score`: a task-mode score for every candidate, AzooKey's included. The prefix is the prompt (left context re-trimmed for that candidate's output length so prompt + output − 1 ≤ 128), the continuation `encode(text) + [3]`; build the prefix with `prefix` / `scored` and sum with `score` (as `zh_v1_batch3.score_continuations`).
   - `lm_score`: as Chinese `VimeLanguageModel.scores(context:candidates:)`: `BOS + encode(context + text)` in one joint call, summed after the common prefix with `BOS + encode(context)`.
   - Order: `gen_score + 1.6 × lm_score` descending, ties keep merge order; candidates with one score after those with both; none last.
   - The fixture gives each candidate's token ids and scores.
5. **When to show what** (plan stage 5, following the Chinese 2026-11-21 product decision): generation never blocks a keystroke.
   - On every keystroke, show AzooKey's top 20 reordered by the same scores first. This needs no generation, only two scores per candidate.
   - After a pause or in the background, finish generation, merge the generated candidates, reorder by the rule above and update the list.
   - If the user commits before generation finishes, use the reranked list of that moment.
6. **Fixture gate** (all 125 cases):
   - `prompt_ids` identical;
   - every `merged` candidate's `gen_prompt_ids`, `gen_output_ids`, `lm_ids`, `lm_from` identical;
   - scoring the fixture's merged list, display top 1 equal to `shown[0]` in at least 123/125 cases, scores within about 0.6;
   - with the phone's own generation plus the fixture's AzooKey list, display top 1 equal to `shown[0]` in at least 123/125 cases.

   List every differing case.
7. **Device test** (iPhone 13 Pro and iPhone 16 Pro Max, V4.2 and v2 side by side in one build, end to end per keystroke in the keyboard extension, method of the Chinese [v2 report](../../zh/reports/mac-20261121/zh-v2-20m.md)):
   - reranked list (first part of step 5): p50/p95 and the share of keystrokes within 60 ms;
   - full flow (generation + merge + scoring): p50/p95 per keystroke and once after a pause;
   - load time, peak memory and the keyboard extension's memory headroom;
   - the same figures for V4.2 reranking beside them.

   Release condition: on the 13 Pro the reranked list stays within 60 ms on every keystroke, as V4.2 does. The full flow is reported and runs in the background as in step 5.
8. **Fallback**: V4.2 resources stay behind one switch that changes model and flow together (named in the style of Chinese `vime.chineseModelVersion`), so a problem found after review needs no rebuild.

## Not in scope

- No retraining; λ, beam, top 10 and the AzooKey candidate count stay as they are.
- No test or blind reads. The fixture is development-only.
- No changes to recipe, variant or search to pass a gate. Report a failure as it is; VimeML records any deviation after seeing the result.

## Return

- **Mac**:
  - a report (Chinese);
  - `outputs/deployment/ja-v2-20m-align-fp32/`, the KV v4 `*-align.json`, `outputs/deployment/ja-v2-20m-search-int8/` (`summary.json`, `cases.jsonl`), the `measure` output;
  - the manifest from the `ios` output.

  Zip these together; do not commit them to VimeML.
- **Vime**:
  - fixture test output;
  - the device measurement table;
  - the commit of the change (in the Vime repository).

## Handoff package

`ja-v2-20m-c40-handoff-20261128.zip`, 97,917,499 bytes, SHA256 `05d6f57c…9325`. It holds 10 files and `SHA256SUMS`, built with the `weights`, `bundle`, `fixture` and `package` stages of `prepare_ja_v2_20m.py` (default `--model colloq-c40`) on Windows PyTorch CPU FP32.

- **Weights**: source checkpoint `19b04ddf…`, step 30,000, 20,036,736 parameters.
  - `deployment.pt` is `16e798c2…`, with no optimizer state.
  - The bundle manifest is `325c6aa8…`.
  - KV steps differ from stateless by at most 1.9e-5.
- **Generation fixture** (`cases.jsonl` `9cf4c102…`):
  - 125 cases, 0 generation fallbacks, 0 LM-scoring fallbacks.
  - Merged lists hold 10–30 candidates per case, median 18.
  - In 11 of the 120 development cases the displayed top 1 comes from generation and is not in the AzooKey list.
- **Check against the stage 3.3 result**: the 120 development cases against the judged stage 3.3 `colloq-c40` development rows:
  - display order identical in 120/120;
  - generated top 10 also identical in 120/120;
  - `gen_score` differs by at most 1.8e-4.
- **Search-check self-test**: `coreml_ja_v2_20m.py search --torch` on the PyTorch FP32 KV graphs:
  - generation, scores and pipeline all 125/125;
  - largest score difference 1.9e-4.
