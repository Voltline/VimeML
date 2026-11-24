# vimeml-zh-v2-20m colloq-c40: Core ML build (Mac) and swap-in (Vime)

This is a self-contained task for the Mac session and for Vime. `colloq-c40` replaced `kd-cont` as `vimeml-zh-v2-20m` on 2026-11-23, after its [test confirmation](../reports/zh-colloq-confirmation-20261123/report.md) passed. On the colloquial test, self-rescored Top-1 rose from 39.0 to 42.3; on the public pinyin test it rose from 49.1 to 51.4; the Standard blind net against v1 was +7.

`colloq-c40` is `kd-cont` trained for 30,000 more updates with colloquial text ([colloquial post-training plan](../training/training-zh-colloq-plan.md)). Its shape, tokenizer, prompt format and decoding contract are exactly `kd-cont`'s. So this task is the [kd-cont brief](mac-zh-v2-20m.md) run again with new weights, under the gates as finally judged there (its 2026-11-20 deviation). Nothing in the Core ML recipe, the search or the Vime contract changes.

## Model

| Item | Value |
| --- | --- |
| Release name | `vimeml-zh-v2-20m` (arm `colloq-c40`) |
| Source checkpoint | `artifacts/models/tiny-zh13-12k-pinyin-colloq-c40/final.pt`, step 30,000, SHA256 `0ec5ffb1…1027` (not in the package) |
| Bundle | `artifacts/releases/vimeml-zh-v2-20m-c40/inference/` (`weights.pt`, `config.json`, `tokenizer.model`, `manifest.json`; the manifest's `checkpoint_sha256` is the one above) |
| Tokenizer | v1's `zh-unigram-12288`, SHA256 `68bd37c4…7e5a`; pad 0, unk 1, bos 2, eos 3; task ids CTX 4, READ 5, OUT 6 |
| Shape | as v1 and `kd-cont`: vocabulary 12,288, d384, nine layers, six heads, SwiGLU 960, context 128, 20,036,736 parameters |
| FP32 fixtures | `outputs/deployment/vimeml-zh-v2-20m-c40-fp32-input/` (same cases as v1's) |
| Generation fixture | `outputs/vime-fixtures/zh-generation-v2-20m-c40-task-r1/` (the same 125 cases as `kd-cont`'s fixture, regenerated with `colloq-c40`; `prompt_ids` are identical to `kd-cont`'s, and the shown top 1 differs from `kd-cont`'s in 23 cases) |

The handoff ZIP `zh-v2-20m-c40-handoff-20261123.zip` (105 MB, SHA256 `40225249…92ea`) holds these files at their repository paths, plus the reading dictionary, with a `SHA256SUMS` file. The `kd-cont` files keep their own paths, so both can sit in one checkout. Code comes from `main`, and `scripts/zh/deployment/prepare_zh_v2_20m.py --model colloq-c40` rebuilds every file in the ZIP.

## Mac tasks

Work in the existing kvbatch checkout:

1. `git pull` on `main`.
2. Unzip the handoff at the repository root and run `shasum -a 256 -c SHA256SUMS`.
3. Use the v4 environment, and let `B=artifacts/releases/vimeml-zh-v2-20m-c40/inference`.

Package names below use the `-c40` suffix, so they do not overwrite `kd-cont`'s.

1. **Stateless packages** (`scripts/zh/deployment/coreml_zh_v1.py`):
   - `convert --bundle $B --output artifacts/deployment/tiny-zh-v2-20m-c40-fp32-v1`.
   - `align --bundle $B --model <fp32 package> --reference-fixtures outputs/deployment/vimeml-zh-v2-20m-c40-fp32-input --output outputs/deployment/zh-v2-20m-c40-align-fp32`. This must pass at 3e-4.
   - `compress --source <fp32 package> --alignment outputs/deployment/zh-v2-20m-c40-align-fp32/alignment.json --output artifacts/deployment/tiny-zh-v2-20m-c40-int8-b32-v1`, with the deployed recipe `int8-b32` unchanged.
2. **Batched KV v4** (`scripts/zh/deployment/coreml_zh_kvbatch4.py`, every stage with `--bundle $B`):
   - `check`, then `build --variant fp32kv --stateless <c40 stateless INT8>`. Source alignment and INT8-vs-stateless alignment must pass as in the v4 report.
   - `search <c40 int8-merged> <out> --cases outputs/vime-fixtures/zh-generation-v2-20m-c40-task-r1/cases.jsonl --reference <c40 src-merged>`. The search gate is the one the kd-cont brief settled on:
     - the rescored (shown) top 1 equals the FP32 fixture in at least 123/125 cases;
     - text-set differences are allowed only at the tenth-place boundary, with scores within about 0.6.

     List every case that differs.
   - `fixture <c40 int8-merged> <out> --stateless <c40 stateless INT8>`, then `ios` with the alignment, search and fixture outputs.
   - The resource keeps `kd-cont`'s name, `TinyChineseV2KVBatchV4FP32KVINT8.mlmodelc`, because c40 replaces kd-cont under the same release.
   - `measure` with `kd-cont`'s v4 INT8 package and c40's, in the same session, with `--prompt 48` and `--prompt 80`. Latency should match, since the shape is the same. Report any difference above noise.
3. **Report** in the v4 format: sizes, alignment, search check, latency, and anything that failed. Do not change the recipe, the variant or the search to make a gate pass.

## Vime: swap kd-cont for colloq-c40

The contract is that of the [kd-cont brief](mac-zh-v2-20m.md), unchanged:

- task-mode prompt `[2 BOS, 4 CTX] + encode(context) + [5 READ] + encode(keys) + [6 OUT]`;
- beam 16 per key position, top 10, log P(EOS) added at the end;
- self rescoring `gen_score + 1.6 × lm_score`, with ties kept in generation order;
- `sentenceContext` limit 64.

The current product decision (2026-11-21) also stands:

- v2 is the default model for Rime reranking, prediction and typo correction;
- sentence generation stays behind its development switch;
- v1 ships as the fallback behind `vime.chineseModelVersion`.

1. **Replace the package.** Use c40's `TinyChineseV2KVBatchV4FP32KVINT8.mlmodelc` in place of `kd-cont`'s. No other code change is expected. If any is needed, report it.
2. **Fixture gate.** Run the existing fixture test against `zh-generation-v2-20m-c40-task-r1` for all 125 cases:
   - `prompt_ids` exact;
   - `rescored` top 1 equal to the fixture in at least 123/125 cases;
   - text-set differences only at the tenth-place boundary;
   - scores within about 0.6.

   List every case that differs.
3. **Device check** (iPhone 13 Pro, same build, `kd-cont` and c40 side by side, the method of the [kd-cont report](../reports/mac-20261121/zh-v2-20m.md)):
   - Rime rerank first-list p50/p95 and the share of keys within 60 ms;
   - load time and peak memory;
   - key-by-key and one-shot generation times.

   These should match `kd-cont`'s. Rerank must stay within 60 ms for every key, as before.
4. **Typo correction** keeps its current rule and the margin calibrated on v1, as with `kd-cont`.

## What the gain is, and where it reaches users

The confirmed gains were measured on **sentence generation** (self-rescored Top-1). In this release, generation stays behind its development switch. Users therefore see the change mostly through Rime reranking and prediction, which use the same model's LM scores.

Those uses were not measured separately for c40:

- the Standard blind rerank against v1 is net +7, not significant, against `kd-cont`'s −2;
- on development, final-particle prediction Top-1 rose from 0.190 (`kd-cont`) to 0.271, which is the 吧/把 case that started the colloquial work.

Once non-blocking generation ships, the generation gain applies directly.

## Mac result and deviation (2026-11-23)

**Integrity and alignment**

- The handoff's `SHA256SUMS` check passes, and the shape matches `kd-cont`.
- `int8-merged` is 23,271,109 bytes; the compiled `TinyChineseV2KVBatchV4FP32KVINT8.mlmodelc` is 22 MB, as for `kd-cont`.
- Stateless FP32 against PyTorch passes 5/5, max 3.89e-5 (gate 3e-4).
- KV v4 check: eager and traced, both 2.17e-5.
- KV v4 fp32kv alignment (logits, scored, score, prefix): max 2.57e-5, 0 rows outside tolerance.
- INT8 KV v4 against stateless INT8: step 1.62e-5, score sums 1.69e-5, same order 8/8.

**Search** (`zh-generation-v2-20m-c40-task-r1`, INT8 against the FP32 fixture)

- Generation-order top 1 is equal in 121/125 cases (`kd-cont`: 122/125). The common-score difference is at most 0.21. p50 is 65.9 ms (`kd-cont`: 65.9 ms).
- Shown order: the INT8 top 10 were rescored with c40's FP32 LM score at λ 1.6. The shown top 1 equals the fixture in **122/125** cases, one short of the 123 gate (`kd-cont`: 125/125). The final-score difference on common texts is at most 0.21.
- All three differences are long all-initials inputs. In two of them, INT8's beam path reaches a candidate that is not in the FP32 top 10 at all, and it wins after rescoring. In the third, it ranks the FP32 second place first, with an FP32 gap of 1.7.
- The text set equals the fixture in 89/125 cases. In 14 cases it differs above the tenth place. v1's shipped INT8 package had the same set in 86/125 against its own fixture.

**Latency** (Mac, `kd-cont` against c40, p50)

| prompt | step | `kd-cont` | c40 |
| --- | --- | ---: | ---: |
| 48 | `prompt_prefix` | 3.36 ms | 3.52 ms |
| 48 | `extend_prefix_1` | 1.26 ms | 1.25 ms |
| 48 | `scored_step_b16` | 3.74 ms | 3.46 ms |
| 48 | `score_b16` | 17.89 ms | 17.88 ms |
| 80 | `prompt_prefix` | 5.12 ms | 5.22 ms |
| 80 | `extend_prefix_1` | 1.43 ms | 1.41 ms |
| 80 | `scored_step_b16` | 4.08 ms | 3.83 ms |
| 80 | `score_b16` | 18.21 ms | 18.14 ms |

**Deviation: c40 goes to Vime although the search gate misses by one case.**

The miss comes from INT8 beam-path divergence on long all-initials inputs. It is not an export fault: every alignment passes at about 2e-5, and Rime reranking and prediction, the uses that are on by default, use `score`, which aligns at 1.7e-5. Sentence generation, the only use the search gate guards, stays behind its development switch.

The decision was made after seeing the result. The Vime fixture gate is judged the same way: shown top 1 of at least 122/125, with every differing case listed.
