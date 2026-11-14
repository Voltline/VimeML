# zh-200m Chinese desktop model: six 200M arms (preregistered)

Drafted 2026-11-12. This is the 200M training plan of the [desktop model roadmap](benchmarks/predict-zh-plan.md): the roadmap already registered the arm structure (three seeds of the default recipe, vocabularies 16K/24K/32K, context 128/512) and the decision rules; this page registers shapes, parameter budget, seed values and batch strategy. It counted as registered once committed; later changes are recorded as deviations at the end, and before any arm started training the decision rules could only be completed, not changed. Training is recorded in the [training report](reports/zh-200m-20261113/training.md) and the decision in the [judgement report](reports/zh-200m-20261114/judgement.md).

## Questions

1. **Scaling gain**: from 20M (vimeml-tiny-zh-v1) to 200M parameters, how much do prediction metrics improve, and is 800M worth training under the roadmap's gate?
2. **Vocabulary**: do 16K, 24K and 32K differ at the 200M budget? (The tokenizer report compared compression only; this is the set it left for 200M, see the [vocabulary comparison](reports/zh-v1-20261106/tokenizer-vocab.md).)
3. **Context**: does 512 differ from 128 on prediction? The desktop has no phone-like memory limit, and prediction relies on longer left context.

## Six arms

About 200M parameters each, same data and recipe, changing only vocabulary, context and seed:

| Arm | Vocabulary | Context | Layers | Parameters | Windows per update | Seed |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `16k-c512` | 16,384 | 512 | 15 | 198,212,608 | 128×512 | 766903225 |
| `24k-c512-s1` | 24,576 | 512 | 14 | 194,540,544 | 128×512 | 766903225 |
| `24k-c512-s2` | 24,576 | 512 | 14 | 194,540,544 | 128×512 | 2092096097 |
| `24k-c512-s3` | 24,576 | 512 | 14 | 194,540,544 | 128×512 | 2109773078 |
| `32k-c512` | 32,768 | 512 | 14 | 202,929,152 | 128×512 | 766903225 |
| `24k-c128` | 24,576 | 128 | 14 | 194,147,328 | 512×128 | 766903225 |

(The "windows per update" column is as registered; it was changed to 512 windows for every arm, see below and deviation 2.)

- **Shape**: the tiny_gpt family is unchanged: d1024, 16 heads, SwiGLU d_ff 2560, RMSNorm, no bias, tied embedding and output layer, learned positions. Layers are the integer closest to 200M: the embedding 16K saves against 24K buys exactly one more layer; 32K has the same layers as 24K with 4% more parameters (precedent: vimeml-tiny-zh-v1's 8K arms were 0.8% over budget). `24k-c128` has the default arm's shape with only the position table changed, isolating context.
- **Batch strategy**: every arm uses a fixed **512 windows** per update, the same batch size as vimeml-tiny-zh-v1. The original registration was "a fixed 65,536 capacity tokens" (128 windows × 512 for c512 arms, 512 windows × 128 for c128), but windows turned out to average only about 34 tokens, so 128 windows gave only about 4,320 real tokens per update, a fifth of v1; see deviation 2.
- **Seeds**: the three `24k-c512` seeds were drawn at once with `secrets.randbelow(2**31)` and registered here; the other arms use s1's seed. Seed spacing was checked so that chat streams do not overlap.
- **Data**: exactly as vimeml-tiny-zh-v1: the P1 pool (21.28B characters) once, plus LCCC chat at a 5% supervised-token quota drawn as random passes through the whole run. No repetition (v1's conclusion).
- **Recipe**: as v1: cosine 6e-4 → 6e-5, warmup 1000, wd 0.1, β 0.9/0.95, clip 1.0, prefix crop 0.30, BF16 + compile.
- **Tokenizers**: 16K/24K/32K, trained on exactly the same frozen sample, seed and SentencePiece settings as 8K/12K, changing only vocabulary size.
- **Validation points**: 20 (1/20 … 20/20), each with full FP32 web/chat validation and memorization-sample BPC; one checkpoint kept at 1/20 and one at 1/5. **Only the final checkpoint is evaluated**, not chosen by any result.

## Evaluation (scored separately; rules frozen here)

> Scored and decided on 2026-11-14: none of the three contrasts is effective, so the defaults apply: 24K, context 512. See the [judgement report](reports/zh-200m-20261114/judgement.md).

As in the roadmap, all on development:

- **Primary metric**: Top-3 hit rate (micro) on track A of the prediction benchmark `ime-predict-zh-v1-r1`. Tracks B/C/D, per-source strata and confidence curves are descriptive.
- **Three contrasts in the decision family**: 16K vs 24K, 32K vs 24K, c128 vs c512. Effectiveness follows v1's standard: the primary-metric difference exceeds the three-seed range, and the Holm-corrected cluster test p < 0.05.
- **Defaults when none is effective**: 24K, context 512 (registered in the roadmap).
- **Each arm also reports**: full-pinyin Top-1 on reranking benchmark `ime-standard-zh-v1-r1` development (a gate: no significant drop); web/chat BPC; the memorization sample; the rate at which the model's Top-1 prediction appears verbatim in the training data; desktop INT8 package size, resident memory and p50/p95 latency (on the Mac, after training is delivered).
- Blind is not read. The prediction blind is opened once only, in the final desktop model's confirmation plan; the reranking blind is a reference test only.

## Running

Data preparation extends v1's pipeline for zh-200m: tokenizer entry point `scripts/tokenizer/train_zh200m.py` (config `configs/tokenizer-zh200m.toml`); encoding and windows `scripts/tokenizer/encode_zh200m.py` (`configs/data-zh200m.toml`, c512; `index --context 128` builds the 24K c128 windows; after an interruption, `--resume` in the same output directory continues, reusing only files with a completion record and matching inputs); six-arm mixtures `scripts/corpus/prepare_zh200m.py`; training entry point `scripts/training/train_zh200m.py` and six `configs/train-zh200m-*.toml`. Mixture `dataset_sha256` values are pinned into the configs once built; training jobs are submitted per arm and resume automatically from `last.pt`. The tests in `tests/test_zh200m_training.py` pin the arm table and config guards.

## Before training

- [x] Three tokenizers (16K/24K/32K) trained on the frozen sample, with manifest and SHA256 fixed. Done 2026-11-12: [train_zh200m.py](../scripts/tokenizer/train_zh200m.py) ran on a compute machine; the sample SHA256 equals vimeml-tiny-zh-v1's train-sample.txt (`f28dc84e…`); the three tokenizer.model SHA256s are in `artifacts/tokenizers/zh-200m/manifest.json`; reserved IDs `<CTX>` 4 / `<READ>` 5 / `<OUT>` 6 / `\n` 7, 256 byte pieces, validation round-trip 428,750/428,752 (2 meta-space lines skipped, as in v1). Training directly on the development machine was not feasible because of CPU throttling, so it ran on a compute machine instead (at the user's direction).
- [x] P1 and LCCC encoded for all three vocabularies; c512 windows (three vocabularies) and 24K c128 windows built; six-arm mixture maps built with `dataset_sha256` pinned into the configs. Done 2026-11-13: the three token stores checked byte for byte (16K 30,075,892,282 / 24K 28,406,081,214 / 32K 27,411,228,006); the uploaded window indexes were incomplete and were rebuilt deterministically from the token stores with window counts checked (16K c512 331,553,667, 24K c512 331,436,403, 24K c128 353,916,830, 32K c512 400,066,540); the six mixture SHAs are pinned and `matches_pinned_sha` is true for every arm.
- [x] CPU smoke: a few updates per arm, all 20 measurement points, resume path; `--dry-run` checks parameter counts (198,212,608 / 194,540,544 / 202,929,152 / 194,147,328) and steps. Replaced by full dependency checks plus first-step output inside the training jobs (local compute was not enough to run the smoke); all six passed before real training.
- [x] GPU smoke: measure microbatch memory and step time; if the microbatch must change, change only the microbatch size, not the batch strategy, and record it as a deviation. Confirmed in the first hour of training: c512 arms `lm_microbatch` 128 (4 accumulation steps, memory comparable to a single 512-window forward), about 0.06 s per step; 24k-c128 uses 512.
- [x] Training data distributed to the training machines through a private channel and checked against the SHA lists (details not in the repository).

## Limits

- After the first 200M measurement, the roadmap's desktop budget (p95 ≤ 100 ms, resident ≤ 1 GB) may be revised once; sizes over budget are still reported but are not release candidates.
- 22.4B characters is not generous for 200M (about 4B tokens meets the Chinchilla ratio; actual is about 3–4 times that). If the scaling gain is clear, whether to expand the corpus is decided before 800M, per the roadmap.
- Context 512 may still be short for long dialogues; track A's left-context truncation rule is frozen on the benchmark side, consistent on both sides.

## Deviations

1. 2026-11-12: the encoding runs used `--no-verify-inputs`. export-r1 had been fully verified that day in two independent runs (a standalone `sha256sum -c SHA256SUMS` returned rc=0; the tokenizer run's `check_files` full hash passed), so the encoding CLI only checked the SHA256SUMS file list and skipped per-file content hashing, to avoid rereading 84 GB per vocabulary. The files were not changed after self-verification (read-only on the same volume). The development machine's SentencePiece trainer was heavily throttled in the EM phase, so tokenizer training moved to a compute machine; remote storage has high latency for small writes, so encoding wrote to the machine's local disk first and copied back in one go.
2. 2026-11-13: batch strategy changed from "fixed capacity" to "fixed 512 windows". After training started, windows turned out much shorter than the context (about 34 tokens on average): 128 windows per update on c512 arms gave only about 4,320 supervised tokens, 512 windows on the c128 arm about 15,690, against about 21,500 for v1. At the 2/20 point the four c512 arms had web BPC 6.46 against 4.47 for 24k-c128, a gap that could not come from parameters or data. From 2026-11-13 19:30 the four c512 arms were rerun with 512 windows (`lm_microbatch` kept at 128, 4 accumulation steps, same memory); 24k-c128 was already running with 512 windows and was not rerun. The change came before any arm completed, and the decision rules did not change.
