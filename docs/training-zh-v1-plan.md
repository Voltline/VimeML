# vimeml-tiny-zh-v1: fresh data vs repeated epochs, 8K vs 12K (preregistered)

Renamed 2026-11-11: this series was first called "Vime 1.3 Chinese 20M"; the formal name is `vimeml-tiny-zh-v1`, following the Japanese `vimeml-tiny-ja-v*` naming. Only names and doc paths changed, not the decision rules. `zh13` and `zh-v13` in code, configs, outputs and storage paths are internal code names that trained outputs and checksum lists depend on, so they stay.

Drafted 2026-11-06. It counted as registered only once every item under [Before training](#before-training) was done and committed; before any arm started training, the text could only be completed, not have its decision rules changed. All items were completed on 2026-11-10 and the eight arms trained that day, see the [training report](reports/zh-v1-20261110/training.md) and the [benchmark results](reports/zh-v1-20261110/ime-benchmark.md).

## Questions

1. **The cost of repeated data.** For the same 22.4B characters seen, is it better to use all-new data, or to repeat a much smaller set for many epochs as Japanese V4.2 did? The Japanese corpus has only about 1.1B characters and had to be repeated 20 times; the Chinese web train split has about 37.9B characters, the first chance not to repeat.
2. **8K or 12K.** The [tokenizer vocabulary comparison](reports/zh-v1-20261106/tokenizer-vocab.md) shows 12K compresses 11% better and fits about 20 more characters of context, but costs a layer at the 20M budget. Tokenizers alone cannot decide this; it needs a comparison at equal data.

Both questions go into one batch.

## Eight arms

Total budget: each arm sees about **22.4B characters** of training text, the same order as V4.2's cumulative exposure (12.1B supervised targets, about 21B characters). That is about 16.0B tokens at 12K and about 18.1B at 8K. The budget is fixed in characters rather than tokens so that both vocabularies see the same amount of text.

| Arm | Vocabulary | Layers | Parameters | Web pool | Web epochs | Seed |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| `12k-fresh-s1` | 12K | 9 | 20,036,736 | P1 (about 21.3B characters) | 1 | 1591996171 |
| `12k-fresh-s2` | 12K | 9 | 20,036,736 | P1 | 1 | 136617650 |
| `12k-fresh-s3` | 12K | 9 | 20,036,736 | P1 | 1 | 1137767227 |
| `12k-ep5` | 12K | 9 | 20,036,736 | P5 (about 4.26B characters) | 5 | 1591996171 |
| `12k-ep20` | 12K | 9 | 20,036,736 | P20 (about 1.06B characters) | 20 | 1591996171 |
| `8k-fresh` | 8K | 10 | 20,160,384 | P1 | 1 | 1591996171 |
| `8k-ep5` | 8K | 10 | 20,160,384 | P5 | 5 | 1591996171 |
| `8k-ep20` | 8K | 10 | 20,160,384 | P20 | 20 | 1591996171 |

- **Nested pools.** Each web document gets a fixed value u in [0, 1) from `sha256('zh-v13-pool\0' + id)`. P1 takes u < q, with q set so P1 holds about 21.28B characters (about 56% of web train); P5 takes u < q/5 and P20 u < q/20. So P20 ⊂ P5 ⊂ P1: the three approaches differ only in repetition, and changing vocabulary does not change which documents are chosen. Actual character counts are recorded during data preparation, and differences in what each arm sees are reported as they are.
- **Chat.** All arms use the same LCCC train pool (about 209M characters) at 5% of supervised tokens, as in V4.2. Over 22.4B characters the chat part repeats about 5 times, equally in every arm; "epochs" refers to web data only. Chat is drawn as one series of passes running through the whole training run: pass k is a fresh random permutation with the k-th set of crop seeds, and at the end of each web epoch chat is drawn up to 5/95 of the cumulative web supervised tokens. So whether web data runs for 1 epoch or 20, each repetition of an utterance gets a new crop position, and chat is drawn the same way in every arm.
- **Seeds.** The three seeds were drawn at once with `secrets.randbelow(2**31)`. The three `12k-fresh` arms differ only in seed and estimate the run-to-run noise of these models on the Chinese benchmark; the other five use seed s1 but differ in pool or vocabulary, so their data order differs anyway.
- **Model.** tiny_gpt_v4 architecture: d384, 6 heads, SwiGLU d_ff 960, learned positions 128, RMSNorm, no bias, tied embedding and output layer.
- **Training recipe.** As V4.2: cosine 6e-4 → 6e-5 with 1,000 warmup steps, scheduled over each arm's own total updates; batch 512 windows; BF16 + compile; AdamW (wd 0.1, β 0.9/0.95, clip 1.0); prefix crop 30%.
- **Training units.** Web text by paragraph (split on newlines) and chat by utterance, each a BOS + text + EOS sequence; paragraphs over 128 tokens are cut into adjacent windows as in V4, matching the handling of Japanese sentences.

## Evaluation (fixed before training)

- **Checkpoint.** Each arm's final checkpoint at the end of training, not chosen by BPC or IME results. Each arm also saves checkpoints at 1/20 and 1/5 of its budget, for describing learning curves only. At every k/20 of the updates (k = 1…20, rounded up) the full LM validation below runs to draw the learning curve; these intermediate results take no part in checkpoint choice.
- **LM metrics.** BPC on the export-r1 validation web and chat sets, per character (NLL ÷ characters ÷ ln 2), computed in full FP32, so the two vocabularies compare directly. BPC is also computed on 10,000 fixed training documents drawn from P20; its difference from validation reflects memorization from repetition (descriptive only).
- **IME metrics.** Development split of the Standard Chinese IME benchmark `ime-standard-zh-v1-r1`, full-pinyin track, 1,456 items; exact Top-1 is the primary metric. The abbreviated-pinyin track, Top-3, pool-conditional Top-1 and per-source results are descriptive. The Rime first-candidate baseline is 1,001/1,456. Blind is not read.
- **Scoring.** The LM reranks the candidate pool, tokenizing `BOS + left context + candidate` jointly, no EOS, FP32, full-vocabulary probabilities. The scored span starts after the common token prefix of "left context tokenized alone" and all candidate sequences, and the score is the sum of log-probabilities over that span, as in the Japanese V4.2 `evaluate_ime.score_candidates`. When the last piece of the left context merges with the start of a candidate, the common prefix breaks before the merge point, and that piece is rescored from the fork for all candidates together; candidates are never encoded alone and none is dropped. A smoke run with a randomly initialized model found this merge in 118 of the 1,456 items. 3 items have an empty Rime pool; the score file writes an empty list and these items stay in the denominator as misses. Left context plus candidate over 128 tokens raises an error instead of truncating (the benchmark's longest left context is 64 characters, so this never triggers). Implementation: [score_zh13_lm.py](../scripts/benchmarks/score_zh13_lm.py). Statistics use `compare` in [score_standard_ime_zh.py](../scripts/benchmarks/score_standard_ime_zh.py): each dialogue contributes one item, so items are paired directly, with exact McNemar p and sign-flip p.
- **Noise band.** Mean, range and standard deviation (n − 1) of Top-1 over the three `12k-fresh` arms, and the same for BPC. Paired p values for the three seed pairs are also reported, to show how often two runs of the same recipe come out "significant".

**Decision**: five contrasts form one test family, Holm-corrected, family α = 0.05, two-sided.

| Contrast | Question |
| --- | --- |
| `12k-fresh-s1` vs `12k-ep20` | Fresh data vs 20 epochs at 12K |
| `8k-fresh` vs `8k-ep20` | Fresh data vs 20 epochs at 8K |
| `8k-fresh` vs `12k-fresh-s1` | Vocabulary with fresh data |
| `8k-ep5` vs `12k-ep5` | Vocabulary at 5 epochs |
| `8k-ep20` vs `12k-ep20` | Vocabulary at 20 epochs |

A contrast is "effective" only if both hold:

1. the absolute Top-1 difference between the two arms exceeds the range of the three `12k-fresh` seeds;
2. the Holm-corrected exact McNemar p < 0.05.

Comparisons of `ep5` with the other two approaches, BPC and the abbreviated-pinyin track are descriptive only, for the dose-response direction, and take no part in the decision.

## Running (data and eight arms)

Data download and training are submitted as jobs to a job-scheduled cluster, one GPU per job. [`launch_zh13_all.sh`](../scripts/training/launch_zh13_all.sh), which starts all eight processes on one multi-GPU machine, is kept for cases where one machine is used whole.

1. Fetch data: a CPU job runs a local data download script (not in the repository). Credentials are passed only as environment variables and never written to files. The job must end by printing `zh13 data matches configs/data-zh13.sha256 (186 files)`.
2. Check each arm: `prepare_zh13.py --arm <arm>` must print `matches_pinned_sha: true`, and `train_zh13.py --dry-run` must report 20,036,736 parameters for 12K or 20,160,384 for 8K, with the matching step counts.
3. Submit the eight arms: a local submit script (not in the repository) submits one single-GPU job per arm, named `vimeml-zh13-<arm>`, each calling [`cluster_zh13.sh`](../scripts/training/cluster_zh13.sh) (checks the maps, then trains; resumes automatically when `last.pt` exists).
4. Watch: after preemption or a restart, rerun the same submit command; running arms are skipped and stopped ones resume from `last.pt`. If one arm repeats the same error, stop and report instead of changing code to get around it.
5. W&B: no online tracking for this batch; without a credentials file the script records offline, and `wandb sync` can run afterwards if needed.

## How the conclusions are used

- **Data.** If fresh data wins effectively, later Chinese models use non-repeated data and spend extra budget on new documents first. If 20 epochs wins effectively, repetition is better at this scale and later sampling is designed around multiple epochs. If neither is effective, the default is fresh data: no repetition is closer to how larger models are trained and carries no overfitting risk. If both data contrasts (12K and 8K) are effective in opposite directions, the 12K result decides, since 12K is the default vocabulary.
- **Vocabulary.** Whichever vocabulary wins effectively is used; if the three data approaches disagree in direction, the fresh-data result decides; if the fresh-data contrast is not effective and 5 and 20 epochs point in opposite directions, the default is also 12K. If none is effective, the default is 12K, based on the "most balanced" finding of the tokenizer report.
- These are exploratory development conclusions. A separate confirmation plan, preregistered on the Chinese blind set, comes before deployment.
- 16K, 24K and 32K for the 200M model are not in this batch and have their own plan.

## Before training

- [x] Export the review part and run `finish` to freeze export-r1. P1/P5/P20 are all drawn from the frozen train split, and later documents would change the pools, so freezing comes first. Done 2026-11-10: review documents went through a first review and a second review of political content, and only those with every vote keep were included, 381,935 documents (train 381,744, validation 191); export-r1 has 104 data files, 45.0 GB, uploaded whole to private storage.
- [x] Formal tokenizers: one 8K and one 12K, retrained on the frozen train split with `\n` as a user_defined_symbol, other settings as in the tokenizer report (Unigram, identity, byte_fallback, coverage 0.9995). Write SHA256s. Code: [train_zh13.py](../scripts/tokenizer/train_zh13.py). Done 2026-11-10, output in `artifacts/tokenizers/zh-v13/`:
  - Sampling differs from the vocabulary report: review files are merged back into the web files by source shard, each shard gets the same quota, and documents are drawn uniformly at random within a shard (the vocabulary report took 2 row groups per file). This keeps review documents at their natural share, 1.67% of web paragraphs (25,053 / 1,500,000); equal sampling per file would have given 24%. Another 80,000 chat utterances are drawn.
  - Sample file `train-sample.txt`: 277,161,535 bytes, 97.57M characters (80.54M Han), SHA256 `f28dc84edb08e54f3cd8b345444b6654e62c6c0fe0a1907e25dbb0a2cb4a77b9`.
  - `zh-unigram-8192/tokenizer.model`: SHA256 `2bb385d2ae78941d945339c99823cd8752fe97e4abbdeea3461dd8361cc0d790`.
  - `zh-unigram-12288/tokenizer.model`: SHA256 `68bd37c45bcb018b32b2eef5b241561ab5b6dc1b77c1c3aadcc1b41c4bf47e5a`.
  - Both checked: reserved IDs `<CTX>` 4, `<READ>` 5, `<OUT>` 6, `\n` 7, 256 byte pieces, all 428,750 validation lines round-trip exactly.
  - The whole directory is uploaded to private storage; the training side fetches it into `artifacts/tokenizers/zh-v13`, and SHA256s are checked against the manifest before encoding.
- [x] Encoding and pools: P1 encoded once per vocabulary (P5 and P20 are subsets of P1 and need no separate encoding); record q, document and character counts per pool, and manifests of validation and the P20 memorization sample. Code: `measure` and `encode` in [encode_zh13.py](../scripts/tokenizer/encode_zh13.py). Run locally (where export-r1 lives) on 2026-11-10 with [build_zh13_data.sh](../scripts/tokenizer/build_zh13_data.sh):
  - Pools: web train has 29,599,087 documents, 38.05B characters. P1 takes 16,551,843 documents, 21.280B characters; P5 3,310,736 documents, 4.257B; P20 827,248 documents, 1.065B. The memorization sample has 10,000 documents, 12.61M characters, all in P20. Threshold (first 8 bytes of sha256) q = 10317501391349349668; P5 uses q // 5, P20 q // 20. `pools.json` SHA256 `3231ac37dc536b3479d630db27539e56db613c86994b3c87963dc8aff50af4da`.
  - Token data: P1 web has 330,491,330 paragraphs; 1,111 paragraphs containing U+2581 are dropped. 12K gives 15.15B content tokens, 1.404 characters per token; 8K gives 17.10B, 1.245 characters per token. c128 windows: 361M for 12K, 371M for 8K. Validation has 14,872 documents, 19.38M characters; LCCC train has 16,326,323 utterances, 209M characters.
  - Manifests record no timings (timings go to `run.json`, which is not hashed). The same export, tokenizers and sentencepiece 0.2.1 were built in two places, and the smoke results match byte for byte.
- [x] Mixture maps: per-arm epoch maps and chat quotas, with manifest SHAs written into the configs. Code: [prepare_zh13.py](../scripts/corpus/prepare_zh13.py). All eight built on 2026-11-10, `dataset_sha256` written into each config, and `train_zh13.py --dry-run` passes for all eight. Actual exposure per arm: web 21.280B characters (21.285B for ep5 and 21.301B for ep20, slightly over 1/5 and 1/20 because of threshold rounding). Chat about 4.12 passes for 12K and 4.20 for 8K (the estimate above was about 5; chat tokens are drawn at 5/95 of web supervised tokens, so slightly fewer in characters). Total supervised tokens about 14.52B for 12K and 16.37B for 8K. Updates about 845K for 12K and 867K for 8K.
  - The training side does not re-encode. It downloads the built tokenizers, token data, window indexes and eight-arm maps from private storage, about 85 GiB, and checks each file against [data-zh13.sha256](../configs/data-zh13.sha256) (186 files) in the repository, all in one step with a local download script (not in the repository). It is not rebuilt there because the training environment's numpy version differs from the local one, so byte-for-byte equality is not guaranteed; checked by SHA after download, the data is identical to the local build.
- [x] Training entry point, eight configs, single-arm cluster script and eight-arm launch script, following V4.4 (W&B credentials only from a private file, never a machine login): [train_zh13.py](../scripts/training/train_zh13.py), `configs/train-zh13-*.toml`, [cluster_zh13.sh](../scripts/training/cluster_zh13.sh) and [launch_zh13_all.sh](../scripts/training/launch_zh13_all.sh). CPU smoke (24 updates, 20 measurement points, resume) passes; full validation token counts match the manifests item by item.
- [x] Scorer: writes the JSONL needed by `score_standard_ime_zh.py --scores`, fixes the handling of piece merges, runs end to end with a randomly initialized model; handling added to [Scoring](#evaluation-fixed-before-training) above.
- [x] Judging script: [judge_zh13.py](../scripts/benchmarks/judge_zh13.py) reads the eight arms' score files and applies the rules above directly, writing `judgement.json` and `report.md`: noise band, Holm correction of the five contrasts, both effectiveness conditions, the choices under "How the conclusions are used", and all descriptive results. Before computing it checks that score files match their SHA256, scores come from the arm's `final.pt` (SHA256 against the file on disk), training status is complete with all 20 measurement points, arms with the same vocabulary share one tokenizer, and the Rime baseline is still 1,001. Any mismatch refuses a result. The two opposite-direction cases under "How the conclusions are used" (12K and 8K data contrasts disagreeing; 5 and 20 epoch vocabulary contrasts disagreeing) now have rules, which the script follows with no manual judgement. Tested with synthetic eight-arm data on normal conclusions, every decision branch and the refusal paths.
- [x] The training side fetches training data from private storage (export-r1 raw text not needed) and checks each file against `configs/data-zh13.sha256`. Done 2026-11-10: the local download script ran with zero failures on 191 files (85 GiB), `sha256sum -c` passed for all, and the log ended with `zh13 data matches configs/data-zh13.sha256 (186 files)`. The eight arms trained the same day; see the [training report](reports/zh-v1-20261110/training.md).
