# vimeml-tiny-zh-v1 eight-arm training record (2026-11-10)

All eight arms of the [preregistered plan](../../training-zh-v1-plan.md) trained as planned. This page records only the training process and LM validation results; IME scoring and the `judge_zh13.py` decision are in the [benchmark results](ime-benchmark.md). Blind was not read.

## Submission and running

- 2026-11-10 19:00: eight single-GPU jobs, one per arm, submitted at once with a local submit script (not in the repository).
- Training data was fetched from private storage the same day and checked against `configs/data-zh13.sha256` (186 files). Before submission, `train_zh13.py --dry-run` was rerun for each arm; parameter counts (12K 20,036,736 / 8K 20,160,384), step counts and `dataset_sha256` all matched the preregistration.
- 19:16 all eight arms training; 23:24 first arm done, 00:10 last arm done. No preemption, failure or resume; no arm needed a rerun.
- Each arm trained for 13,844–15,809 seconds (3.8–4.4 hours), about 0.01 seconds per step. W&B recorded offline as planned (`artifacts/tracking/tiny-zh13-*/wandb/offline-run-*`); no blind or test set was used.

| Arm | Finished | Training seconds |
| --- | ---: | ---: |
| 12k-fresh-s1 | 23:24 | 13,960 |
| 12k-fresh-s2 | ~23:25 | 13,844 |
| 12k-fresh-s3 | ~23:31 | 14,268 |
| 12k-ep5 | ~23:34 | 14,432 |
| 12k-ep20 | ~23:31 | 14,106 |
| 8k-fresh | ~00:09 | 15,809 |
| 8k-ep5 | ~00:08 | 15,650 |
| 8k-ep20 | ~00:08 | 15,845 |

## Integrity checks

All eight arms pass: `summary.json` status `complete`, steps equal to each arm's `max_steps` (844,730–868,027), `final.pt` present and loadable (optimizer step equals total steps), `milestones.json` with exactly 20 validation points and none missing, W&B offline run directory complete. Checkpoint choice is the preregistered final checkpoint, not chosen by BPC or IME results.

## Results (20th measurement point, full FP32 validation)

| Arm | Web BPC | Chat BPC | P20 memorization sample BPC |
| --- | ---: | ---: | ---: |
| 12k-fresh-s1 | 4.354450 | 5.690730 | 4.340365 |
| 12k-fresh-s2 | 4.355557 | 5.696010 | 4.341708 |
| 12k-fresh-s3 | 4.353255 | 5.680975 | 4.338744 |
| 12k-ep5 | 4.346868 | 5.680422 | 4.326262 |
| 12k-ep20 | 4.377418 | 5.683382 | 4.328934 |
| 8k-fresh | 4.365979 | 5.677755 | 4.351840 |
| 8k-ep5 | 4.363467 | 5.680283 | 4.342975 |
| 8k-ep20 | 4.384047 | 5.673251 | 4.334392 |

Descriptive observations (not used for the decision):

- The three `12k-fresh` seeds' web BPC range is 0.0023 (4.353255–4.355557): run-to-run noise in BPC is small. The IME Top-1 noise band comes from the judging script's statistics on the eight arms' scores.
- Lowest validation BPC is `12k-ep5` (4.346868); both `ep20` arms are above `fresh` and `ep5` of the same vocabulary, consistent with "repetition memorizes without improving validation". The gap between memorization-sample BPC (10,000 training documents in P20) and validation BPC: fresh about 0.014, ep5 about 0.021, ep20 about 0.049; more repetition, deeper memorization of training documents.
- BPC is per character for both 8K and 12K, so they compare directly; in this table the 12K arms are slightly better overall.
- All of the above are exploratory development observations; conclusions follow the preregistered decision of [judge_zh13.py](../../../scripts/benchmarks/judge_zh13.py).

## Outputs

- Per-arm outputs in `artifacts/models/tiny-zh13-*/`: `final.pt`, `milestones.json` (full validation at 20 points), `metrics.jsonl`, `plan.json`/`config.json`/`summary.json` and two milestone checkpoints.
- A scoring package (the eight `final.pt` files, metadata and a SHA256 list) was uploaded to private storage for IME scoring on the local machine that holds the frozen benchmark; the tokenizers were already in private storage.
- Local scoring entry point: `score_zh13_lm.py CHECKPOINT TOKENIZER_MODEL OUT.jsonl`; statistics and decision as in the preregistered plan.
