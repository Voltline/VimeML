# zh-800m Chinese desktop model: training plan (preregistered)

Drafted 2026-11-15. This is the 800M training plan of the [desktop model roadmap](benchmarks/predict-zh-plan.md), whose gate was met on 2026-11-14 (the step-0 survey did not defer 800M either). It registers shape, vocabulary, data, recipe and evaluation; it counts as registered once committed, and later changes go to the deviation log at the end. The registered text is the Chinese version, [docs/zh-CN/training-zh-800m-plan.md](zh-CN/training-zh-800m-plan.md); this page is its translation.

## Questions

1. **Scaling gain**: how much more do prediction metrics gain from 200M to 800M? This gives three points: 20M, 200M, 800M.
2. **Distillation teacher**: with the same 12K tokenizer as the 20M, does 800M work as a token-level distillation teacher for the vimeml-tiny-zh-v1 line?

One arm only: the 200M vocabulary and context contrasts were not effective.

## Shape

| Arm | Vocabulary | Context | d_model | Heads | SwiGLU d_ff | Layers | Parameters | Seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `12k-c512` | 12,288 | 512 | 1536 | 24 × 64 | 4096 | 27 | 784,157,184 | 1816189249 |

- The tiny_gpt family: RMSNorm, no bias, tied embedding and output, learned positions. 28,314,624 parameters per layer; the layer count is the integer closest to 800M. Head size stays 64. Model: `src/vimeml/training/model_zh800m.py`, with an FP32 KV-cache path (required for every new model).
- **12K vocabulary (the vimeml-tiny-zh-v1 tokenizer), not the 200M default of 24K.** 16K/24K/32K showed no effective difference at 200M, and the 2026-11-13 decision puts every pinyin-generation model on 12K: token-level distillation into the 20M, and the 12K pinyin data, `zh_search` and the Vime chain carry over. The cost is about 10% more tokens for the same text than 24K.
- **Context 512.** Windows are paragraphs, about 34–42 tokens on average, and c128 vs c512 was not effective at 200M; 1024 would be used by almost no sample.
- The seed was drawn once with `secrets.randbelow(2**31)` when this plan was written.

## Data

One pass, no repetition. Web is three 12K token stores joined:

| Store | Content | Origin |
| --- | --- | --- |
| `zh13-web-12k` | export-r1 P1 (21.28B characters), with tiers and the memorization probe | existing vimeml-tiny-zh-v1 store, not re-encoded |
| `zh800m-web-r1rest-12k` | export-r1 train documents outside P1 (16.77B characters), same pools | new, selection `rest` |
| `zh800m-web-r2-12k` | every export-r2 train document (88.4B characters) | new, selection `all` |

- About 126.5B web characters in all (all of r1, 38.05B, plus r2). At 12K, about 1.40 characters per content token: about 90B content tokens, about 115 per parameter.
- **Chat: LCCC (`zh13-chat-12k`) at 2% of supervised tokens, not the 20M/200M 5%.** At 5% LCCC would repeat about 24 times, too much memorization risk at 800M; 2% is about 10 passes, still more than twice the 200M amount. Same scheme as v1: one run-long sequence of passes, pass k a fresh permutation with crop seed set k; 12 sets are prepared.
- Each web store gets its own crop seed (`seed + k·constant`). The mixture map is uint32, one file per epoch, SHA256-pinned in the config.
- Validation is `zh13-web-12k`'s validation (export-r1) and LCCC validation, directly comparable with 20M and 200M; the memorization probe stays the 10,000 tier-0 documents of P1.
- The store format is unchanged (`vimeml_sentence_tokens_v1`). `encode_zh13.py` gains the `rest` and `all` selections, which write a web store only, without `train.tiers.bin`, and record `selection` in the manifest.

## Recipe

- **2048 windows per update** (about 80k supervised tokens, four times 200M). At 512 windows, 80B+ tokens would take about 4.8M updates, too small a batch for 800M and too small to split across ranks. About 1.0M updates expected.
- Cosine 3e-4 → 3e-5, warmup 2000; wd 0.1, β 0.9/0.95, clip 1.0, prefix crop 0.30, BF16 + compile. The learning rate drops from 6e-4 to 3e-4 for the larger model and batch.
- `lm_microbatch` 64 for now; it may change after the GPU smoke (microbatch only, never the batch), recorded as a deviation.
- Data parallel: `VIMEML_NPROC` > 1 launches torchrun; global batch and data order equal one GPU's (see the data-parallel section of the [200M plan](training-zh-200m-plan.md)). Resume may change the rank count.
- 20 milestones with FP32 full web/chat validation and the memorization probe; checkpoints kept at 1/20 and 1/5, `last.pt` every 2000 updates. Only the final checkpoint is evaluated.
- Compute: about 6·N·T ≈ 4.4×10²⁰ FLOPs, about 30 times one 200M arm.

## Evaluation (rules frozen here)

All on development; blind is not read.

- **Prediction** (roadmap rules): `ime-predict-zh-v1-r1` track A Top-3 is the primary metric, with paired cluster tests against v1 and the 200M default arm `24k-c512-s1`; tracks B/C/D descriptive.
- **Rerank guard**: `ime-standard-zh-v1-r1` development full-pinyin Top-1.
- **BPC and memorization**: web and chat validation BPC, the memorization probe, and the share of prediction Top-1 candidates found verbatim in training data.
- **Pinyin generation is not part of this plan.** This run is LM pretraining only. Pinyin ability comes from post-training (as r2-cont) or from switching to joint pinyin pretraining, decided by the 12K joint-pinyin 200M evaluation; switching is recorded as a deviation before training starts and adds a `[pinyin]` section to the config.

## How to run

```bash
# token data (compute node; --root can write to node-local disk and copy back)
python -I -X utf8 scripts/tokenizer/encode_zh800m.py encode --part r1rest
python -I -X utf8 scripts/tokenizer/encode_zh800m.py encode --part r2
python -I -X utf8 scripts/tokenizer/encode_zh800m.py index          # c512 indexes of zh13-web-12k / zh13-chat-12k
# mixture; pin the printed dataset_sha256 in configs/train-zh800m-12k-c512.toml
python -X utf8 scripts/corpus/prepare_zh800m.py --arm 12k-c512
python -X utf8 scripts/training/train_zh200m.py --config configs/train-zh800m-12k-c512.toml --dry-run
# training (auto-resumes from last.pt)
VIMEML_NPROC=<GPUs> scripts/training/cluster_zh800m.sh
```

The entry point is shared with 200M: `train_zh200m.py` and `joint_zh200m.py` pick model, mixture and batch size from `stage = "zh-800m"`. `tests/test_zh800m.py` runs a fake export through the real 12K tokenizer end to end: P1 and `rest` partition export-r1, `all` covers every document, every web window appears exactly once in the map, chat is 2%, the map rebuilds byte for byte, and `build_plan` accepts the mixture.

## Before training

- [ ] r1rest and r2 stores encoded, `selection` in their manifests; c512 indexes built.
- [ ] Mixture built, `dataset_sha256` pinned, `--dry-run` confirms 784,157,184 parameters and the step count.
- [ ] GPU smoke: 20 `smoke_updates` reach all 20 milestones; measure microbatch memory and step time; fix the rank count.
- [ ] Training data distributed through private channels and checked against the SHA list (details stay out of the repo).

## Limitations

- export-r2 had no human review (every review-queue document was dropped), so its quality profile differs a little from r1's P1.
- One seed: compare against v1 and 200M with the 200M seed range in mind (track A Top-3 0.0165).
- No small-scale learning-rate or batch sweep (compute is ample; standard values). If the first milestones look clearly wrong, that is handled as a deviation.

## Deviations

(none yet)
