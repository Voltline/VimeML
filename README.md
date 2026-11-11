# VimeML

Compact causal language models for candidate reranking in the Vime input method: Japanese kana-to-kanji conversion, and Chinese pinyin input since `vimeml-tiny-zh-v1`.

For Japanese, AzooKey supplies dictionary candidates. The language model reranks the fixed candidate pool using left context; dictionary retrieval remains an application dependency. Short continuation and next-word suggestions are secondary research features. These models are not instruction-tuned conversational assistants.

## Model releases and status

| Model | Parameters | Tokenizer | Context | Status |
| --- | ---: | --- | ---: | --- |
| V1 | 7,386,624 | 16K SentencePiece Unigram, V1 vocabulary | 128 | Public baseline and fallback: [Voltline/vimeml-tiny-ja-v1](https://huggingface.co/Voltline/vimeml-tiny-ja-v1) |
| V2.1 | 12,537,920 | 16K SentencePiece Unigram, V2 vocabulary | 128 | Previous deployment, public: [Voltline/vimeml-tiny-ja-v2.1](https://huggingface.co/Voltline/vimeml-tiny-ja-v2.1) |
| V4.2 | 15,070,848 | 8K SentencePiece Unigram, V4 vocabulary | 128 | **Deployed**, public: [Voltline/vimeml-tiny-ja-v4.2](https://huggingface.co/Voltline/vimeml-tiny-ja-v4.2) |

V4.2 epoch 20 is the deployed model: symmetric INT8 block-32 weight compression, FP32 computation, and CPU-only Core ML on iOS 18 or later (17.16 MB package). It replaced V2.1 `extend5` step 40,000 (14.33 MB package with the same recipe). V3 A/B, V4-B and V4.1 are completed experiments without a stable replacement advantage.

V4.2 (2026-10-20) is the strongest V4 model: a randomly initialized 8K-vocabulary model trained for twenty epochs on web text with 5% dialogue. It leads on Standard development but is only +6 over V2.1 on the fixed reference (p=0.56), trails on AJIMEE, and differs mainly in kana/kanji notation choices. Its INT8 Core ML package is 17.16 MB; on an iPhone 16 Pro Max test host it lowers peak footprint by 5.6 MB and tail rerank latency, while single calls are 0.16–0.35 ms slower. On the preregistered Standard v2 blind it scores 598/729 against V2.1's 576 (+22, author-cluster p = 0.0508, just short of the registered threshold); on the written-text homophone challenge it scores 438/619 against 450 (not significant) ([confirmation results, Chinese](docs/zh-CN/benchmarks/confirmation-v2-results-20261021.md)). Keyboard-extension memory and typing latency have not been measured.

V4.3 (more web data) and V5 (d512/L8, 28.4M parameters, same data and budget as V4.3) are completed negative experiments: V5 lowers validation BPC by 0.10 but leaves IME Top-1 unchanged within noise, so neither is converted or deployed ([V5 training](docs/training-v5.md)).

Release bundles include inference weights, matching tokenizer assets, Core ML resources, and aggregate evaluation records. Training checkpoints, datasets, credentials, and private benchmark texts are stored separately. The custom checkpoint format is not a Transformers `AutoModel` implementation. Details appear in the [model card](MODEL_CARD.md).

## Evaluation summary

IME columns report exact accepted-reference Top-1 counts on fixed AzooKey candidate pools, scored by the joint-tokenization suffix log-probability sum. Standard IME labels are frozen AI-reviewed annotations, not native-speaker gold.

| Model | Standard dev / 728 | Fixed reference / 725 | Legacy / 500 | AJIMEE / 200 | Historical dev / 137 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V2.1 FP32 | 505 | 535 | 374 | 144 | 122 |
| V2.1 INT8 (previous deployment) | 506 | — | **375** | 144 | 122 |
| V3 A | 508 | **546** | 368 | 136 | 123 |
| V4-B | 498 | — | 377 | 144 | 120 |
| V4.1 epoch 16 | 505 | — | 376 | 137 | **124** |
| V4.2 FP32 | **523** | 541 | 370 | 139 | 123 |
| V4.2 INT8 (deployed) | 522 | — | 371 | 139 | 123 |
| V4.3 epoch 30 | 519 | — | 367 | 143 | 123 |
| V5 epoch 30 | 514 | — | 365 | **145** | **124** |

The 725-case reference split was consumed in the initial V2.1/V3 A comparison; later models are scored once under a preregistered plan (V3 B: 533; V4.2: 541). Standard development has been reused across experiments. Paired p-values are exploratory. INT8 rows come from the Mac Core ML harness; strict logits alignment fails for both INT8 models, while task-level counts move by at most one case.

Historical sets with original V1/V2 baselines:

| Model | Validation BPC ↓ | AJIMEE / 200 | Historical development / 137 | Draft development / 2,000 |
| --- | ---: | ---: | ---: | ---: |
| V1 FP32 | 3.4736559 | 124 | 122 | 1,453 |
| V2.0 FP32 | 3.2598221 | 125 | 124 | 1,463 |
| V2.1 release FP32 | 3.0802686 | 144 | 122 | 1,487 |
| V2.1 release INT8 | Not measured | 144 | 122 | 1,481 |

On the same validation text, V4-B, V4.2, V4.3 and V5 reach 3.027777, 3.045152, 3.043831 and 2.942180 BPC with the 8K tokenizer. Full methods and limitations appear in [evaluation](docs/evaluation.md); V4.2 details are in the [training summary](docs/training-v42.md) and [Core ML report](docs/reports/mac-20261020/v42-coreml.md).

## Chinese pinyin model

| Model | Parameters | Tokenizer | Context | Status |
| --- | ---: | --- | ---: | --- |
| `vimeml-tiny-zh-v1` | 20,036,736 | 12K SentencePiece Unigram, Chinese | 128 | Blind-confirmed; export, device tests and release pending |

`vimeml-tiny-zh-v1` reranks the pinyin candidates of librime with rime-ice using left context. It was trained from random initialization on 22.4B characters of cleaned and reviewed FineWeb2 `cmn_Hani` web text and LCCC dialogue, as the release arm (`12k-fresh-s1`) of a preregistered eight-arm experiment ([plan](docs/zh-CN/training-zh-v1-plan.md), [training](docs/zh-CN/reports/zh-v1-20261110/training.md), [judgement](docs/zh-CN/reports/zh-v1-20261110/ime-benchmark.md)). Internal code and artifact paths still use the codename `zh13`.

Exact Top-1 on `ime-standard-zh-v1-r1` (CrossWOZ, RiSAWOZ and KdConv dialogue turns; AI-reviewed labels, not native-speaker gold). The blind split was consumed by the preregistered confirmation, which passed (+329/−19 against Rime, p = 3.6e-74); the comparisons were scored afterwards with blind as a reference test ([report, Chinese](docs/zh-CN/reports/zh-v1-20261111/blind-and-baselines.md)).

| System | Pinyin dev / 1,456 | Pinyin blind / 1,445 | Abbreviated blind / 964 |
| --- | ---: | ---: | ---: |
| **vimeml-tiny-zh-v1** | **93.6%** | **93.8%** | **35.0%** |
| Qwen2.5-0.5B, same pool and scoring | 91.3% | 91.8% | 31.0% |
| Qwen3-0.6B-Base, same pool and scoring | 89.8% | 89.2% | 30.6% |
| rime-ice + Wanxiang grammar model | 79.0% | 82.2% | 15.7% |
| rime-ice (candidate pool source) | 68.8% | 72.2% | 14.6% |
| Luna Pinyin (Rime default) | 68.3% | 68.9% | 12.9% |

Rime configurations rank on their own without left context; the models rerank a frozen 20-candidate pool, which contains the answer for 97.4% of blind pinyin items and about 41% of abbreviated items. A dialogue-level leak scan of the training data found no republished benchmark dialogues.

## Source installation and inference

Python 3.11 or later is required. Platform-specific versions are pinned in `requirements.txt`; the recorded Linux/CUDA training environment uses `requirements/cuda.txt`. Optional benchmark preparation dependencies are listed in `requirements/evaluation.txt`, and external-model comparisons in `requirements/external-eval.txt`.

```bash
python -m venv .venv
# POSIX activation; Windows: .venv\Scripts\Activate.ps1
source .venv/bin/activate
python -m pip install -e .
python scripts/training/infer.py \
  --checkpoint artifacts/models/tiny-ja-v2.1-e16k-d320-l6-extend5/best.pt \
  --tokenizer artifacts/tokenizers/ja-unigram-16k-v2 \
  --output outputs/inference/v21-local
```

This command assumes the matching local training artifacts and their manifests are present. The default inference CLI selects V1; V4 checkpoints load through `vimeml.training.infer_v4.JapaneseLMV4`, and `scripts/benchmarks/evaluate_v42.py` reproduces the V4.2 fixed-pool comparison. Hugging Face inference bundles have separate packaged-runtime instructions in their model cards. Core ML conversion and runtime evaluation require macOS. The iOS application is maintained in the separate Vime repository; `examples/ios/` contains historical integration references.

## Repository structure

| Path | Contents |
| --- | --- |
| `src/vimeml/` | Corpus processing, tokenization, training, evaluation, and deployment implementations |
| `scripts/` | Command-line entry points organized by subsystem |
| `configs/` | Versioned experiment and data-processing configurations; `configs/annotations/` holds Japanese corpus review decisions |
| `requirements/` | Additional dependency pins for CUDA training, benchmark preparation and external-model evaluation |
| `docs/` | Primary English documentation and dated experiment records |
| `docs/zh-CN/` | Chinese documentation and detailed experiment reports |
| `examples/ios/` | Core ML probe, SentencePiece bridge, and historical client examples |
| `templates/` | Artifact exchange templates |
| `tests/` | Targeted implementation tests |
| `datasets/`, `artifacts/`, `outputs/`, `runs/`, `handoff/` | Local, Git-ignored data, model artifacts, results, and transfer manifests |

## Documentation and licensing

[Documentation index](docs/index.md) · [Model card](MODEL_CARD.md) · [Chinese overview](README.zh-CN.md) · [Contribution conventions](CONTRIBUTING.md) · [Licensing and attribution](NOTICE.md)

Project code and published model weights use GPL-3.0-or-later with an additional permission under section 7 for application distribution services such as the App Store ([LICENSE-EXCEPTION](LICENSE-EXCEPTION)). Releases before 2026-10-22 used GPL-2.0-only. Upstream libraries and source datasets retain their own licenses. Private research benchmark text is not included in the public repository or model releases. Third-party notices remain with the corresponding components.
