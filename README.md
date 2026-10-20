# VimeML

Compact Japanese causal language models for kana-to-kanji candidate reranking and experimental text completion in the Vime input method.

AzooKey supplies dictionary candidates. The language model reranks the fixed candidate pool using left context; dictionary retrieval remains an application dependency. Short continuation and next-word suggestions are secondary research features. These models are not instruction-tuned conversational assistants.

## Model releases and status

| Model | Parameters | Tokenizer | Context | Status |
| --- | ---: | --- | ---: | --- |
| V1 | 7,386,624 | 16K SentencePiece Unigram, V1 vocabulary | 128 | Public baseline and fallback: [Voltline/vimeml-tiny-ja-v1](https://huggingface.co/Voltline/vimeml-tiny-ja-v1) |
| V2.1 | 12,537,920 | 16K SentencePiece Unigram, V2 vocabulary | 128 | **Deployed**, public: [Voltline/vimeml-tiny-ja-v2.1](https://huggingface.co/Voltline/vimeml-tiny-ja-v2.1) |
| V4.2 | 15,070,848 | 8K SentencePiece Unigram, V4 vocabulary | 128 | Research release, opt-in client variant: [Voltline/vimeml-tiny-ja-v4.2](https://huggingface.co/Voltline/vimeml-tiny-ja-v4.2) |

V2.1 is the deployed model: `extend5` checkpoint step 40,000, symmetric INT8 block-32 weight compression, FP32 computation, and CPU-only Core ML on iOS 18 or later (14.33 MB package). V3 A/B, V4-B and V4.1 are completed experiments without a stable replacement advantage.

V4.2 (2026-10-20) is the strongest V4 model: a randomly initialized 8K-vocabulary model trained for twenty epochs on web text with 5% dialogue. It leads on Standard development but is only +6 over V2.1 on the fixed reference (p=0.56), trails on AJIMEE, and differs mainly in kana/kanji notation choices. Its INT8 Core ML package is 17.16 MB; on an iPhone 16 Pro Max test host it lowers peak footprint by 5.6 MB and tail rerank latency, while single calls are 0.16–0.35 ms slower. V2.1 remains the client default; V4.2 is available only as an opt-in variant pending keyboard-extension measurement.

Release bundles include inference weights, matching tokenizer assets, Core ML resources, and aggregate evaluation records. Training checkpoints, datasets, credentials, and private benchmark texts are stored separately. The custom checkpoint format is not a Transformers `AutoModel` implementation. Details appear in the [model card](MODEL_CARD.md).

## Evaluation summary

IME columns report exact accepted-reference Top-1 counts on fixed AzooKey candidate pools, scored by the joint-tokenization suffix log-probability sum. Standard IME labels are frozen AI-reviewed annotations, not native-speaker gold.

| Model | Standard dev / 728 | Fixed reference / 725 | Legacy / 500 | AJIMEE / 200 | Historical dev / 137 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V2.1 FP32 | 505 | 535 | 374 | **144** | 122 |
| V2.1 INT8 (deployed) | 506 | — | **375** | **144** | 122 |
| V3 A | 508 | **546** | 368 | 136 | 123 |
| V4-B | 498 | — | 377 | **144** | 120 |
| V4.1 epoch 16 | 505 | — | 376 | 137 | **124** |
| V4.2 FP32 | **523** | 541 | 370 | 139 | 123 |
| V4.2 INT8 | 522 | — | 371 | 139 | 123 |

The 725-case reference split was consumed in the initial V2.1/V3 A comparison; later models are scored once under a preregistered plan (V3 B: 533; V4.2: 541). Standard development has been reused across experiments. Paired p-values are exploratory. INT8 rows come from the Mac Core ML harness; strict logits alignment fails for both INT8 models, while task-level counts move by at most one case.

Historical sets with original V1/V2 baselines:

| Model | Validation BPC ↓ | AJIMEE / 200 | Historical development / 137 | Draft development / 2,000 |
| --- | ---: | ---: | ---: | ---: |
| V1 FP32 | 3.4736559 | 124 | 122 | 1,453 |
| V2.0 FP32 | 3.2598221 | 125 | 124 | 1,463 |
| V2.1 release FP32 | 3.0802686 | 144 | 122 | 1,487 |
| V2.1 release INT8 | Not measured | 144 | 122 | 1,481 |

On the same validation text, V4-B and V4.2 reach 3.027777 and 3.045152 BPC with the 8K tokenizer. Full methods and limitations appear in [evaluation](docs/evaluation.md); V4.2 details are in the [training summary](docs/training-v42.md) and [Core ML report](docs/reports/mac-20261020/v42-coreml.md).

## Source installation and inference

Python 3.11 or later is required. Platform-specific versions are pinned in `requirements.txt`; the recorded Linux/CUDA training environment uses `requirements-autodl.txt`. Optional benchmark preparation dependencies are listed in `requirements-evaluation.txt`.

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
| `configs/` | Versioned experiment and data-processing configurations |
| `annotations/` | Versioned corpus review decisions |
| `docs/` | Primary English documentation and dated experiment records |
| `docs/zh-CN/` | Chinese documentation and detailed experiment reports |
| `examples/ios/` | Core ML probe, SentencePiece bridge, and historical client examples |
| `templates/` | Artifact exchange templates |
| `tests/` | Targeted implementation tests |
| `datasets/`, `artifacts/`, `outputs/`, `runs/`, `handoff/` | Local, Git-ignored data, model artifacts, results, and transfer manifests |

## Documentation and licensing

[Documentation index](docs/index.md) · [Model card](MODEL_CARD.md) · [Chinese overview](README.zh-CN.md) · [Contribution conventions](CONTRIBUTING.md) · [Licensing and attribution](NOTICE.md)

Project code and published model weights use GPL-2.0. Upstream libraries and source datasets retain their own licenses. Private research benchmark text is not included in the public repository or model releases. Third-party notices remain with the corresponding components.
