# VimeML

Compact Japanese causal language models for kana-to-kanji candidate reranking and experimental text completion in the Vime input method.

AzooKey supplies dictionary candidates. The language model reranks the fixed candidate pool using left context; dictionary retrieval remains an application dependency. Short continuation and next-word suggestions are secondary research features. These models are not instruction-tuned conversational assistants.

## Model releases

| Release | Parameters | Tokenizer | Context | Hugging Face |
| --- | ---: | --- | ---: | --- |
| V1 | 7,386,624 | 16K SentencePiece Unigram, V1 vocabulary | 128 | [Voltline/vimeml-tiny-ja-v1](https://huggingface.co/Voltline/vimeml-tiny-ja-v1) |
| V2.1 | 12,537,920 | 16K SentencePiece Unigram, V2 vocabulary | 128 | [Voltline/vimeml-tiny-ja-v2.1](https://huggingface.co/Voltline/vimeml-tiny-ja-v2.1) |

V2.1 is the current deployment release: `extend5` checkpoint step 40,000, with symmetric INT8 block-32 weight compression, FP32 computation, and CPU-only Core ML execution on iOS 18 or later. The Core ML package is 14.33 MB. V1 remains a baseline and fallback. V3 A and V3 B are completed research experiments; neither demonstrated a stable task-level advantage sufficient to replace V2.1.

Release bundles include inference weights, matching tokenizer assets, Core ML resources, and aggregate evaluation records. Training checkpoints, datasets, credentials, and private benchmark texts are stored separately. The custom checkpoint format is not a Transformers `AutoModel` implementation. Details appear in the [model card](MODEL_CARD.md).

## Evaluation summary

The BPC column uses the same frozen V1/V2 validation text. IME columns report exact accepted-reference Top-1 counts on fixed AzooKey candidates.

| Model | Validation BPC ↓ | AJIMEE / 200 | Historical development / 137 | Draft development / 2,000 |
| --- | ---: | ---: | ---: | ---: |
| V1 FP32 | 3.4736559 | 124 | 122 | 1,453 |
| V2.0 FP32 | 3.2598221 | 125 | 124 | 1,463 |
| V2.1 restart1 FP32 | 3.0811788 | 137 | 121 | 1,476 |
| V2.1 release FP32 | 3.0802686 | 144 | 122 | 1,487 |
| V2.1 release INT8 | Not measured | 144 | 122 | 1,481 |

Quantization preserves the two small-set hit counts but reduces draft-set Top-1 by 0.3 percentage points. Strict elementwise logits alignment fails for the INT8 model; task-level measurements support its deployment under the recorded conditions, without establishing numerical equivalence. The 2,000-case labels remain drafts. The newer [Standard Japanese IME benchmark](docs/benchmarks/standard-ime.md) uses frozen AI-reviewed labels, not native-speaker gold annotations. Full methods, limitations, and V3 results appear in [evaluation](docs/evaluation.md).

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

This command assumes the matching local training artifacts and their manifests are present. The default inference CLI selects V1. Hugging Face inference bundles have separate packaged-runtime instructions in their model cards; those bundles are not optimizer checkpoints. Core ML conversion and runtime evaluation require macOS. The current iOS application is maintained in the separate Vime repository; `examples/ios/` contains historical integration references.

## Repository structure

| Path | Contents |
| --- | --- |
| `src/vimeml/` | Corpus processing, tokenization, training, evaluation, and deployment implementations |
| `scripts/` | Command-line entry points organized by subsystem |
| `configs/` | Versioned experiment and data-processing configurations |
| `annotations/` | Versioned corpus review decisions |
| `docs/` | Primary English documentation and dated experiment records |
| `docs/zh-CN/` | Chinese documentation snapshots and reference material |
| `examples/ios/` | Core ML probe, SentencePiece bridge, and historical client examples |
| `templates/` | Artifact exchange templates |
| `tests/` | Existing targeted implementation tests |
| `datasets/`, `artifacts/`, `outputs/`, `runs/`, `handoff/` | Local, Git-ignored data, model artifacts, results, and transfer bundles |

## Documentation and licensing

[Documentation index](docs/index.md) · [Model card](MODEL_CARD.md) · [Chinese overview](README.zh-CN.md) · [Contribution conventions](CONTRIBUTING.md) · [Licensing and attribution](NOTICE.md)

Project code and published model weights use GPL-2.0. Upstream libraries and source datasets retain their own licenses. Private research benchmark text is not included in the public repository or model releases. Third-party notices remain with the corresponding components.
