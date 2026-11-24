# VimeML

Compact causal language models for the Vime input method.

- **Chinese.** The v2 models generate sentences directly from pinyin keys, rerank librime (rime-ice) candidates and predict the next phrase. One model does all three jobs.
- **Japanese.** The models rerank AzooKey kana-to-kanji candidates using left context.

These are custom PyTorch models for input methods. They are not instruction-tuned assistants and not Transformers `AutoModel` integrations.

[中文说明](README.zh-CN.md) · [Documentation index](docs/index.md) · [Model card](MODEL_CARD.md) · [Licensing and attribution](NOTICE.md)

## Releases

| Model | Language | Parameters | Use | Hugging Face |
| --- | --- | ---: | --- | --- |
| **vimeml-zh-v2-20m** | Chinese | 20,036,736 | iPhone: pinyin generation, Rime rerank, prediction (Core ML INT8) | [Voltline/vimeml-zh-v2](https://huggingface.co/Voltline/vimeml-zh-v2) `20m/` |
| **vimeml-zh-v2-200m** | Chinese | 193,625,088 | Desktop (PyTorch, GPU): the same three jobs | [Voltline/vimeml-zh-v2](https://huggingface.co/Voltline/vimeml-zh-v2) `200m/` |
| vimeml-tiny-zh-v1 | Chinese | 20,036,736 | iPhone: Rime rerank and prediction; fallback for v2 | [Voltline/vimeml-tiny-zh-v1](https://huggingface.co/Voltline/vimeml-tiny-zh-v1) |
| **V4.2** | Japanese | 15,070,848 | iPhone: AzooKey rerank (deployed) | [Voltline/vimeml-tiny-ja-v4.2](https://huggingface.co/Voltline/vimeml-tiny-ja-v4.2) |
| V2.1 | Japanese | 12,537,920 | Previous deployment | [Voltline/vimeml-tiny-ja-v2.1](https://huggingface.co/Voltline/vimeml-tiny-ja-v2.1) |
| V1 | Japanese | 7,386,624 | Baseline | [Voltline/vimeml-tiny-ja-v1](https://huggingface.co/Voltline/vimeml-tiny-ja-v1) |

Each release holds inference weights (no optimizer state), the tokenizer, Core ML packages, a frozen copy of the source, and aggregate evaluation results. Training checkpoints, corpora and private benchmark text are not published.

## Chinese

### v2: pinyin to sentence

| Model | Target | Status |
| --- | --- | --- |
| `vimeml-zh-v2-20m` | iPhone | Released 2026-11-23 (`colloq-c40`) |
| `vimeml-zh-v2-200m` | Desktop | Released 2026-11-23 (`200m/`; public pinyin 54.7%) |
| `vimeml-zh-v2-800m` | Desktop | Confirmed on test 2026-11-24 |

All v2 models share v1's 12,288-piece tokenizer. The prompt is the text before the cursor (up to 64 characters), then the keys: full pinyin, initials, or a mix. Beam search keeps 16 hypotheses per key position, and a character is allowed only if its reading or initial matches the next keys. The top 10 sentences are rescored by the same model in language-model mode.

**vimeml-zh-v2-20m** started from v1, then went through pinyin-task training, distillation from the 200M, and colloquial post-training ([lineage and method](https://huggingface.co/Voltline/vimeml-zh-v2)). Every number below comes from a test split opened once under a preregistered plan. The metric is Top-1 exact match of the first candidate shown.

| Test set | Shipped v1 path (Rime + v1 rerank) | `kd-cont` | **v2-20m** |
| --- | ---: | ---: | ---: |
| `ime-pinyin-zh-v1-r1` test, 3,436 inputs | 31.0% | 49.1% | **51.4%** |
| … full pinyin / mixed / all initials | 81.6 / 23.7 / 9.2 | 82.6 / 47.4 / 22.2 | **84.4 / 49.8 / 24.7** |
| `ime-pinyin-zh-colloq-v1-r1` test, 4,560 colloquial inputs | — | 39.0% | **42.3%** |

On the Standard blind full-pinyin rerank of Rime candidates, v2-20m scores 94.3% against v1's 93.8% (net +7, p 0.31). Reports: [v2-20m confirmation](docs/reports/zh-v2-confirmation-20261120/report.md), [colloquial confirmation](docs/reports/zh-colloq-confirmation-20261123/report.md), [release scope](docs/release-zh-v2-plan.md).

On the phone, v2-20m runs as one batched-KV Core ML INT8 package (23.3 MB). In the Vime client it reranks Rime candidates and predicts by default; sentence generation is behind a development switch until key-by-key generation is fast enough on older phones. Measured with `kd-cont`, which has the same shape and package: Rime rerank p50 4.1 ms on an iPhone 13 Pro and 3.0 ms on an iPhone 16 Pro Max; one-shot generation 91 and 59 ms ([iPhone report](docs/reports/mac-20261121/zh-v2-20m.md), [c40 Mac result](docs/reference/mac-zh-v2-20m-colloq.md)).

The desktop models run in PyTorch. With CUDA graphs in FP16 on a laptop GPU, the 200M takes 83/205 ms p50/p95 per input, including search and rescoring ([desktop latency](docs/reports/zh-desktop-latency-20261118/report.md)). On the public pinyin test it reaches 54.7% (full pinyin 86.5%), and its Standard rerank is 95.2% against v1's 93.8% ([200M confirmation](docs/reports/zh-v2-200m-confirmation-20261123/report.md)). The 800M (second architecture, chosen over the first on development: [arch2 evaluation](docs/reports/zh-800m-arch2-eval-20261123/report.md)) reaches 63.3% on the same test (full pinyin 88.7%), 8.6 points above the 200M, with a Standard rerank of 95.3% ([800M confirmation](docs/reports/zh-v2-800m-confirmation-20261124/report.md)).

### v1: candidate reranking

`vimeml-tiny-zh-v1` reranks the pinyin candidates of librime with rime-ice using left context. It was trained from random initialization on 22.4B characters of cleaned FineWeb2 `cmn_Hani` web text and LCCC dialogue. Internal code and paths use the codename `zh13`. The Hugging Face release includes the stateless Core ML package and the batched KV packages up to v4 (shared prefix, one multifunction package).

Blind exact Top-1 on `ime-standard-zh-v1-r1` (CrossWOZ, RiSAWOZ, KdConv dialogue turns; AI-reviewed labels). The models rerank a frozen 20-candidate pool from rime-ice:

| System | Full pinyin / 1,445 | Abbreviated / 964 |
| --- | ---: | ---: |
| **vimeml-tiny-zh-v1** | **93.8%** | **35.0%** |
| Qwen2.5-0.5B, same pool and scoring | 91.8% | 31.0% |
| Qwen3-0.6B-Base, same pool and scoring | 89.2% | 30.6% |
| rime-ice + Wanxiang grammar model | 82.2% | 15.7% |
| rime-ice (candidate pool source) | 72.2% | 14.6% |

The preregistered confirmation against Rime passed (+329/−19, p = 3.6e-74). Reports (Chinese): [plan](docs/zh-CN/training-zh-v1-plan.md), [training](docs/zh-CN/reports/zh-v1-20261110/training.md), [blind and baselines](docs/zh-CN/reports/zh-v1-20261111/blind-and-baselines.md), [iOS release](docs/zh-CN/reports/zh-v1-20261111/ios-release.md).

## Japanese

V4.2 epoch 20 is the deployed model: INT8 block-32 weights, FP32 compute, CPU-only Core ML on iOS 18, a 17.16 MB package. It replaced V2.1 `extend5` step 40,000.

Exact Top-1 on fixed AzooKey candidate pools (Standard labels are AI-reviewed):

| Model | Standard dev / 728 | Fixed reference / 725 | Legacy / 500 | AJIMEE / 200 |
| --- | ---: | ---: | ---: | ---: |
| V2.1 FP32 | 505 | 535 | 374 | 144 |
| V4.2 FP32 | **523** | 541 | 370 | 139 |
| V4.2 INT8 (deployed) | 522 | — | 371 | 139 |
| V5 epoch 30 | 514 | — | 365 | **145** |

V4.2 and V2.1 are not significantly different. On the preregistered Standard v2 blind, V4.2 scores 598/729 against 576 (author-cluster p = 0.0508, just short of the threshold). V3, V4-B, V4.1, V4.3, V5 and the WSD runs are completed experiments that did not replace V4.2. V5 lowers validation BPC by 0.10 without moving IME Top-1. V4.4 measured the seed noise band (Standard 501–523), and a four-seed ensemble reached 619 on the blind.

The Japanese line was paused for the Chinese work and resumes next. Details: [evaluation](docs/evaluation.md), [V4.2 training](docs/training-v42.md), [V4.2 Core ML](docs/reports/mac-20261020/v42-coreml.md), [V4.4 ablation](docs/training-v44.md), [confirmation results (Chinese)](docs/zh-CN/benchmarks/confirmation-v2-results-20261021.md).

## Installation

Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e .
```

`requirements.txt` pins the platform-specific versions. `requirements/` adds the Linux CUDA training environment (`cuda.txt`), benchmark preparation (`evaluation.txt`) and external-model comparisons (`external-eval.txt`).

To run a released model, use the Hugging Face bundle on its own: each one includes its source and an `infer.py` (for v2, `python infer.py --keys woxiangchifan --context 今天中午`). The source repository runs from local checkpoints and their manifests. Core ML conversion and runtime checks need macOS. The iOS client lives in the separate Vime repository.

## Repository structure

| Path | Contents |
| --- | --- |
| `src/vimeml/` | Corpus processing, tokenization, training, evaluation and deployment |
| `scripts/` | Command-line entry points by subsystem |
| `configs/` | Versioned experiment and data configurations |
| `requirements/` | Extra dependency pins |
| `docs/` | English documentation and dated experiment records |
| `docs/zh-CN/` | Chinese documentation; the reference text for plans registered in Chinese |
| `examples/ios/` | Core ML probe, SentencePiece bridge, historical client integration |
| `templates/` | Artifact exchange templates |
| `tests/` | Targeted tests |
| `datasets/`, `artifacts/`, `outputs/`, `runs/`, `handoff/` | Local, Git-ignored data, models, results and transfers |

## License

Code and published weights use GPL-3.0-or-later, with an additional permission under section 7 for app-store distribution ([LICENSE-EXCEPTION](LICENSE-EXCEPTION)). Releases before 2026-10-22 used GPL-2.0-only. Datasets and third-party components keep their own terms; [NOTICE.md](NOTICE.md) lists the attribution that applications bundling a model must show. [Contribution conventions](CONTRIBUTING.md).
