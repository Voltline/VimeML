---
language:
- ja
license: gpl-2.0
pipeline_tag: text-generation
tags:
- pytorch
- coreml
- japanese
- causal-lm
- contextual-reranking
- tiny-gpt
---

# VimeML Tiny Japanese GPT v1

A frozen **7,386,624-parameter**, decoder-only Japanese language model for local contextual candidate reranking and short within-sentence continuation. This repository provides the FP32 inference bundle and an iOS18 INT8-weight Core ML variant. SentencePiece tokenization, kana candidate retrieval, scoring and search run outside the Core ML graph.

This is a custom PyTorch model, not a Transformers `AutoModel` integration or an instruction/chat model. The source model, loader, scoring and conversion code are included under `source/`; no `trust_remote_code` or training checkpoint is needed.

## Files and identity

| Path | Contents |
| --- | --- |
| `inference/` | Unique FP32 state_dict, config, original SentencePiece model, token manifest and provenance manifest; no optimizer |
| `coreml/ios18-int8-block32/` | Original Core ML `.mlpackage` and manifest |
| `source/` | Frozen VimeML Python source, conversion scripts, configuration and unified requirements |
| `infer.py` | Read-only wrapper around the original fingerprint-checked loader, generation and scoring |
| `evaluation/summary.json` | Curated results and source report hashes, without benchmark texts or device traces |
| `RELEASE.json`, `SHA256SUMS.txt`, `verify_release.py` | Release inventory and local byte verification |

- Source repository: [Voltline/VimeML](https://github.com/Voltline/VimeML), commit **@@SOURCE_COMMIT@@**.
- Inference manifest SHA256: `@@BUNDLE_SHA@@`.
- Core ML manifest SHA256: `@@COREML_SHA@@`.
- Tokenizer SHA256: `@@TOKENIZER_SHA@@`.
- Original training checkpoint SHA256: `@@CHECKPOINT_SHA@@`; the training checkpoint is not distributed here.

The original manifests remain byte-for-byte intact. Their old absolute paths describe provenance, not paths required on a downloader's machine. Core ML conversion scripts evolved after the original experiment; the old manifest's script hash is retained.

## Architecture and tokenization

Pre-LayerNorm GPT with GELU, learned absolute positions and causal attention: vocabulary 16,384; context length 128; width 256; 4 layers; 4 heads; FFN width 1024. Input embedding and LM head share one parameter. There is no KV cache or sliding context. Each generation step recomputes the entire prefix.

SentencePiece 0.2.1 unigram, identity normalization, whitespace preservation and byte fallback. PAD/UNK/BOS/EOS IDs are 0/1/2/3. The exact original `tokenizer.model` is required; retraining a tokenizer does not preserve this vocabulary binding.

## Quick start: downloaded files

Python ≥3.11. After downloading the repository, install the included unified requirements. Select the appropriate PyTorch build first if needed; do not use the PyTorch-only package index for the whole requirements file. The Darwin branch pins the original Core ML tool environment; no Core ML prediction occurs during FP32 inference.

```sh
python -m pip install -r source/requirements.txt
python verify_release.py
python infer.py --prompt "今日は雨が降っているので、" --max-new-tokens 8
python infer.py --prompt "雨が降っているので、" --candidate "傘を持っていく" --candidate "橋を渡っていく"
```

The second inference command ranks the supplied strings using the original contextual suffix logP sum. It does not perform kana conversion or dictionary lookup. Invalid/overlong candidates raise in this diagnostic wrapper; the application implementation retains the dictionary order on eligibility failure.

The FP32 loader checks all bundle files and the included source fingerprints, restores the shared head, and uses `torch.load(..., weights_only=True)`. Keep the source, tokenizer and inference files together. `verify_release.py` checks every release file without loading a model; checksums detect altered bytes but are not a digital signature.

## Candidate scoring

Encode `context + candidate` jointly. Cancel the common token prefix across BOS+context and the jointly encoded candidate sequences, then sum original full-vocabulary log-softmax values over the remaining suffix. Do not add a candidate-final EOS. Ties preserve the original order. This token suffix likelihood is a scoring proxy, not exact string conditional probability.

Deployment uses **pure LM scoring**; AzooKey retrieves kana candidates. The older FP32 `AzooKey score + 2 × LM sum` policy is a historical comparison, not the INT8 deployment policy. Quantization does not establish that λ=2 remains optimal.

## Training and evaluation

Training used nine local shards of [FineWeb2-Edu Japanese](https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese) and Japanese [Tatoeba](https://tatoeba.org/), cleaned, split into sentences, exactly deduplicated and grouped across 98%/1%/1% train/validation/test splits. There were 25,713,003 unique sentences and 25,185,368 training sentences. The tokenizer learned from 2 million sampled training sentences.

One data pass, 196,853 updates and 573,295,237 prediction targets. The frozen best checkpoint's complete validation loss is 4.5415708951, perplexity 93.8380942. Fixed-subset validation selected the checkpoint; the test split was not used for training, checkpoint selection or tuning. No test perplexity is claimed.

Reranking uses fixed N-best 20 AzooKey candidates, without injecting correct answers. [AJIMEE JWTD_v2/v1](https://github.com/azooKey/AJIMEE-Bench) has 200 cases and allows multiple accepted spellings. The synthetic development set has 137 reviewed cases.

| Frozen pool | AzooKey original Top-1 | FP32 pure LM Top-1 / Top-5 | INT8 pure LM Top-1 / Top-5 | Pool coverage |
| --- | ---: | ---: | ---: | ---: |
| Development | 111/137 | 122/137 / 134/137 | 122/137 / 134/137 | 136/137 |
| AJIMEE | 87/200 | 124/200 / 151/200 | 124/200 / 151/200 | 162/200 |

INT8 vs Windows FP32: development changes 119 complete rankings and 0 first choices; AJIMEE changes 163 complete rankings and 2 first choices, both between acceptable spellings. The original development compression report instead compares against uncompressed conservative Core ML: 118 complete rankings change. Equal hit counts do not mean equal scores or complete rankings. See the release summary for the separate baselines and hashes.

## Core ML variant

**Weight-only symmetric INT8, block size 32; FP32 computation; CPU_ONLY; minimum iOS 18.** It is not INT8 activation or arithmetic inference. Logical `.mlpackage` size is **8,077,801 bytes**. This is not compiled size, IPA size or resident memory.

Input `input_ids`: INT32 `[1,T]`, 1≤T≤128. Output `logits`: FLOAT32 `[1,T,16384]`, full raw vocabulary logits. Only right-side PAD is supported. Shared embedding/head are retained. Use `.cpuOnly`; GPU/ALL paths for this package triggered a quantized-gather assertion in the client experiments.

On Mac, compile manually for the target platform before integrating:

```sh
xcrun coremlcompiler compile coreml/ios18-int8-block32/model.mlpackage compiled-int8 --deployment-target 18.0
```

The INT8 model **fails the strict FP32 logits alignment gate** (maximum absolute difference about 3.66). Candidate fixtures, causal and right-PAD checks passed, and the frozen pool hit counts above were preserved. Do not describe the model as numerically identical to FP32.

An AJIMEE-text diagnostic reported mean KL(FP32‖INT8)≈0.00589, mean NLL 5.56698→5.57289 and position top-choice agreement 94.4%. These are means of per-example position means, not a corpus token-weighted aggregate. Text content was capped at 127 tokens plus BOS; the old report did not record truncation counts. Similar aggregate KL to a simulated INT8 recipe does not prove exact conversion equivalence. Tested palette4 group16 degraded substantially and is not included.

## Performance evidence and limits

Recorded iPhone 16 Pro Max optimized CPU_ONLY **test-host App** results: approximately 14 candidate reranking p50 10.4ms/p95 12.7ms; next-word p50 5.8ms; T128 prediction 3.0ms. Host memory increased by about 24.2MB and peaked at 74.8MB. These are not keyboard-extension latency or memory measurements.

A separate 152.7 second manual real-keyboard recording produced 127 approximately 1 second samples: physical footprint peak 21.25MiB and private resident peak 51.922MiB. The user confirmed LM ranking and next-word suggestions; installed build/model hashes were not independently captured. Short spikes may have been missed, the tail still grew slowly, and long-session stability is not certified. The different memory metrics cannot be substituted for one another. No energy claim is made.

## Intended use and limitations

Japanese within-sentence reranking and short completion experiments, primarily for on-device input methods. The model is small, not instruction-tuned, and can repeat text, drift, truncate or invent facts. Outputs need product-level review; the fixed beam suite had only 10/20 prefixes with clearly natural suggestions in an AI review. These offline pool metrics do not measure application-wide accuracy or guarantee keyboard performance.

Semantic near-duplicates and training overlap with public evaluation text have not been comprehensively audited. AJIMEE has been used for error analysis and is not a fresh blind test. Development labels and review came from AI, not native-speaker adjudication. Official benchmark context can cross sentences, while the client uses current-sentence context. Candidate coverage limits reranking accuracy.

## License and attribution

Model weights and the included VimeML code are released under **GNU GPL v2.0**, chosen by the author. See `LICENSE` and `source/LICENSE`. Training data and third-party software retain their own terms; they are not relicensed by this model repository.

FineWeb2-Edu Japanese's dataset card declares ODC-BY; [Tatoeba text terms](https://tatoeba.org/en/terms_of_use) default to CC-BY 2.0 FR with author attribution requirements. No corpus text is redistributed here. AJIMEE's source data is CC-BY-SA 3.0; this release distributes aggregate metrics, not its evaluation text. SentencePiece runtime code is not vendored into this release; installed dependencies retain their upstream licenses. Source configuration and training/conversion implementations are included, while datasets, API responses, optimizer state, device traces and the separate Vime client are excluded.
