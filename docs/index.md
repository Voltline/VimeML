# Documentation index

Primary documents are English; [Chinese documentation](zh-CN/index.md) mirrors them under `zh-CN/` at the same paths. Where a preregistered plan was registered in Chinese, the Chinese file is the registered text and the English page is a translation.

**Current status (2026-11-23).**

- **Chinese.** `vimeml-zh-v2-20m` (`colloq-c40`) is released in [Voltline/vimeml-zh-v2](https://huggingface.co/Voltline/vimeml-zh-v2): test Top-1 51.4% on public pinyin and 42.3% on colloquial pinyin, against 31.0% for the shipped v1 path. In the Vime client it reranks and predicts by default; sentence generation is behind a development switch. vimeml-tiny-zh-v1 stays released as the fallback.
  - `vimeml-zh-v2-200m` (desktop) passed its test confirmation on 2026-11-23: public pinyin 54.7%, Standard rerank 95.2%; it joins the v2 repository next.
  - `vimeml-zh-v2-800m` (desktop): the first architecture is evaluated (65.3% development Top-1); the second architecture is in training.
- **Japanese.** V4.2 epoch 20 / INT8 block-32 is deployed; it replaced V2.1 `extend5` step 40,000, and V1 is the baseline. V4.4 measured the seed noise band; V4.5 distillation is registered but not run. The Japanese line resumes next.

## Model, data, and runtime

| Document | Scope |
| --- | --- |
| [Model card](../MODEL_CARD.md) | Architecture, intended use, releases, metrics, and limitations |
| [Data](data.md) | Frozen Japanese corpora, cleaning, splits, and provenance |
| [Evaluation](evaluation.md) | Fixed-pool scoring, statistics, and benchmark roles |
| [Core ML](coreml.md) | Conversion, quantization, client contract, and V2.1/V4.2 measurements |
| [Artifact layout](artifacts.md) / [cross-platform exchange](reference/artifact-exchange.md) | Local assets, Git collaboration, and ZIP transfer |

## Chinese line

### Releases and data

| Document | Scope |
| --- | --- |
| [vimeml-zh-v2 release scope](release-zh-v2-plan.md) | `vimeml-zh-v2-20m` (mobile, released 2026-11-23), `vimeml-zh-v2-200m` and `vimeml-zh-v2-800m` (desktop), all 12K with pinyin generation; 200M six arms and 20M joint pretraining are research only; release gates; data compliance applies to 20M only |
| [vimeml-tiny-zh-v1 corpus](data-zh-v1.md) | Frozen as export-r1 (2026-11-10): FineWeb2 cmn_Hani (21 shards) and LCCC; Traditional-to-Simplified, rule filter, spam-site removal, review queue |
| [Chinese tokenizer vocabulary probe](reports/zh-v1-20261106/tokenizer-vocab.md) | Tokenizer-only comparison of 8K–32K Unigram vocabularies: 20M compares 8K and 12K; 16K–32K deferred to 200M |

### Benchmarks and confirmations

| Document | Scope |
| --- | --- |
| [Standard Chinese IME benchmark](benchmarks/standard-ime-zh.md) | CrossWOZ/RiSAWOZ/KdConv, 2,901 items, full and abbreviated pinyin tracks, librime + rime-ice candidate pools |
| [Public pinyin-generation test set ime-pinyin-zh-v1](benchmarks/pinyin-ime-zh.md) | Frozen 2026-11-14: CrossWOZ, RiSAWOZ, KdConv, CPED; 578/576 sentences, full, mixed and all-initials keys; screened against export-r1; test held back |
| [Prediction benchmark ime-predict-zh-v1](benchmarks/predict-ime-zh.md) | Frozen 2026-11-11: CrossWOZ, RiSAWOZ, KdConv, CPED, 1,200 dialogues each; blind closed |
| [Public Chinese keyboard-typo set](benchmarks/typo-ime-zh.md) | `ime-typo-zh-v1`: 578 public development sentences in clean, one-typo and two-typo full pinyin; candidate pool frozen from v1; LM vs task-mode display rule, @2 and clean false display, calibrated margin |
| [Blind confirmation plan](benchmarks/confirmation-zh-v1-plan.md) / [baseline plan](benchmarks/baselines-zh-v1-plan.md) | Preregistrations: `12k-fresh-s1` vs Rime on blind, run once; rime-ice, rime-ice + grammar, Luna Pinyin, Qwen2.5-0.5B, Qwen3-0.6B-Base as references |
| [Blind confirmation and baselines](reports/zh-v1-20261111/blind-and-baselines.md) | Blind Top-1 93.8% vs Rime 72.3%: confirmed; ahead of Qwen2.5-0.5B (91.8%), Qwen3-0.6B (89.2%), rime-ice + grammar (82.2%), Luna Pinyin (68.9%) |
| [vimeml-zh-v2 test confirmation plan](benchmarks/confirmation-zh-v2-plan.md) | Preregistration: v2-20m (`kd-cont`) vs the online path (Rime 16 + v1 rerank) on the public pinyin test split, with full-pinyin and Standard guards; v2-200m vs v2-20m; each run once. v2-20m passed on 2026-11-20 ([report](reports/zh-v2-confirmation-20261120/report.md)); `colloq-c40` later replaced `kd-cont` as v2-20m; v2-200m passed on 2026-11-23: 49.1 → 54.7 (+343/−150, p 5e-5), Standard net +21 over v1 (p 0.002) ([report](reports/zh-v2-200m-confirmation-20261123/report.md)) |
| [Colloquial candidate test confirmation plan](benchmarks/confirmation-zh-colloq-plan.md) / [result](reports/zh-colloq-confirmation-20261123/report.md) | Preregistration: colloq-c40 vs `kd-cont` on the colloquial pinyin test split, with public pinyin (second use), full-pinyin and Standard guards. Passed 2026-11-23: colloquial 39.0 → 42.3 (+217/−69, p 5e-5), public 49.1 → 51.4; c40 becomes v2-20m |

### 20M line: v1 to v2-20m

| Document | Scope |
| --- | --- |
| [vimeml-tiny-zh-v1 experiment plan](training-zh-v1-plan.md) | Preregistration: eight arms at 22.4B characters, fresh data vs 5 vs 20 epochs crossed with 8K vs 12K, three seeds of 12K fresh |
| [vimeml-tiny-zh-v1 training record](reports/zh-v1-20261110/training.md) | All eight arms completed; final-checkpoint BPC and the memorization probe |
| [vimeml-tiny-zh-v1 IME benchmark and judgement](reports/zh-v1-20261110/ime-benchmark.md) | Development full-pinyin Top-1 93.3%–94.4% vs 68.8% for Rime; no contrast effective, defaults apply (fresh data, 12K) |
| [zh three-task round-2 training](reports/zh-pinyin-r2-20261114/training.md) / [evaluation](reports/zh-pinyin-r2-20261114/evaluation.md) | Continue (60k) and restart (90k) arms on the v2 pinyin stores. Private-benchmark generation Top-1 29.9% beats Rime 16 + v1 rerank (26.0%, p 5e-5); rerank −24 on Standard and prediction −2.8 Top-3 vs v1; KV generation p50 92 ms on CPU. Public recheck: +8.0 overall, but −4.0 on full pinyin (p 0.02), so route by input kind |
| [zh three-task arm 3: joint pretraining](reports/zh-pinyin-pt-20261115/record.md) | 2026-11-15 20M trained from scratch with pinyin mixed into pretraining: rerank only −4 (n.s.), prediction +1.1 over r2-cont, but generation −2.2 private and 37.8% vs 39.6% public; joint pretraining does not suit the 20M, keep continuation |
| [Inference-time rescoring for pinyin generation](reports/zh-pinyin-rescore-20261115/report.md) | Rerank the generated top 10 by genScore + 1.6 × v1 LM score: r2-cont 29.9% → 36.4% private, 39.5% → 51.3% public; full pinyin now significantly above production (+4.2 public), so no routing by input kind |
| [Beam-width sweep for Vime's pinyin search](reports/zh-pinyin-beam-20261115/report.md) | v1 LM mode with context, beam per key position: Top-1 at beam 16/12/8/4 is 52.5/51.6/49.1/44.1 public, 33.6/32.9/32.0/29.2 private; every drop significant, loss on abbreviated input; default stays 16; v1 rescoring adds nothing in this mode |
| [v2-20m distillation plan](training-zh-kd-plan.md) | The 12K joint-pinyin 200M as teacher, online distillation into the 20M (α 0.5, T 1); arms kd-base and kd-cont; decided by self-rescored Top-1 against v1 |
| [Development evaluation of the distillation arms](reports/zh-kd-eval-20261117/report.md) | 2026-11-17 self-rescored public pinyin Top-1: kd-cont 54.8 (+2.1 over v1, p = 0.041, pass), kd-base 53.9 (+1.2, n.s.); Standard ties v1; prediction still below v1 (A Top-3 25.6 vs 27.6); private set agrees, full pinyin +4.4 over production v1; kd-cont is the v2-20m candidate |
| [Development evaluation of the round 3 distillation arms](reports/zh-kd-r3-eval-20261117/report.md) | 2026-11-17 self-rescored public pinyin Top-1: r3-plain 54.8, r3-data 54.7, r3-ctrl 54.6 tie kd-cont (54.8), r3-alpha (α 0.9) 53.5 is significantly lower; prediction A Top-3 25.8–26.5, below v1's 27.6; Standard ties v1; kd-cont stays the v2-20m candidate, round 4 swaps in the 800M teacher |
| [Keyboard typo plan](training-zh-typo-plan.md) | New `typo` pinyin kind (26-key edits on public text), task-mode scoring against the typed keys, per-export calibration, public `ime-typo-zh-v1` benchmark; 20M with/without arms; result 2026-11-19: the arm does not pass, LM rule kept ([report](reports/zh-typo-eval-20261119/report.md)) |
| [v2-20m colloquial plan](training-zh-colloq-plan.md) | Continue `kd-cont` with screened colloquial text (export-r1 chat paragraphs, Douban, MMChat, Tieba) via `[[data.extra]]`; arms ctrl / c10 / c20 / c10-hard (colloquial on cross-entropy alone); decided by the colloquial pinyin set `ime-pinyin-zh-colloq-v1` against ctrl, gated on public pinyin and Standard |

### Desktop models: 200M and 800M

| Document | Scope |
| --- | --- |
| [Prediction benchmark and desktop roadmap](benchmarks/predict-zh-plan.md) | Preregistration: `ime-predict-zh-v1`, a public-model scale probe, then gated 200M and 800M desktop models |
| [Prediction step-0 scaling survey](reports/zh-predict-step0-20261115/survey.md) | Qwen2.5 0.5B/1.5B/3B rise monotonically on all four tracks, 3B vs 0.5B track A Top-3 +7.18; v1 and zh-200m are significantly below 0.5B; 800M not deferred |
| [zh-200m training plan](training-zh-200m-plan.md) | Preregistration: d1024, 16K/24K/32K at context 512 plus a 24K context-128 control, three 24K seeds |
| [zh-200m training record](reports/zh-200m-20261113/training.md) | All six arms finished 2026-11-13; web BPC 3.8251–3.9153 vs 4.3533 for zh-v1; window-index, context-cap, and batch-policy fixes |
| [zh-200m prediction and rerank judgement](reports/zh-200m-20261114/judgement.md) | Track A Top-3 0.2999–0.3164 (seed range 0.0165); no contrast effective, defaults apply (24K, context 512); rerank gate passed; Top-1 verbatim-in-training rate no higher than 20M |
| [Development evaluation of the 12K joint-pinyin 200M](reports/zh-200m-pinyin-eval-20261116/report.md) | 2026-11-16 public pinyin Top-1 58.8% rescored with its own prior (v1 52.7, +6.2; 57.3 with the v1 prior); Standard full Top-1 95.2 vs 93.6 (p = 0.0002); best on all four prediction tracks; 200M stays on joint pretraining with its own scores; r2-cont self-rescored only +0.8 over v1 (n.s.), v2-20m waits for distillation |
| [Desktop latency, PyTorch on Windows](reports/zh-desktop-latency-20261118/report.md) | 2026-11-18/19: full pipeline (search + rescoring) p50/p95 with CUDA graphs in FP16: 200M 83/205 ms, kd-cont 34/68, v1 32/63; FP16 halves the 200M, graphs carry the 20M |
| [zh-800m training plan](training-zh-800m-plan.md) | Preregistration: d1536/L27, 784M parameters, 12K tokenizer, context 512; all of export-r1 + export-r2 in one pass, LCCC 2%, 2048 windows per update |
| [zh-800m arch1 evaluation](reports/zh-800m-eval-20261120/report.md) | 2026-11-20, development only: prediction track A Top-3 0.3394 (200M 0.3164), rescored pinyin Top-1 65.3% (200M 57.3%, v1 LM mode 52.6%), Standard net +26 over v1; no long verbatim recall |
| [zh-800m second architecture (arch2)](training-zh-800m-arch2-plan.md) | Formal training at k = 4 with arch1's LR (3e-4 → 3e-5, set by the precheck control round): d1792/L23, Q 28 × 64 + KV 7 × 64 (GQA), RoPE, QK-norm, 777M parameters; paragraph windows packed into rows of 2048 with per-segment causal masks, same data and order as arch1, updates 4× larger (set by a 200M precheck; 8× not tried), no FP8; locate the bottleneck (communication/host/compute) first; target one pass under 1.5 days at no loss of quality |

### Core ML and deployment

| Document | Scope |
| --- | --- |
| [Mac results check and iOS release](reports/zh-v1-20261111/ios-release.md) / [Core ML report](reports/mac-20261111/zh-v1-coreml.md) | INT8 b32 vs FP32 +2/−1 full pinyin; about 1.3× V4.2 on the Mac; Hugging Face release |
| [zh-v1 batched KV decoding on Core ML](reports/mac-20261115/zh-v1-kvbatch.md) | 2026-11-15: one call advances a whole beam bucket; INT8 flex B 1..16 within 4.8e-5 of stateless INT8; M3 Mac p50 B=16 3.35–3.76 ms, B=8 2.69–2.92 ms; beam search equals the stateless-INT8 search 125/125; added to the Hugging Face release copy; iPhone not yet measured |
| [zh-v1 batched KV decoding v2 on Core ML](reports/mac-20261115/zh-v1-kvbatch-v2.md) | 2026-11-15: read-only caches, multi-token prefill and a scored head; `unified-scored` chosen: one package for prompt and decode (no chunk package), Core ML footprint ~40 MiB as v1, T=1 graphs ≥200 MiB; M3 Mac p50 B=16 ≤4.4 ms, search p50 79 ms; search equals v1 125/125; added to the Hugging Face release copy; iPhone not yet measured |
| [zh-v1 sentence scoring on Core ML (sequence head)](reports/mac-20261116/zh-v1-kvsequence.md) | 2026-11-16: log-prob of a given token at every new position, so a batch of up to 16 known continuations is one call; alignment max 3.34e-5; M3 Mac B=16 T=16 one call 17.5–18.4 ms vs 54.5–62.2 ms for 16 decode calls (3.1–3.4×); memory grows with T (T ≤ 8 as `unified-scored`, T=16 +9 MiB, T=64 +63 MiB), cap T at 8 or 16 per call; fixture ranking equals FP32 8/8; an extra resource, search keeps `unified-scored`; added to the Hugging Face release; iPhone not yet measured |
| [zh-v1 batched KV v3 on Core ML (shared prefix, one package)](reports/mac-20261118/zh-v1-kvbatch-v3.md) | 2026-11-18: one multifunction INT8 package (`logits`, `scored`, `score`, 23.1 MB, one weight copy), prompt passed once as a shared prefix, scoring returns target log-probs only; INT8 gate max 4.15e-5; search equals v2 125/125; M3 Mac B=16 decode 2.9–3.3 ms (v2 3.0–3.9), scoring T=16 16.1–17.1 ms (sequence 17.2–17.8); memory peak 18.5/31.4/56.8 MiB at T 16/32/64 vs 40.5/70.2/150.0 for stateless + v2 + sequence, ~4.5 MiB retained vs 16–57; 4 vocab chunks rejected (higher peak, +5.3 MB); stateless package can go after a device check; added to the Hugging Face release; iPhone not yet measured |
| [zh-v1 Neural Engine probe (step 0)](reports/mac-20261118/zh-v1-ane-probe.md) | 2026-11-18: no-go. M3 Mac, FP16 fixed-shape graphs over the v3 inputs. Scoring on the ANE is 6.9–7.4× faster than v3 INT8 on CPU (B16×T16 2.18 vs 16.11 ms), and decode is 1.4–2.0× faster. Blocker: the process peak reaches ~150 MiB at model load (v3: 15.8 MiB). Top-1 132/133; the one miss is an FP32 near tie (0.033 nats). Flipped pairs 41 vs v3's 92. Per-block INT8 does not run on the ANE. First ANE compile takes 14–44 s. FP16 on CPU keeps 124 MiB resident |
| [zh batched KV v4 on Core ML (`prefix` function, FP16 KV)](reports/mac-20261120/zh-kvbatch-v4.md) | 2026-11-20: v3 plus a `prefix` function without the head; `fp32kv` chosen; iPhone 16 Pro Max `score>prefix` +7.5 MB steady vs +34.8 MB for `score>scored` (the ~27 MB comes with the second head function and returns once `scored` is loaded, so `scored` loads lazily); prefix 48 tokens 4.60 → 3.60 ms, one token 1.93 → 1.25 ms; search equals v3 125/125; FP16 KV rejected (no lower peak, slower step), FP16 compute failed alignment; added to the Hugging Face release |
| [vimeml-zh-v2-20m on Core ML and in Vime](reports/mac-20261121/zh-v2-20m.md) | 2026-11-21, `kd-cont`: stateless and KV v4 INT8 packages, task-mode search gate, Vime integration; iPhone 13 Pro / 16 Pro Max Rime rerank p50 4.1 / 3.0 ms, one-shot generation 91 / 59 ms, peak 75 / 63 MB |

Mac briefs, the task descriptions handed to the macOS session (each report above answers one):

| Brief | Task |
| --- | --- |
| [zh-v1 Core ML](reference/mac-zh-v1-coreml.md) | Convert and quantize vimeml-tiny-zh-v1 with the V4.2 recipe; Mac and iPhone measurements |
| [zh-v1 batched KV](reference/mac-zh-v1-kvbatch.md), [v2](reference/mac-zh-v1-kvbatch-v2.md), [sequence head](reference/mac-zh-v1-kvsequence.md), [v3](reference/mac-zh-v1-kvbatch-v3.md) | Successive KV-cache packages for bucketed beam search and sentence scoring |
| [zh batched KV v4](reference/mac-zh-kvbatch-v4.md) | `prefix` function and FP16-KV variants, driven by iPhone memory profiles |
| [zh-v1 Neural Engine probe](reference/mac-zh-v1-ane-probe.md) | Step 0: whether the ANE is worth a deployment path for the 20M models |
| [vimeml-zh-v2-20m](reference/mac-zh-v2-20m.md) | Core ML build of `kd-cont` and task-mode integration in Vime |
| [vimeml-zh-v2-20m colloq-c40](reference/mac-zh-v2-20m-colloq.md) | Rebuild with c40 weights and swap in Vime; includes the 2026-11-23 Mac result and its search-gate deviation (122/125) |

## Japanese line

| Document | Scope |
| --- | --- |
| [V1 training](training.md) | Original baseline and reproducibility |
| [V2 training](training-v2.md) / [V2 design](plan_v2.md) | Architecture, tokenization, cropping, and schedules |
| [V3 experiments](plan_v3.md) | Adaptation and joint retraining; no stable replacement advantage |
| [V4 training](training-v4.md) | Fixed 8K architecture, A/B schedules, results, and recovery |
| [V4 post-training](posttraining-v4.md) / [conversion data](data-v4-c1.md) | C1/C1r2 experiments, frozen data, conditional scoring |
| [V4.1 joint pretraining](training-v41.md) | Sixteen-epoch LM/chat/conversion experiment |
| [V4.2 LM/chat pretraining](training-v42.md) | Twenty-epoch LM/chat experiment; the deployed release |
| [V5 scale-up pretraining](training-v5.md) | d512/L8 (28.4M) vs V4.3: LM BPC −0.10, IME Top-1 unchanged; not deployed |
| [WSD schedule runs](training-wsd.md) | V4.2 and V5 for 50 epochs with warmup-stable-decay: best BPC at each size, IME Top-1 unchanged; not deployed |
| [V4.4 ablation plan](training-v44-plan.md) / [results](training-v44.md) | Six-arm single-factor ablation: no treatment effective; Standard seed band 501–523 (SD 9.5); exploratory 4-seed ensemble 535 |
| [V4.5 distillation plan](training-v45-plan.md) | Registered, not run; the Japanese line resumes next |
| [IME error decomposition](reports/analysis-20261021/ime-errors.md) | Remaining Top-1 errors: candidate-pool misses, notation variants, few semantic errors |
| [Training-free rescoring](reports/analysis-20261021/rescoring.md) | PMI, length and engine-score fusion; none passes held-out tracks |

### Japanese benchmarks

| Document | Scope |
| --- | --- |
| [Standard IME](benchmarks/standard-ime.md) / [initial comparison](benchmarks/standard-ime-results-20261008.md) | Source splits, AI review, candidate export, reference-set consumption |
| [IME metrics v2](benchmarks/metrics-v2.md) | Notation-lenient and pool-conditional Top-1, author-cluster tests, decision rule, track roles |
| [Standard v2 blind](benchmarks/standard-ime-v2-blind.md), [homophone challenge](benchmarks/homophone-ime.md), [incremental prototype](benchmarks/incremental-ime.md), [candidate export v2](benchmarks/candidates-v2-handoff.md) | Frozen confirmation, written-text homophone, and segmented-input tracks; the first two are consumed |
| [Confirmation plan v2](benchmarks/confirmation-v2-plan.md) / [results](benchmarks/confirmation-v2-results-20261021.md) | V2.1 → V4.2 +22 on blind, author-cluster p = 0.0508: not confirmed. V4.3 → V5 +18 on homophones (Holm p = 0.025) |
| [Seed-ensemble replication](benchmarks/ensemble-replication-v2-results-20261105.md) | Four-seed ensemble beats V4.2 on blind 619 vs 598 (p = 0.0012), replicated; homophones +7, not significant |

## Experiment records

| Experiment | Records |
| --- | --- |
| V2 family | [Summary](reports/v2-summary.md) |
| V2 data/model preparation | [Tokenizer](reports/v2-20261007/tokenizer.md), [architecture](reports/v2-20261007/model.md), [token store](reports/v2-20261007/token-data.md), [prefix crop](reports/v2-20261007/prefix-crop.md) |
| V2.0 | [Environment](reports/v2-20261007/environment.md), [evaluation](reports/v2-20261007/evaluation.md) |
| V2.1 restart1 | [Plan](reports/v2-20261007/v21-plan.md), [throughput](reports/v2-20261007/v21-performance.md), [interrupted predecessor](reports/v2-20261007/v21-recovery.md), [evaluation](reports/v2-20261007/v21-evaluation.md) |
| V2.1 extension | [Training and selection](reports/v2-20261007/v21-extend.md) |
| Historical expanded IME set | [Construction](reports/v2-20261007/ime-3000-handoff.md), [initial evaluation](reports/v2-20261007/expanded-ime-evaluation.md), [label/error audit](reports/v2-20261007/label-error-audit.md) |
| V2.1 Apple deployment | [Core ML](reports/mac-20261008/v21-coreml.md), [quantization review](reports/mac-20261008/v21-quantization-review.md), [iPhone measurements](reports/mac-20261008/v21-iphone.md), [KV cache](reports/mac-20261009/v21-kv-cache.md) |
| V3 A / B | [A evaluation](reports/v3-20261008/a-evaluation.md), [B plan](reports/v3-20261008/b-plan.md), [B evaluation](reports/v3-20261008/b-evaluation.md) |
| V4-a preparation | [Tokenizer and vocabulary comparison](reports/v4-20261014/tokenizer.md), [8K token data and windows](reports/v4-20261014/token-data.md), [fixed architecture](reports/v4-20261014/model.md) |
| V4 base LM | [A continuation](reports/v4-20261015/evaluation.md), [B twelve-epoch cosine](reports/v4-20261015/b12cosine-evaluation.md) |
| V4 conversion post-training | [C1r2 data revision and diagnostics](reports/v4-20261016/c1r2.md) |
| V4.1 joint pretraining | [Results summary](training-v41.md), [complete evaluation](reports/v41-20261016/evaluation.md) |
| V4.2 LM/chat pretraining | [Results summary](training-v42.md), [complete evaluation](reports/v42-20261020/evaluation.md), [fixed reference and regressions](reports/v42-20261020/reference-regressions.md) |
| V4.2 Apple deployment | [Core ML, INT8 and Mac measurements](reports/mac-20261020/v42-coreml.md); Mac briefs for [V4.2](reference/mac-v42-coreml.md) and [V4.3](reference/mac-v43-coreml.md) |
| External Japanese LMs | [TinySwallow](reports/v4-20261016/external-1p5b.md), [RakutenAI](reports/v4-20261016/external-rakuten-1p5b.md), [Sarashina](reports/v4-20261016/external-sarashina-1p4b.md) |

## Historical V1 references

[FP32 results](reference/results.md), [Core ML measurements](reference/coreml-v1.md), [simulator memory](reports/mac-20261006/simulator-memory.md), [real keyboard extension](reports/mac-20261006/iphone-keyboard-memory.md), [artifact import](reports/mac-20261006/handoff.md), and [merge verification](reports/mac-20261006/merge-verification.md).

Methods: [corpus pipeline](reference/data-pipeline.md), [AJIMEE export](reference/ajimee-benchmark.md), [development generation](reference/development-generation.md), [hybrid ranking](reference/hybrid-ranking.md), [phrase demo](reference/phrase-demo.md), and [Chinese prediction page](zh-CN/reference/phrase-demo-zh.md) (Chinese).

`history/` preserves pre-measurement [deployment design](history/deployment.md), [conversion commands](history/coreml-guide.md), and [workflow status](history/workflow.md). Historical expectations are distinct from subsequent measurements. Original JSON, candidate-level scores, and traces remain in Git-ignored `outputs/`.
