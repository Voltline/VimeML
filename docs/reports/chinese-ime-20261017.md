# Chinese IME local research — 2026-10-17

This experiment ran both published Chinese IME rerankers, a portable Rime/Octagram baseline, and a modern-runtime PinyinGPT-Concat diagnostic. The 4M model is a useful next research baseline. Candidate generation and regressions need attention before product integration. No existing Japanese training configuration, checkpoint, or client integration was changed.

## Scope and evidence

| Route | Local status | Role |
|---|---|---|
| `metasequoiaime/pinyin-ime-reranker-4M` | Official Rust inference, real Rime pools, latency and loaded working set measured | Small reranker baseline |
| `metasequoiaime/pinyin-ime-reranker-25M` | Same harness and pools as 4M | Desktop accuracy reference |
| Rime 1.17.0 + Octagram | Portable DLL; `rime-pinyin-simp`; `zh-hans-t-essay-bgw.gram` | Traditional baseline and candidate generator |
| PinyinGPT-Concat | Public `aihijo` checkpoint; modern Transformers adapter; full/abbreviated segmented input tested | Direct constrained generation diagnostic |
| GeneInput | Default repository inspected; fetched revision contains only a README, with no executable model release | Task-design reference, not measured |
| AttnInput | Paper entry checked; no local inference reproduction performed | Architecture reference, not measured |

This does not reproduce the rerankers' published MSIME scores: the sentence labels are shared, but the engine, dictionary, candidate lists and denominator differ. Wanxiang grammar models were not measured; the traditional lane here is Octagram's published simplified-Chinese word model.

Local artifacts are in `artifacts/chinese-ime-research-20261017/`; outputs are in `outputs/chinese-ime-research-20261017/`. Both are ignored by Git. Original downloads, notices, commits, weight digests, frozen inputs, pools and individual scores are retained there. The reports and research adapters can be tracked independently of those assets.

## Method

The models were selected before scoring. There was no training, threshold search, model selection on the labels, or changes to the pre-existing Japanese evaluation pools.

The frozen input file has 105 cases: 60 upstream hand-authored sentences, 32 newly authored paired context/homophone probes, and 13 robustness probes covering abbreviations, a mixed abbreviation, spelling errors, a syllable boundary, numeric mixing and an unsupported nine-key string. These are small diagnostic/reference sets, not blind tests or population accuracy estimates. The context labels are author expectations and can admit alternatives, including `终止`/`中止` in one probe.

Rime runs in independent local shared/user directories. User learning and completion are disabled. The recorder takes the first nine engine candidates, preserving their order, without injecting gold answers. A candidate is marked as covering the complete input only if selecting it in a fresh session commits exactly its text. Prefix candidates therefore remain visible in the dump but cannot participate in full-key reranking. Candidate generation starts without committed-text history; the neural reranker receives each probe's supplied context.

Two candidate configurations were retained:

1. The ordinary one-sentence configuration.
2. A separate n-best configuration using `translator/max_sentences: 9` and `translator/sentence_cutoff_threshold: 1.0`, with the same dictionary. This is a candidate-search ablation, not a tuned production setting. The first nine returned candidates can still include prefixes and fewer than nine full sentences.

The supported keys and threshold semantics were checked against the pinned Rime 1.17.0 source. An exploratory zero-threshold configuration yielded the original single-sentence pools; those files remain separately under `rime-multi/` and are not the n-best results below.

Both rerankers use the unmodified upstream Rust crate and mean conditional log probability per candidate character. Scores include the first candidate character. The actual model configurations are 6 layers / width 192 / context 64 / vocabulary 8192 for 4M, and 8 / 448 / 128 / 12288 for 25M. Quantized matrices remain INT8 in this runtime.

The public Rime C API does not expose the original engine's dictionary-source fact. The research gate therefore uses an explicit **dictionary-membership proxy**: an exact text/normalized-pinyin pair found in the source dictionary, plus complete-key coverage. This is not a verified replacement for production candidate provenance, particularly for correction or alternate spellings. Ungated scores are retained separately.

## Sentence results

Exact top-1 matches on the 60 upstream reference sentences:

| Candidate generator | Engine | + 4M | + 25M | Gold present among full top-nine candidates |
|---|---:|---:|---:|---:|
| Rime, ordinary sentence setting | 41 | 41 | 41 | 41 |
| Rime + Octagram, ordinary setting | 45 | 45 | 45 | 45 |
| Rime, n-best | 41 | 44 | 46 | 46 |
| Rime + Octagram, n-best | 45 | **49** | **52** | **54** |

In the ordinary setting, each reference sentence has only one complete candidate. Reordering cannot recover an answer that is missing. Expanding the search reveals both reranker gains and regressions.

On Octagram n-best pools, 4M rescues **7** engine errors and breaks **3** correct answers. The 25M model rescues **8** and breaks **1**. Its 52 matches do not reach the 54-case candidate ceiling. The larger model is correct on three cases where the small model is wrong; none go the other way. With only three discordant pairs, exact two-sided McNemar p = 0.25; this does not establish a reliable model ranking for deployment.

The upstream project identifies 12 `ta`-initial cases as gender-ambiguous. Removing those by the documented input-prefix rule leaves 48 reference cases. In the Octagram n-best lane, the counts become **33 → 37 → 40**, with gold available in **42** pools. This sensitivity check does not make the remaining cases a new blind benchmark.

## Context and input robustness

The 32 paired context probes have gold present in 31 Rime pools. Both Rime configurations choose the authored label in 15 cases. The dictionary-membership gate retains all 32 leaders, so gated reranking stays at 15. The following is an ungated score diagnostic:

| Model / policy | Exact matches /32 | Rescue / break relative to engine |
|---|---:|---:|
| 4M, no context | 11 | — |
| 4M, supplied context | 18 | 5 / 2 |
| 25M, no context | 10 | — |
| 25M, supplied context | 20 | 6 / 1 |

Context helps this deliberately ambiguous set, but overriding every dictionary result still causes regressions. Conversely, blindly importing the upstream gate can suppress the intended context-sensitive behavior entirely. A Chinese product needs reliable source/coverage facts and its own policy evaluation.

On the 13 robustness probes, Octagram n-best has 4 engine matches and only 7 reachable gold answers. Gated 4M reaches 5 (2 rescues, 1 break); 25M reaches 6 (2 rescues, 0 breaks). The nine-key input is outside this Rime schema's capability. These counts diagnose coverage gaps; they do not measure general typo or nine-key support.

## Latency and memory

Host: Windows 11 build 26300, Intel Core Ultra 7 255HX, 20 logical CPUs exposed, default Rust release build with MSVC, Rust 1.85.1. These are exploratory desktop wall-clock measurements without CPU affinity or a controlled power/thermal regime. No iOS, Android or Core ML runtime was tested.

| Measurement | 4M | 25M |
|---|---:|---:|
| Weight file, decimal MB | 4.486 | 25.480 |
| Process working set immediately after load, decimal MB | 10.0 | 31.7 |
| Startup peak working set, decimal MB | 14.9 | 57.1 |
| Fresh active rerank, Octagram n-best sentences, p50 / p95 ms | 11.45 / 27.07 | 70.64 / 141.96 |
| Official `SentenceModel::score`, 9 × 8 characters, p50 / p95 ms | 30.96 / 34.28 | 169.57 / 177.03 |
| Official `SentenceModel::score`, 9 × 15 characters, p50 / p95 ms | 58.29 / 75.06 | 318.44 / 343.81 |
| Growing-prefix cache trace, step 8, p50 / p95 ms | 4.24 / 6.25 | 33.27 / 35.13 |
| Growing-prefix cache trace, step 15, p50 / p95 ms | 4.51 / 5.53 | 35.12 / 37.46 |

Fresh active-rerank percentiles exclude gates and single-candidate decisions; the two models see identical pools. Official shape sweeps use 40 observations per cell and no cross-call resumption. Longer official cells can repeat candidate text, as the upstream benchmark explicitly reports. The full official sweep is exploratory; later cumulative-typing sections overlapped other preparation, so the table above uses its initial shape sweep rather than the cumulative figures.

The separate growing-prefix trace uses nine distinct initial characters, 15 successive character extensions, and 30 repetitions. Each repetition starts with a new cache and visits each candidate list once; it does not alternate two already cached complete lists. This isolates incremental reuse on a synthetic shape, not real Roman-letter keystrokes or changing engine segmentations. Cached and fresh scores agree within 1e-4. The upstream benchmark's unchanged-list microsecond numbers are exact cache hits and are not used as typing latency.

Working set is for a standalone process paused after model loading, with the redundant file-byte buffer released. It includes runtime overhead. Startup peaks include loading temporaries. Neither figure measures the fully warmed keyboard extension with its engine, dictionaries and caches.

## PinyinGPT diagnostic

The README's `visualjoyce` model endpoint returned an access error. The publicly accessible [`aihijo/transformers4ime-pinyingpt-concat`](https://huggingface.co/aihijo/transformers4ime-pinyingpt-concat) checkpoint was pinned and hash-verified. It has **102,408,960 distinct parameters** and a **488,536,999-byte** PyTorch file.

The legacy source targets Transformers ~4.15 / PyTorch ~1.10. A separate modern-runtime adapter uses Transformers 4.57.6 and the upstream segmented prompt, aligned positions, syllable-specific character masks, beam width 16 and ten returned sequences. The published configuration omits 443 pinyin tokens found in the weights; the adapter sets vocabulary size to 21,571 before loading. No learned tensor is missing; only obsolete attention-mask buffers are ignored. Cached/full next-token logits differ by at most 4.77e-6 in the smoke check.

This is **not raw unsegmented input evaluation**. Pronunciation and boundaries come from the reference text via pypinyin, as in the upstream evaluation convention; the abbreviated mode uses the first letter of each syllable. It implements neither typo correction nor nine-key conversion. Seven reference readings differ from the original TSV's raw key string.

| Segmented task | Top-1 | Top-10 |
|---|---:|---:|
| Upstream sentences, full pinyin | 44/60 | 46/60 |
| Upstream sentences, first-letter abbreviation | 14/60 | 28/60 |
| Same readings as original raw keys, full pinyin | 38/53 | 39/53 |
| Same readings as original raw keys, abbreviation | 13/53 | 26/53 |
| Authored context probes, full pinyin | 29/32 | 32/32 |
| Authored context probes, abbreviation | 28/32 | 31/32 |

The matched-reading slice still has oracle syllable boundaries and a different candidate space from Rime. These columns must not be used as a direct reranker leaderboard. The checkpoint does not document enough run settings to claim reproduction of ACL paper numbers.

The FP32 run used an RTX 5060 Laptop GPU (8151 MiB reported). Peak allocated tensor memory was 488.2 MB. Observed sentence generation p50/p95 was 97.7/165.1 ms for full pinyin and 98.7/180.4 ms for abbreviation; two-character context probes were about 35–39 ms. This GPU pass overlapped portable Rime capture and is only a first-pass timing record, not a controlled speed comparison with the CPU rerankers.

## Validation and reproducibility

The upstream Rust suite passed **59 tests**, with **10 explicitly ignored** real-model/manual performance tests. Both official `verify` runs passed file digest, loader, metadata and ranking checks. Each model's **3,016** saved candidate scores were checked against upstream PyTorch inference, covering both n-best pools with and without context. Maximum absolute errors were 5.96e-5 and 1.20e-4 nats/character, respectively, with **zero argmax disagreements**. The independent score comparison and growing-trace comparison are recorded in the output JSON; production bitwise equality is not claimed.

Research entry points:

- [`chinese_ime_research.py`](../../scripts/benchmarks/chinese_ime_research.py): preparation, Rime capture, both rerankers, memory sampling and growing trace.
- [`chinese_ime_capture.cpp`](../../scripts/benchmarks/chinese_ime_capture.cpp): public Rime C API recorder.
- [`chinese_ime_probe.rs`](../../scripts/benchmarks/chinese_ime_probe.rs): adapter around the unmodified reference crate.
- [`chinese_pinyingpt_probe.py`](../../scripts/benchmarks/chinese_pinyingpt_probe.py): isolated modern PinyinGPT diagnostic.
- [`check_chinese_ime_scores.py`](../../scripts/benchmarks/check_chinese_ime_scores.py): independent forward-score check.

Python 3.13.5 / PyTorch 2.10.0+cu128 were already installed. Existing optional packages under `artifacts/external-eval/site-packages` provide Transformers 4.57.6, safetensors 0.7.0 and PyYAML. pypinyin 0.55.0 was installed only into this experiment's `site-packages`. The Rust toolchain lives inside the experiment, with no persistent PATH changes. The repository venv and system IME were not modified.

After restoring the pinned downloads listed in `models/manifest.json`, `environment.json` and `artifact-manifest.json`, and supplying the local frozen `probes.jsonl`, the local sequence is:

```powershell
.\.venv\Scripts\python.exe scripts/benchmarks/chinese_ime_research.py prepare
# Build capture.exe against the downloaded Rime headers/rime.lib using MSVC /std:c++17 /EHsc /utf-8.
# Build the generated artifacts/.../probe/Cargo.toml with the isolated toolchain, --release --locked.
.\.venv\Scripts\python.exe scripts/benchmarks/chinese_ime_research.py capture
.\.venv\Scripts\python.exe scripts/benchmarks/chinese_ime_research.py score
.\.venv\Scripts\python.exe scripts/benchmarks/chinese_ime_research.py prepare-multi
.\.venv\Scripts\python.exe scripts/benchmarks/chinese_ime_research.py capture-multi
.\.venv\Scripts\python.exe scripts/benchmarks/chinese_ime_research.py score-multi
.\.venv\Scripts\python.exe scripts/benchmarks/chinese_ime_research.py latency
.\.venv\Scripts\python.exe scripts/benchmarks/check_chinese_ime_scores.py
.\.venv\Scripts\python.exe scripts/benchmarks/chinese_pinyingpt_probe.py
```

The full model revisions and digests are retained locally. Key identities:

| Asset | Revision / SHA256 |
|---|---|
| Reranker source | `cb39eb5463cbf6358180152f361504120af2790e` |
| 4M model revision | `e5b1f7e768d7cb2b6ff334db4e34af153920c6ff` |
| 4M weight SHA256 | `86ac529510cb3b4968a5e6ade83ec8080f5b34a0a681e75c362accbbd387d1c1` |
| 25M model revision | `75295b373a7ef0d14e9bebd32c1625c9cfcfea49` |
| 25M weight SHA256 | `0a6ecba69bf1d39c7eb49549c716477dd6c03fdf05757773e1f50435262fb469` |
| PinyinGPT revision | `76dd20dc92d8236a350fb732e99dde6fa15e2263` |
| PinyinGPT weight SHA256 | `1c5ebb9e7b15d75ea8899b914fc8363f4745703115253071f7834780263c74bb` |
| Frozen 105-case SHA256 | `430a279566ff9b6ae452e41f3eb379cd37741f7a3f6d4338587927972f5f4b2a` |

The reranker weight/reference licenses are Apache-2.0; their training repository is GPL-3.0. Original LICENSE/NOTICE and embedded C4/LCCC attribution are preserved. The tested PinyinGPT card declares CC-BY-NC-SA-4.0, and the tested Octagram data carries LGPL-3.0. No third-party model or source is embedded into the product by this experiment.

## Research direction

Continue with 4M as the size/latency baseline and retain 25M as the desktop comparison. First freeze a stronger full-candidate pool and collect a substantially larger, independently reviewed Chinese set with context, abbreviations, corrections, multiple acceptable answers and source/coverage metadata. Keep rescue and break counts separate. Validate any cache policy on recorded real keystrokes, then measure the target Apple runtime before making mobile claims.

PinyinGPT supports pursuing explicit pinyin conditioning, but its segmented-input adapter and weak sentence-abbreviation results do not justify adopting it as a first mobile model. GeneInput remains useful for defining the broader task; AttnInput remains an unmeasured architecture reference here.

Primary references: [reranker project](https://github.com/metasequoiaime/chinese-ime-lm), [4M card](https://huggingface.co/metasequoiaime/pinyin-ime-reranker-4M), [25M card](https://huggingface.co/metasequoiaime/pinyin-ime-reranker-25M), [Rime 1.17.0](https://github.com/rime/librime/releases/tag/1.17.0), [Octagram data](https://github.com/lotem/rime-octagram-data), [PinyinGPT source](https://github.com/VisualJoyce/Transformers4IME), [ACL 2022 paper](https://aclanthology.org/2022.acl-long.133/), [GeneInput paper](https://aclanthology.org/2024.findings-acl.218/), [GeneInput repository](https://github.com/spirit-wang/Generative-Input), [AttnInput entry](https://openreview.net/forum?id=KdYsPwavXr).
