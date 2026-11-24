# zh desktop latency, PyTorch on Windows (2026-11-18/19)

Desktop latency of the full pinyin pipeline (Vime bucketed search, beam 16, top 10, then self-rescoring
of the 10 candidates) for `zh200m-12k-pinyin-pt` (v2-200m candidate), `kd-cont` (v2-20m candidate) and
vimeml-tiny-zh-v1 (lm mode). Measured on a Windows laptop (RTX 5060 Laptop GPU; Intel CPU, 8 torch
threads), Python 3.13, PyTorch. Inputs: every 6th record of ime-pinyin-zh-v1 development, all of its
inputs (580 = 97 full, 386 mixed, 97 initials), in-field context; 5 warm-up inputs not counted.

Plan deviation: the release plan measured desktop latency on a Mac. A desktop client, if one is
built, would be a Windows client and not written in Swift, so desktop latency is measured on Windows
in PyTorch instead.

Scripts: `scripts/zh/benchmarks/latency_zh.py` (search + rescoring), `scripts/zh/benchmarks/latency_rime_zh.py`
(Rime candidates + rerank). CUDA graph backend: `src/vimeml/deployment/zh_torch_graph.py`.

## Results (2026-11-19, ms, p50/p95)

Full pipeline (search plus rescoring), after the search changes below:

| Model | Graph FP16 | Graph FP32 | Eager FP32, CUDA | CPU, 8 threads |
| --- | --- | --- | --- | --- |
| 200M | **83/205** | 170/419 | 172/407 | 795/1900 |
| kd-cont | **34/68** | 42/96 | 94/210 | 143/336 |
| v1 (lm) | **32/63** | 37/75 | 105/255 | 125/294 |

By input kind, Graph FP16:

| Model | Full | Mixed | Initials |
| --- | --- | --- | --- |
| 200M | 100/264 | 84/194 | 61/113 |
| kd-cont | 41/87 | 34/65 | 27/42 |
| v1 | 38/77 | 33/58 | 25/41 |

Parts of the total (p50): 200M Graph FP16 model 62, Python 11, rescoring 10; kd-cont Graph FP16 model 20,
Python 8, rescoring 6. Graph capture takes 1.2–2.7 s once at load and is not counted.

- **20M models:** CUDA graphs are the main gain (eager 94 → graph 42 ms for kd-cont); FP16 adds little.
- **200M:** each step reads the weights, so graphs alone change nothing (172 → 170 ms) and FP16 halves
  it (83 ms).
- FP16 against FP32 on search output: 200M top-1 579/580, kd-cont 578/580.

## Changes

1. `zh_search.log_softmax` returns a host array. The search read log-probs from device tensors one
   entry per move, a device sync each (29 of 36 s in a profile): about 4x faster on CUDA.
2. The search loop caches each key position's moves, scores a whole group of moves with one array
   read, and builds a hypothesis only when it enters the beam (stable descending order, as before).
   Top-10 texts and scores are identical to the previous search on all 580 inputs, for kd-cont
   (task) and v1 (lm). Profile of 60 kd-cont inputs on the graph backend: 8.19 → 3.26 s.
3. `zh_torch_graph.GraphBackend` replays the decode step from CUDA graphs (fixed batch of 16 rows,
   cache widths 32/64/128) and moves tokens, positions and row indices to the device in one copy.
   FP32 search output is identical to eager on all 580 inputs (max score difference 6e-6).

## Before the search changes (2026-11-18)

### Eager PyTorch, FP32, full pipeline, log_softmax fix only (ms, p50/p95)

| Model | CUDA | CPU, 8 threads |
| --- | --- | --- |
| 200M | 190/427 (model 130, Python 39, rescore 11) | 810/1984 (model 616) |
| kd-cont | 123/280 | 181/438 |
| v1 (lm) | 114/259 | 165/382 |

By input kind:

| Model, device | Full | Mixed | Initials |
| --- | --- | --- | --- |
| 200M CUDA | 243/506 | 190/419 | 141/292 |
| 200M CPU | 1022/2536 | 813/1794 | 546/1184 |
| kd-cont CUDA | 147/358 | 123/270 | 102/184 |
| kd-cont CPU | 235/543 | 185/410 | 142/296 |
| v1 CUDA | 134/318 | 114/254 | 93/179 |
| v1 CPU | 202/462 | 168/381 | 131/262 |

### CUDA graphs, search only, before the loop changes (ms, p50/p95)

Eager decode is bound by kernel launches; for 200M one step also reads about 800 MB of FP32 weights
(graph replay alone 7.1/8.0/9.9 ms at cache widths 32/64/128). The graph backend captures a fixed
batch of 16 rows per width.

| Model | Backend | p50/p95 | Against eager FP32 |
| --- | --- | --- | --- |
| 200M | eager FP32 | 206/469 | — |
| 200M | graph FP32 | 169/442 | identical top-10 on all 580, max score diff 7.5e-6 |
| 200M | graph FP16 | 116/295 | top-1 579/580, top-10 lists equal on 519 |
| 200M | graph BF16 | 126/324 | top-1 574/580 |
| kd-cont | eager FP32 | 131/322 | — |
| kd-cont | graph FP32 | 76/212 | identical |
| kd-cont | graph FP16 | 61/164 | top-1 578/580 |

## Rime

librime 1.17 with rime-ice (the engine of the Standard benchmark) returns candidates in about 7 ms,
but nearly always only one whole-input sentence, so reranking has nothing to choose from (v1 rerank
top-1 18.3 on 60 inputs). On the public set, Rime-16 + v1 rescoring scores Top-1 31.6 against 58.8 for
200M with its own rescoring. Rime is not an accuracy alternative; it stays the baseline.
