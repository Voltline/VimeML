# vimeml-tiny-zh-v1 批量 KV v3 Core ML 结果（共享前缀、合并函数）—— 2026-11-18

按 [v3 任务说明](../../mac/mac-zh-v1-kvbatch-v3.md)执行。沿用前几次任务的检出目录（`~/Documents/Sources/VimeML-zh-kvbatch`），fast-forward 拉到 `main` `586a04e`。

**拉取之前**

- Mac 上未跟踪的 kvsequence 报告副本（`docs/zh/reports/mac-20261116/zh-v1-kvsequence.md`）与上游文件冲突。
- 拉取前把它移到了 `outputs/deployment/mac-20261116-kvsequence-local-backup/`。它与上游副本只差 3 行编辑性改动，没有丢失内容。
- 拉取后 `git status` 干净。

**未改动**

- checkpoint、tokenizer、INT8 配方（linear_symmetric，per_block 32，47 个矩阵）和打分规则。
- `RELEASE.json` 和 `SHA256SUMS.txt`：复制发布文件前后都是 `899ca51a…` / `58096d45…`（`logs/12-release-copy.log`）。
- `src/vimeml/deployment/zh_v1_batch3.py` 和 `scripts/zh/deployment/coreml_zh_v1_batch3.py`：没有转换或对齐问题需要修复。
- Mac 上没有提交或推送任何内容。

**环境**

- 硬件与系统：Apple M3，16 GiB，macOS 27.3（26C5035r），Xcode 27.3（27C5020n）。
- Python：uv venv 中的 3.11.17，PyTorch 2.7.1，coremltools 9.0，NumPy 2.3.5，sentencepiece 0.2.1。coremltools 提示 torch 2.7.1 未经测试；没有阶段受到影响。
- Core ML：mlprogram，iOS 18 目标，FP32 计算，`CPU_ONLY`。
- 电源与负载：接交流电，`pmset -g therm` 没有警告。平均负载约 3.4。

所有输出在 `outputs/deployment/zh-v1-kvbatch-v3/`（`$O`）下：各阶段日志在 `logs/`，辅助脚本在 `tools/`。没有覆盖任何输出。被取代的内存测量尝试留作证据（第 7 节）。

## 摘要

- **门槛：** 每个阶段一次通过。
  - FP32 对 PyTorch 对齐最大 4.2e-5。
  - Vime 的 INT8 对无状态包门槛：`int8-merged` 和 `int8-merged-c4` 最大都是 4.15e-5，容差外 0 行（每个值 3e-4，每个和 1e-3）。
- **大小：**
  - `int8-merged` 把 `logits`、`scored` 和 `score` 放在一个 23.1 MB 的包里，共享一份权重。三个独立包需要 68.4 MB。
  - 已发布的 v2 组合（kvbatch-v2 + kvsequence）为 45.5 MB，加上无状态包为 68.3 MB。
- **延迟（Mac p50，B=16）：**
  - v3 解码与 kvbatch-v2 持平或更快：2.9–3.3 ms 对 3.0–3.9 ms。
  - v3 `score` 比 kvsequence 快 4–12%：T16 16.1–17.1 ms 对 17.2–17.8 ms。
- **内存（核心问题）：** v3 明显更省内存。以下为干净测量：预编译 `.mlmodelc`，5 个全新进程，中位数，相对起点的 MiB。

  | T | v2 已发布 峰值 | v3 峰值 | v3 单独 峰值 |
  | --- | ---: | ---: | ---: |
  | 16 | 40.5 | 18.5 | 17.8 |
  | 32 | 70.2 | 31.4 | 31.2 |
  | 64 | 150.0 | 56.8 | 56.2 |

  - 负载结束后，v2 已发布组合保留 16–57 MiB，v3 在每个 T 下都只保留约 4.5 MiB。
  - 每种配置的加载开销都约 2 MiB，因为权重是文件映射的。映射的权重文件：v2 已发布 68.3 MB，v3 45.9 MB，v3 单独 23.1 MB。
- **搜索：**
  - 对 v2 包，125/125 例完全一致。
  - 对 FP32 fixture：完全一致 46，集合相同 86，top-1 125，与 v2 相同。
- **Fixture：** 对无状态 INT8，每个 step 值最大 3.2e-5，每个分数和最大 1.7e-5。
- **决定**（见文末）：
  - 发布 v3（`int8-merged`）；
  - **不**选 -c4；
  - 无状态包可以去掉，但需 Vime 在真机上确认一次。

## 0. 单元测试（`logs/00-*.log`）

- 任务说明中的命令 `python -m unittest tests.test_deployment_zh_v1_batch3` 报 `ModuleNotFoundError: test_deployment_zh_v1_batch`。v3 测试模块按裸模块名导入 v1 测试模块，而 `tests/` 不在 `sys.path` 上。
- 用 `PYTHONPATH=tests python -m unittest tests.test_deployment_zh_v1_batch3`，5 个测试全部通过。
- 这是调用方式的问题，不涉及代码改动。建议上游修复：把 `tests` 作为包导入，或在任务说明中写上 `PYTHONPATH`。

## 1. PyTorch 检查（`torch-check.json`，`logs/01-check.log`）

- 通过。eager 对参照：`logits`、`scored` 和 `score` 各例（batch 1/5/16，前缀 11）最大 4.1e-5。
- 分 4 块的 `score` 与完整输出头每个和相差 3.5e-5。

## 2–6. 转换、合并、压缩与对齐

转换与合并（`logs/02-*`、`03-*`）：

- 转换了四个 FP32 包：`fp32-logits`、`fp32-scored`、`fp32-score`、`fp32-score-c4`。
- 合并为 `fp32-merged` 和 `fp32-merged-c4`。

FP32 对 PyTorch 对齐（`fp32-merged-align.json`、`fp32-merged-c4-align.json`）：

| 函数 | 最大绝对误差 |
| --- | ---: |
| `logits` | ≤ 4.20e-5 |
| `scored` | ≤ 2.68e-5 |
| `score`（74 个续写），c1 | 每个和 2.46e-5 |
| `score`（74 个续写），c4 | 每个和 2.44e-5 |

INT8 压缩与合并（`logs/05-*`、`06-*`）：

- 每个函数分别压缩（`int8-logits`、`int8-scored`、`int8-score`、`int8-score-c4`），再合并为 `int8-merged` 和 `int8-merged-c4`。

INT8 对无状态 `ios18-int8-block32` 的门槛（`int8-merged-align.json`、`int8-merged-c4-align.json`）：

| 函数 | `int8-merged` | `int8-merged-c4` |
| --- | ---: | ---: |
| `logits` | 4.15e-5 | 4.15e-5 |
| `scored` | 2.18e-5 | 2.18e-5 |
| `score`（每个和） | 3.70e-5 | 3.99e-5 |
| 容差外行数 | 0 | 0 |

### 包大小（字节）

| 包 | 合并 | 各函数独立包之和 |
| --- | ---: | ---: |
| FP32 | 80,697,965 | 241,122,305 |
| FP32 -c4 | 99,593,322 | 259,968,446 |
| **INT8** | **23,139,825** | 68,400,205 |
| INT8 -c4 | 28,470,632 | 73,681,796 |

各 INT8 函数包：

| 函数包 | 字节 |
| --- | ---: |
| `logits` | 22,812,208 |
| `scored` | 22,814,444 |
| `score` | 22,773,553 |
| `score-c4` | 28,055,144 |

已发布的包，用于对比：

| 包 | 字节 |
| --- | ---: |
| 无状态 | 22,792,958 |
| kvbatch-v2 | 22,774,484 |
| kvsequence | 22,772,997 |
| v2 组合 | 45,547,481 |
| v2 组合 + 无状态 | 68,340,439 |

合并后三个函数共享一份权重：合并包只比单个函数包大约 0.3 MB。

**-c4 为什么更大（`logs/06b-c4-constants.log`）**

- 分块输出头在完整的 (12288, 384) 共享 embedding 之外，另存了四个 (3072, 384) INT8 constexpr 权重。相当于多存了一份输出头（+5.3 MB）。
- 它的算子数：constexpr 51 对 47，linear 49 对 46。
- 这些切片没有与 embedding 去重。

## 7. Mac 延迟（`latency.json`、`latency-c4.json`、`latency-table.txt`、`logs/07-*`）

设置：

- `measure` 阶段，参数 `--prompts 16 48 --own 1 8 16 --score-tokens 8 16 --rows 16 --repeats 100`。
- v3 与已发布的 kvbatch-v2（解码）和 kvsequence（打分）在同一进程中交替运行。
- 数值为 3 次交替运行的中位数，p50，单位 ms，B=16。

`int8-merged`：

| prompt | own | T | v2 解码 | v3 解码 | kvsequence 打分 | v3 打分 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 8 | 2.99 | 3.18 | 10.21 | 9.57 |
| 16 | 1 | 16 | 2.95 | 2.91 | 17.30 | 16.37 |
| 16 | 8 | 8 | 2.96 | 3.02 | 10.04 | 9.24 |
| 16 | 8 | 16 | 2.98 | 3.03 | 17.19 | 16.22 |
| 16 | 16 | 8 | 3.09 | 3.10 | 10.08 | 9.34 |
| 16 | 16 | 16 | 3.10 | 3.11 | 17.18 | 16.25 |
| 48 | 1 | 8 | 3.22 | 3.01 | 10.60 | 10.21 |
| 48 | 1 | 16 | 3.22 | 3.02 | 17.79 | 16.91 |
| 48 | 8 | 8 | 3.30 | 3.15 | 10.62 | 10.24 |
| 48 | 8 | 16 | 3.51 | 3.30 | 17.80 | 17.11 |
| 48 | 16 | 8 | 3.90 | 3.34 | 11.23 | 9.85 |
| 48 | 16 | 16 | 3.51 | 3.31 | 17.59 | 16.13 |

`int8-merged-c4`，只列 `score`（解码是同一个函数）：

| prompt | own | T | kvsequence | v3c4 打分 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1/8/16 | 8 | 9.87–11.34 | 8.84–9.17 |
| 16 | 1/8/16 | 16 | 17.04–17.35 | 14.93–15.32 |
| 48 | 1/8/16 | 8 | 10.61–10.69 | 9.33–9.58 |
| 48 | 1/8/16 | 16 | 17.68–17.73 | 15.38–15.84 |

说明：

- prompt 16 时 v3 解码与 kvbatch-v2 持平，prompt 48 时快 0.2–0.6 ms。v2 每一行的缓存里都带着共享 prompt；v3 则是广播。
- v3 `score` 比 kvsequence 快 4–12%。
- -c4 `score` 比 c1 再快 6–10%，在任务说明 10% 的延迟范围内，而且更好。

## 8. 内存

### 按任务说明执行的 `memory` 阶段（`memory.json`、`memory-T32.json`、`memory-T64.json`、`logs/08-memory*.log`）

设置：

- `--prompt 48 --steps 16 --score-tokens 16|32|64 --rounds 20 --runs 3`，每种配置 3 个全新进程，中位数，相对 `start` 的 MiB。
- v2 已发布 = 无状态 + kvbatch-v2 + kvsequence。`-alone` 表示不含无状态包。

| 配置 | T16 加载 | T16 多轮后 | T16 峰值 | T32 峰值 | T64 峰值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2 已发布 | 164.6 | 178.9 | 203.1 | 232.7 | 312.0 |
| v3 | 164.5 | 166.6 | 180.8 | 192.4 | 217.7 |
| v3-alone | 159.9 | 162.9 | 176.4 | 195.7 | 238.7 |
| v3c4 | 168.7 | 170.4 | 186.0 | 233.5 | 266.9 |
| v3c4-alone | 166.1 | 168.9 | 183.9 | 233.9 | 265.1 |

**这些绝对数值受到了污染。** 相对规律成立：v2 已发布组合的峰值随 T 增长得快得多。污染有三个原因：

1. 子进程在 `start` 快照之后才在 `load_set` 里导入 coremltools（它会带入 torch），每次 `load` 都包含约 130–160 MiB 的 Python 导入。
2. 它在 `start` 之前同时构建了 v2 和 v3 的输入（约 1.2 GB）。在这版 macOS 上，释放的大块内存在被复用之前一直算在 `phys_footprint` 里，带来 ±30 MiB 的噪声。
3. 加载 `.mlpackage` 会在进程内编译；编译器的临时缓冲区带来更多噪声。

该输出作为任务说明中该阶段的结果原样保留。

### 干净归因（`memory-clean4-T16/32/64.json`、`tools/memory_clean4.py`、`logs/08e-*`）

每种配置在 5 个全新进程中运行。每个子进程：

1. 导入 coremltools。
2. 只构建该配置用到的输入，在整个运行期间保持存活。
3. 记录 `start` 快照。
4. 像 app 一样，用 `CompiledMLModel` 从在测量进程外预编译的 `.mlmodelc`（`compiled-for-memory/`）逐个加载模型或函数，每次加载后记录快照。
5. 运行 20 轮阶段负载：prompt，自有缓存 1..16 的 16 次解码，`logits`（v3），一次 T 个 token 的 `score`，B=16。

数值为 `phys_footprint`（proc_pid_rusage V4）减去 `start`，单位 MiB，5 次的中位数。大多数运行与中位数相差在 ±1 MiB 内；每次运行的值在 JSON 里。

| 配置 | T | 无状态加载后 | 加载后 | 多轮后 | 峰值 | 最小峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 已发布 | 16 | 1.9 | 2.2 | 16.3 | **40.5** | 40.5 |
| v3 | 16 | 1.9 | 2.3 | 4.3 | **18.5** | 18.4 |
| v3-alone | 16 | – | 1.4 | 4.3 | **17.8** | 17.7 |
| v3c4 | 16 | 1.9 | 2.4 | 6.1 | 21.6 | 20.1 |
| v3c4-alone | 16 | – | 1.5 | 4.4 | 19.4 | 19.3 |
| v2 已发布 | 32 | 1.9 | 2.2 | 23.2 | **70.2** | 70.1 |
| v3 | 32 | 1.9 | 2.4 | 4.6 | **31.4** | 31.3 |
| v3-alone | 32 | – | 1.4 | 5.0 | **31.2** | 30.6 |
| v3c4 | 32 | 1.9 | 2.4 | 4.5 | 35.1 | 34.9 |
| v3c4-alone | 32 | – | 1.4 | 4.4 | 34.5 | 26.9 |
| v2 已发布 | 64 | 2.0 | 2.3 | 57.3 | **150.0** | 149.7 |
| v3 | 64 | 1.9 | 2.4 | 4.6 | **56.8** | 27.6 |
| v3-alone | 64 | – | 1.4 | 4.5 | **56.2** | 56.0 |
| v3c4 | 64 | 2.0 | 2.4 | 4.6 | 65.1 | 65.1 |
| v3c4-alone | 64 | – | 1.4 | 4.4 | 64.4 | 64.4 |

解读：

- **加载。** 每种配置加载编译后的模型都只增加约 0.1–2 MiB 的 `phys_footprint`。权重以文件映射方式载入，`phys_footprint` 不计入。真正的权重开销是映射的文件：
  - v2 已发布：68.3 MB（三个权重文件）；
  - v3：45.9 MB（无状态 + 合并包）；
  - v3-alone：23.1 MB（只有合并包）。

  逐函数快照显示合并包的三个函数共享同一个映射：`after_fn_scored`、`after_fn_score` 和 `after_fn_logits` 相差 ≤ 0.2 MiB。
- **负载后保留的内存。**
  - v2 已发布组合在 T 从 16 增到 64 时保留 16 → 23 → 57 MiB。kvsequence 的 `[B,T,12288]` logits 缓冲区一直分配着。
  - v3 在每个 T 下都保留约 4.5 MiB。`score` 只返回 `target_logprobs`。
- **峰值。** v3 比 v2 已发布组合低 2.2–2.6 倍，从 T16 到 T64 每个 T token 增长约 0.8 MiB，v2 已发布组合约 2.3 MiB。
- **-c4。** 每个 T 下峰值都**高于** c1：T16、T32、T64 分别 +3、+4、+8 MiB。加载开销相同，映射的权重大 5.3 MB。在 Core ML 的 CPU 执行中，分块输出头并不降低打分峰值。
- **离群值。** 两个"最小峰值"离群值（v3 T64 27.6；v3c4-alone T32 26.9）都是单次运行。这两次运行中分配器复用了初始化阶段释放的内存，`after_load` 和峰值都负了约 30 MiB。中位数不受影响。

被取代的尝试，留作证据：

- `memory-clean-T*.json`（`tools/memory_clean.py`）：释放的输入仍留在内存占用中，v3 加载显示为 −340 MiB。
- `memory-clean2-probe.json`：`malloc_zone_pressure_relief` 释放了 0，伪影仍在。
- `memory-clean3-T*.json`（保留全部输入，加载 `.mlpackage`）：进程内编译造成 ±35 MiB 的双峰分布。这促成了第 4 版改为加载 `.mlmodelc`。

## 9. 搜索（`search/`，`logs/09-search.log`）

检查运行 `$C search $O/int8-merged outputs/vime-fixtures/zh-generation-v1-lm-r2/cases.jsonl $O/search`，共 125 例。

| 对比对象 | 完全一致 | 集合相同 | top-1 相同 | 共同候选最大分数差 |
| --- | ---: | ---: | ---: | ---: |
| v2 包（kvbatch-v2） | **125** | 125 | 125 | 1.7e-5 |
| FP32 fixture | 46 | 86 | 125 | 0.56 |

- 两行都符合预期：对 v2 完全一致，对 FP32 的数字与 v2 相同（INT8 的影响）。
- 交替运行时搜索 p50 v3 为 85.0 ms，v2 为 82.7 ms；p95 为 184.8 对 185.9。
- 当时内存测量正在后台运行，这些计时只作参考。

## 10. Fixture（`fixture/`，`logs/10-fixture.log`）

- fixture 由选定的 `int8-merged` 构建。
  - 步：16 行经过 `scored` 和 `logits`，前缀来自 B=1 的 prompt 调用。
  - 打分：8 例经过 `score`，每次调用最多 16 行。
- 对无状态 INT8：每个 step 值最大 **3.24e-5**，每个分数和最大 **1.70e-5**，远在 3e-4 / 1e-3 之内。
- 文件：`step.jsonl`、`step.npz`、`score.jsonl`，以及带包 manifest `7e4c01d2…` 和 tokenizer `68bd37c4…` 的 meta。

## 11. 客户端资源与发布副本

- `$O/ios-kvbatch-v3/`（`logs/11-ios.log`）包含：
  - `Resources/TinyChineseV1KVBatchV3INT8.mlmodelc`；
  - `Resources/VimeLMManifestZhV1KVBatchV3.json`（`kv_batch_v3` 段，含编译后模型的逐文件哈希）；
  - `KVBatchV3Fixture-step.jsonl`、`-step.npz`、`-score.jsonl`、`-meta.json` 和 `manifest.json`。
- 之前的 v1、v2 和 kvsequence 资源未动。
- 发布副本：`artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvbatch-v3/`。
  - 它是 `$O/int8-merged` 的逐字节副本：`diff -r` 无差异，22 MiB。
  - `RELEASE.json` 和 `SHA256SUMS.txt` 前后哈希相同（`logs/12-release-copy.log`）。与以前一样，把副本登记进发布清单留给 Windows 端做。

## 12. iPhone

本次未做。Vime 应在真机上检查：

- 单独加载 v3 时 T = 16 和 32 的峰值；
- `logits` 函数（无前缀、空行缓存）能否覆盖无状态包的用途，以及这条路径的延迟。

## 决定

**1. 发布 v3：是，`int8-merged`。**

- 两项对齐都通过，容差外 0 行。
- 搜索与 v2 包完全相同（125/125 完全一致），fixture 与无状态包相差 3e-5。
- 解码延迟与 v2 持平或更好，打分快 4–12%。
- 内存明显更低：
  - 映射权重：45.9 MB 对 68.3 MB（不含无状态包为 23.1 MB）；
  - 负载后保留：约 4.5 MiB 对 16–57 MiB；
  - 峰值：T 16 / 32 / 64 时 18.5 / 31.4 / 56.8 MiB 对 40.5 / 70.2 / 150.0 MiB。
- 有了 v3，上一份报告中 kvsequence 的 T 上限可以放宽。T ≤ 16 的代价是约 18 MiB 峰值、约 4 MiB 保留；T ≤ 32 约 31 MiB 峰值。

**2. 选 -c4：否。**

- 任务说明的规则是：只有它降低的打分峰值超过它增加的大小时才选它。它没有做到：每个 T 下峰值都更高（+3 到 +8 MiB），而且因为输出头常量重复，包大了 5.3 MB。
- 打分快 6–10% 抵不过这些。

**3. 去掉无状态包：按 Mac 证据为是；需在真机上确认一次。**

- 无前缀、无行缓存的 `logits` 函数通过了对无状态包的 INT8 门槛（4.15e-5）和 fixture 的 `logits` step 检查。
- v3-alone 在每个 T 下的保留内存和峰值都与 v3 相同，映射少 22.8 MB。
- 这里没有测：经 `logits` 计算完整 prompt logits 与无状态包的延迟对比。Vime 应在移除无状态资源前在真机上检查。

## 代码改动

无。`src/vimeml/deployment/zh_v1_batch3.py` 和 `scripts/zh/deployment/coreml_zh_v1_batch3.py` 停留在 `586a04e`。与任务说明唯一的偏离是单元测试的调用方式（`PYTHONPATH=tests`，第 0 节）。内存辅助脚本是 `$O/tools/` 下的新文件，不是仓库代码。

## 文件

| 路径 | 内容 |
| --- | --- |
| `torch-check.json` | PyTorch 检查 |
| `fp32-{logits,scored,score,score-c4}/`、`fp32-merged{,-c4}/`、`fp32-merged{,-c4}-align.json` | FP32 包与对齐 |
| `int8-{logits,scored,score,score-c4}/`、`int8-merged{,-c4}/`、`int8-merged{,-c4}-align.json` | INT8 包与 Vime 的门槛 |
| `latency.json`、`latency-c4.json`、`latency-table.txt` | 延迟 |
| `memory.json`、`memory-T32.json`、`memory-T64.json` | 按任务说明执行的 `memory` 阶段 |
| `memory-clean4-T16/32/64.json`、`compiled-for-memory/` | 干净内存归因（决定依据） |
| `memory-clean-T*.json`、`memory-clean2-probe.json`、`memory-clean3-T*.json` | 被取代的尝试，留作证据 |
| `search/`、`fixture/`、`ios-kvbatch-v3/` | 搜索、fixture、客户端资源 |
| `tools/` | `run_build.sh`、`run_measure_memory.sh`、`memory_clean{,2,3,4}.py` |
| `logs/` | 各阶段日志 |

## Windows 端后续（2026-11-18）

- 结果 ZIP 已与其 SHA256SUMS 核对。发布副本与 `int8-merged` 逐字节相同，并通过 `zh_v1_batch3.verify_merged`。其 manifest 记录的转换源为 `0a05cef6…`，即 `586a04e` 的 `zh_v1_batch3.py`。
- 它的 `weight.bin` 与 sequence 包的逐字节相同：三个函数共享每个 v2 包本来就带的那一份 INT8 权重文件。
- 发布清单现在列出了 `ios18-int8-block32-kvbatch-v3/` 的四个文件，以及 `coreml_manifest_sha256.kvbatch_v3_int8` = `7e4c01d2…`。
- 发布 README 描述了 v3 包。v2 包标为 beam 搜索已被取代。
- `verify_release.py --hashes` 在 382 个文件上通过。
- Hugging Face 发布在 revision `32766748114c38ffc2252b645a51758e066e80b8`（父 revision `ff3dab5f…`）原地更新。重新下载该 revision 后通过 `verify_release.py --hashes`。
- 第 0 节：v3 测试模块现在把 `tests/` 加入 `sys.path`，所以 `python -m unittest tests.test_deployment_zh_v1_batch3` 可以在仓库根目录直接运行。
- 第 8 节：`memory` 阶段现在按 `memory_clean4.py` 的做法：
  - 在 `start` 之前导入 coremltools 并构建全部输入；
  - 包在测量进程之外编译一次，由 `CompiledMLModel` 加载；
  - 每加载一个模型或函数后记录快照；
  - 默认 5 次运行，报告中位数和最小峰值。
