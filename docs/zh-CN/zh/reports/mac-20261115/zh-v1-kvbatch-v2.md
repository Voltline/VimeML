# vimeml-tiny-zh-v1 批量 KV 解码 v2 Core ML 结果 —— 2026-11-15

按 [v2 任务说明](../../mac/mac-zh-v1-kvbatch-v2.md)执行。沿用第一个任务的检出目录（`~/Documents/Sources/VimeML-zh-kvbatch`），快进拉取到 `main` `65c42fd`。

- 源模型：`vimeml-tiny-zh-v1`（推理包 manifest `9d9dad19…`，tokenizer `68bd37c4…`）。
- 未改动：checkpoint、tokenizer、INT8 配方（block32，阈值 2048，47 个矩阵）和搜索规则。
- `RELEASE.json` 和 `SHA256SUMS.txt` 未动（仍为 `899ca51a…` / `58096d45…`，记录在 `logs/07-release-meta-before.sha`）。
- Mac 上没有提交任何内容。
- 第一个任务在 `outputs/deployment/zh-v1-kvbatch/` 下的输出作为基线只读使用。
  - 本任务开始时第一份报告尚未提交；拉取时这份本地副本与上游文件冲突。
  - 拉取前它被移到 `outputs/deployment/mac-20261115-local-report-backup/`。已提交的上游版本只多了 2 行。

**环境**

- 硬件与系统：Apple M3，16 GiB，macOS 27.3（26C5035r），Xcode 27.3（27C5020n）。
- Python：uv venv 中的 3.11.17，PyTorch 2.7.1、coremltools 9.0、NumPy 2.3.5、sentencepiece 0.2.1。
- Core ML：mlprogram，iOS 18 目标，FP32 计算，`CPU_ONLY`。
- 电源与负载：接交流电，`pmset -g therm` 无警告。机器并不空闲（会话期间平均负载 7–24），单次延迟数字有噪声，因此 v1↔v2 对比交替测了三轮。

**输出与改动**

- 所有输出在 `outputs/deployment/zh-v1-kvbatch-v2/`（`$O`）下：各阶段日志在 `logs/`，辅助脚本在 `tools/`。没有覆盖任何输出。失败的步骤以日志保留。
- **修改了 `src/vimeml/deployment/zh_v1_batch2.py`**（只改形状声明，图不变）。见[对 zh_v1_batch2.py 的改动](#对-zh_v1_batch2py-的改动)。改动后重跑了 `check`，通过。

## 摘要

- **门槛：** 全部六个变体和三个固定形状备选都能转换、与 PyTorch FP32 对齐，并通过 Vime 对无状态包的 INT8 门槛（最大 ≤ 6.6e-5，容差 3e-4）。
  - `dec-enum` 和 `dec-b16` 第一次转换失败。两次失败都源于枚举形状的声明方式，而不是图。
  - 修复 `shape_spec` 后，两者都能转换并通过。
- **内存：** 在 Mac 上，仅靠只读缓存**不能**降低内存。T=1 变体要差得多：
  - **T=1 包**（`dec-flex`、`dec-enum`、`scored`）加载时增加约 200 MiB。每遇到一个新的 (B,S) 形状再涨几十 MiB，Core ML 内存最多到约 370 MiB。
  - **可变 T 包**（`unified`、`unified-scored`）加载时增加约 39 MiB。所有形状下都保持在约 40–41 MiB，与 v1 相同（38–40 MiB）。它们还省掉了 chunk 包（13 MiB）。
- **延迟：** `unified` 和 `unified-scored` 都远在 Vime 目标之内：Mac 上 S 直到 128 时 B=16 ≤ 4.4 ms、B=8 ≤ 3.6 ms。
  - B=16 时 T=1 变体比 v1 慢 2–6 倍。
  - 整个 prompt 预填充是一次 2–4 ms 的调用，而不是每个 token 一次调用。
- **搜索：** 每个 INT8 变体都与 v1 INT8 批量搜索结果相同，125/125 排序完全一致，分数相差 5.3e-5 以内。
  - FP32 `dec-flex` 和 FP32 `unified` 与 FP32 fixture 125/125 排序完全一致，分数相差 4.5e-5 以内。
- **选择：`unified-scored`。**
  - 它与 `unified` 并列内存占用最低，不需要 chunk 包，满足延迟目标。
  - 端到端搜索最快（p50 79 ms，`unified` 为 106 ms）。
  - 如果客户端不能使用 scored 输出头，`unified` 是备选；内存和延迟相同。
- **输出：** `ios-kvbatch-v2/` 和发布副本都由它构建。

## 1. PyTorch 检查

| 运行 | 解码最大 | 预填充 + 多 token 最大 | Scored 输出头对 log-softmax | 结果 |
| --- | ---: | ---: | ---: | --- |
| `check.json`（形状修复前） | 5.01e-5 | ≤ 2.7e-5 | 1.91e-6 | 通过 |
| `check-after-shape-fix.json` | 5.01e-5 | ≤ 2.7e-5 | 1.91e-6 | 通过 |

与 Windows 上的检查一致（3.96e-5 / 2.9e-5 / 1.9e-6）。

改动后 `python -m unittest tests.test_deployment_zh_v1_batch` 也通过：12 个测试，OK。venv 中没有安装 pytest。

## 2. 转换、压缩与门槛

两列都是最大绝对误差。FP32 对齐以 PyTorch FP32 为参照；INT8 门槛以无状态 INT8 包为参照。容差 atol = rtol = 3e-4。

| 变体 | 形状 | FP32 对齐 | INT8 门槛 | 结果 |
| --- | --- | ---: | ---: | --- |
| `dec-flex` | B 1..16，T=1，S 1..128（RangeDim） | 8.34e-5 | 5.96e-5 | 通过 |
| `dec-enum`（第一次） | B {1,4,8,16} × S {16,32,64,128} | — | — | **转换失败**（`logs/02-convert-fp32-dec-enum.log`） |
| `dec-enum`（修复后） | 16 个枚举的 (B,S) 对 | 7.77e-5 | 6.58e-5 | 通过 |
| `dec-b16`（第一次） | B=16，S {32,64,128} | — | — | **转换失败**（`logs/02-convert-fp32-dec-b16.log`） |
| `dec-b16`（修复后） | B=16 固定，S 枚举 | 5.02e-5 | 4.39e-5 | 通过 |
| `dec-b16-s32` / `-s64` / `-s128`（备选） | B=16，S 固定 | ≤ 4.49e-5 | ≤ 4.72e-5 | 通过 |
| `scored` | 同 `dec-flex`，scored 输出头 | 3.24e-5 | 4.10e-5 | 通过 |
| `unified` | B 1..16，T 1..128，S 1..128 | 5.29e-5 | 5.25e-5 | 通过 |
| `unified-scored` | 同 `unified`，scored 输出头 | 2.67e-5 | 2.96e-5 | 通过 |

两次转换失败都是 coremltools 9.0 针对 iOS 18 目标报出的：

- **`dec-enum`：**
  - 错误：*"Enumerated shape input mismatch. All enumerated shape inputs must have the same number of shapes for deployment targets iOS 18 or above."*
  - 原因：缓存列出了 16 个 (B,S) 形状，而 `input_ids` 和 `position` 只列了 4 个。
- **`dec-b16`：**
  - 错误：*"EnumeratedShapes should be take a list or tuple with len >= 2, got 1 instead".*
  - 原因：只有一个批大小时，`input_ids` 和 `position` 只有一个枚举形状。

按任务说明，修复前也构建并保留了三个固定 S 的 B=16 备选，下文有测量。

修复前构建的四个变体（`dec-flex`、`scored`、`unified`、`unified-scored`）的 manifest 记录的是旧的 `conversion_source_sha256`。修复只涉及枚举和固定分支，这几个 RangeDim 包从不走这些分支。

## 3. 内存

### `memory` 阶段（每种配置 3 个全新进程，取中位数）

数值是在已加载并运行无状态 INT8 包的进程之上增加的 `phys_footprint`，单位 MiB。无状态包本身增加约 15.5 MiB。来源：`memory.json`、`logs/03-memory.log`。

| 配置 | Chunk | 加载 | 预填充 24 | 首次 B16 S32 | 扫到 S=64 | B16 S128 | 峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 `int8-flex` | — | 37.4 | — | 60.0 | 98.8 | 179.9 | 189.6 |
| v1 + chunk | 13.0 | 47.2 | — | 68.7 | 102.9 | 184.0 | 193.7 |
| `dec-flex` | — | 209.3 | — | 251.8 | 345.7 | 547.0 | 547.0 |
| `dec-flex` + chunk | 13.2 | 214.1 | — | 256.6 | 349.9 | 551.1 | 551.1 |
| `dec-enum` | — | 199.6 | — | 249.6 | 349.9 | 551.1 | 551.1 |
| `dec-b16` | — | 51.9 | — | 61.0 | 101.6 | 182.7 | 183.0 |
| `dec-b16-s32` | — | 22.3 | — | 38.3 | 51.9 | — | 52.1 |
| `dec-b16-s64` | — | 22.4 | — | 65.3 | 92.4 | — | 92.7 |
| `dec-b16-s128` | — | 22.3 | — | 132.8 | 186.2 | 186.2 | 187.1 |
| `scored` | — | 209.4 | — | 250.7 | 343.5 | 544.8 | 544.8 |
| `unified` | — | 38.9 | 40.3 | 61.6 | 99.9 | 181.0 | 182.3 |
| `unified-scored` | — | 39.0 | 40.4 | 60.5 | 99.0 | 180.1 | 181.6 |

在检查过的每个进程中，`proc_pid_rusage` V4 的 `phys_footprint` 与第一份报告使用的 V2 值相差 1 MiB 以内：`memory-attribution/attribution.json` 中全部 7 个标签都是 `v2_equals_v4_phys_footprint: true`。

### 内存去了哪里（`memory-attribution/`，`tools/memory_attribution.py`）

`memory` 阶段在测量窗口内创建每次调用的 NumPy 输入。这些输入很大：一对 B=16、S=128 的 key + value 就有 56 MiB。所以上表中加载之后的增长很大一部分属于 Python，而不是 Core ML。

归因工具把两者分开。它在每个包一个全新进程中：

1. 在加载包之前分配好运行要用的所有输入；
2. 运行同样的调用序列，然后再重复一遍；
3. 对存活的进程采集 `footprint -p` 和 `vmmap --summary`。

相对"无状态包已加载 + 所有输入已分配"的增量，单位 MiB：

| 包 | 加载 | 预填充 24 | B16 S32 | 扫到 S=64 | B16 S128 | 整个序列重复 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 `int8-flex` | 38.1 | — | 39.7 | 40.0 | 40.0 | 40.0 |
| `dec-flex` | 209.1 | — | 224.2 | 296.3 | 395.5 | 396.5 |
| `dec-b16`（S 枚举） | 51.3 | — | 53.7 | 53.8 | 53.8 | 53.8 |
| `dec-b16-s32` | 22.3 | — | 24.8 | 24.8 | — | 24.8 |
| `dec-b16-s128` | 21.8 | — | 24.3 | 24.3 | 24.3 | 24.3 |
| `unified` | 39.3 | 40.4 | 41.0 | 41.0 | 41.0 | 41.0 |
| `unified-scored` | 39.5 | 40.5 | 40.3 | 40.3 | 40.3 | 40.3 |

由此可见：

- **v1 和可变 T 的 v2 包，Core ML 自身的开销约 38–41 MiB，而且是平的。**
  - 包加载后，没有任何部分随 B 或 S 增长。
  - 因此在 Mac 上只读缓存相比 v1 什么也没省：v1 那些缓存大小的临时缓冲区并不会在调用之间保留。
  - Vime 报告的真机数字（Core ML 约 27 MiB）形态相同。Mac 的数字更大，不能当作真机预测。
- **T=1 包的行为不同。**
  - 它们加载时约 209 MiB，然后每个新的 (B,S) 形状再增加约 15–100 MiB。只有每个形状都出现过一次之后才停止增长（重复序列只增加 1 MiB）。
  - 在 `footprint`/`vmmap` 中，`dec-flex` 的 `Malloc Large` + `Malloc Small` 比 `unified` 多约 350 MiB（整个进程的 malloc 为 762 对 406 MB）。
  - 权重文件只映射了 77 MB，`unified` 和 v1 为 154 MB。
  - 这一模式符合 Core ML 对这类图把反量化后的权重放到堆上、并为每个形状缓存一份执行计划。这是从计数器推断的，工具本身没有这样说明。
- **两个图几乎相同。** MIL op 计数（`logs/03c-op-histogram.log`）：
  - `dec-flex` 997 个 op，`unified` 1026 个 op。
  - 同样的 47 个 `constexpr_blockwise_shift_scale`、46 个 `linear` 和 36 个 `matmul`。
  - 唯一的差别是静态 T=1 对可变 T 维度，所以是静态的 1 触发了另一条运行时路径。
- **固定形状**加载最轻（约 22 MiB）。无论固定的 S 是多少，其内存占用都保持平稳，所以 `memory` 表中固定包随 S 的大幅增长来自输入分配。
  - 固定形状的代价是每次调用都要填充到 B=16 和固定的 S，这体现在延迟上（见下文）。
  - S 枚举的 `dec-b16` 占用 52–54 MiB。

## 4. Mac 延迟

设置：`measure`，已加载无状态包，200 次重复，`CPU_ONLY`。数值为 p50 ms（p95 在 JSON 文件中）。固定和枚举形状的包把行填充到自己的形状。

### 解码，p50 ms（B × S）

| 变体 | B1 S16 | B1 S128 | B8 S16 | B8 S32 | B8 S64 | B8 S128 | B16 S16 | B16 S32 | B16 S64 | B16 S128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dec-flex` | 2.32 | 3.55 | 4.59 | 5.44 | 9.12 | 13.65 | 6.56 | 8.07 | 15.74 | 24.16 |
| `dec-enum` | 1.89 | 3.09 | 4.10 | 4.89 | 8.42 | 12.65 | 5.85 | 7.56 | 14.63 | 51.29 |
| `dec-b16`（填充到 B16） | 6.04 | 9.06 | 5.72 | 6.30 | 6.22 | 8.89 | 6.95 | 5.07 | 5.66 | 8.22 |
| `dec-b16-s32` | 4.81 | — | 5.63 | 5.07 | — | — | 4.80 | 4.96 | — | — |
| `dec-b16-s64` | 5.91 | — | 5.45 | 4.45 | 4.43 | — | 4.12 | 4.16 | 3.71 | — |
| `dec-b16-s128` | 7.89 | 6.45 | 7.80 | 6.95 | 8.43 | 6.31 | 7.97 | 6.96 | 7.71 | 8.46 |
| `scored` K=149 | 2.98 | 3.54 | 4.89 | 5.68 | 9.15 | 13.56 | 6.00 | 8.38 | 16.32 | 24.16 |
| `unified` | 2.01 | 2.07 | 2.56 | 3.12 | 3.27 | 3.55 | 3.12 | 3.66 | 4.15 | 4.26 |
| `unified-scored` K=149 | 2.01 | 2.30 | 2.99 | 3.09 | 3.03 | 3.46 | 3.19 | 3.73 | 3.81 | 4.37 |
| `unified-scored` K=450 | 2.02 | 2.43 | 3.02 | 2.71 | 3.19 | 3.10 | 3.54 | 3.35 | 4.09 | 4.32 |
| v1 `int8-flex`（第一份报告） | — | — | — | — | — | — | 3.35 | 3.76 | — | — |

对固定 S 的包，S 小于包自身 S 的列是同一次填充后的调用。

`scored` 在 K=450 时与 K=149 在噪声范围内相同（如 B16 S32 8.09 对 8.38）。完整网格在 `int8-<name>-latency.json`；摘录在 `latency-table.txt`。

### Prompt 预填充，p50 ms

| 变体 | 8 token | 16 token | 32 token | 方式 |
| --- | ---: | ---: | ---: | --- |
| `unified` | 2.15 | 2.34 | 4.08 | 一次调用 |
| `unified-scored` | 2.31 | 2.69 | 3.76 | 一次调用 |
| `dec-flex` | 47.0 | 92.3 | 187.1 | 每次调用一个 token，B=1 |
| `dec-b16-s32` | 43.1 | 97.6 | 163.1 | 每次调用一个 token，填充到 B=16 |

### `int8-dec-flex` 与 v1 `int8-flex` 交替测量（3 轮，p50 ms；`latency-interleaved/`）

| 行 | S | `dec-flex` 各轮 | 中位数 | v1 各轮 | 中位数 | v2 / v1 |
| ---: | ---: | --- | ---: | --- | ---: | ---: |
| 1 | 16 | 2.00 / 2.43 / 2.13 | 2.13 | 1.88 / 1.66 / 1.89 | 1.88 | 1.13 |
| 1 | 32 | 2.00 / 2.55 / 2.01 | 2.01 | 1.89 / 1.70 / 1.92 | 1.89 | 1.06 |
| 4 | 16 | 2.90 / 3.30 / 2.83 | 2.90 | 2.51 / 2.65 / 2.36 | 2.51 | 1.16 |
| 4 | 32 | 3.48 / 3.65 / 3.58 | 3.58 | 2.48 / 2.86 / 2.52 | 2.52 | 1.42 |
| 8 | 16 | 4.52 / 4.97 / 4.69 | 4.69 | 2.72 / 2.92 / 2.94 | 2.92 | 1.61 |
| 8 | 32 | 4.97 / 5.86 / 4.95 | 4.97 | 3.23 / 2.90 / 3.26 | 3.23 | 1.54 |
| 16 | 16 | 6.27 / 6.66 / 6.58 | 6.58 | 3.43 / 3.16 / 3.34 | 3.34 | 1.97 |
| 16 | 32 | 7.64 / 8.20 / 7.83 | 7.83 | 3.66 / 4.00 / 3.61 | 3.66 | 2.14 |

各轮的变慢一致，不是噪声。在这条运行时路径上，T=1 的 v2 图比 v1 慢。可变 T 的 `unified`（B16 S32 3.66 ms）与 v1 一样快。

## 5. 搜索检查（r2 fixture，125 例，beam 16，top 10）

- "对 v1 批量"是与用 v1 INT8 批量包和 chunk 预填充跑的同一搜索对比，后者等于无状态 INT8 搜索（第一份报告）。
- "对 FP32 fixture"是与随附的 FP32 fixture 对比。
- 预填充调用数为每例平均。

| 包 | Prompt | 对 v1 批量（完全一致 / 集合相同 / 最大分数差） | 对 FP32 fixture（完全一致 / 集合相同 / top-1 / 最大差） | p50 / p95 ms | 预填充调用 |
| --- | --- | --- | --- | --- | ---: |
| `int8-unified-scored` | 包，一次调用 | 125 / 125 / 3.3e-5 | 46 / 86 / 125 / 0.565 | 79 / 212 | 1 |
| `int8-unified` | 包，一次调用 | 125 / 125 / 2.3e-5 | 46 / 86 / 125 / 0.565 | 106 / 234 | 1 |
| `int8-dec-flex` | 包，逐 token | 125 / 125 / 2.7e-5 | 46 / 86 / 125 / 0.565 | 197 / 391 | 4.44 |
| `int8-dec-flex` | chunk | 125 / 125 / 2.1e-5 | 同上 | 209 / 455 | 1 |
| `int8-dec-enum` | 包，逐 token | 125 / 125 / 2.6e-5 | 同上 | 325 / 877 | 4.44 |
| `int8-dec-enum` | chunk | 125 / 125 / 2.2e-5 | 同上 | 206 / 481 | 1 |
| `int8-dec-b16` | 包，逐 token | 125 / 125 / 2.1e-5 | 同上 | 192 / 411 | 4.44 |
| `int8-dec-b16` | chunk | 125 / 125 / 2.0e-5 | 同上 | 140 / 330 | 1 |
| `int8-dec-b16-s128` | 包，逐 token | 125 / 125 / 2.2e-5 | 同上 | 196 / 486 | 4.44 |
| `int8-dec-b16-s64` | 包，逐 token | 125 / 125 / 2.2e-5 | 同上 | 120 / 258 | 4.44 |
| `int8-dec-b16-s32` | 包，逐 token | **失败**（`logs/05b-search-dec-b16-s32.log`） | — | — | — |
| `int8-scored` | 包，逐 token | 125 / 125 / 5.3e-5 | 同上 | 265 / 586 | 4.44 |
| `int8-scored` | chunk | **拒绝**（`logs/05-search-scored-chunk.log`） | — | — | — |
| `fp32-dec-flex` | 包，逐 token | — | **125 / 125 / 125 / 4.4e-5** | 260 / 532 | 4.44 |
| `fp32-unified` | 包，一次调用 | — | **125 / 125 / 125 / 4.5e-5** | 138 / 298 | 1 |

说明：

- 每个 INT8 变体对 FP32 fixture 的数字都相同，也与第一份报告中 v1 批量和无状态 INT8 的运行相同（46/86/125）。这一差异来自 INT8 本身，而不是批量包。
- `scored --prompt chunk` 按设计被拒绝。chunk 包返回完整 logits，而后端需要可变 T 的 scored 包，或用 scored 输出头逐 token 预填充（*"Use a flexible-T package or token-by-token prefill with the scored head."*）。错误信息作为证据保留。
- `dec-b16-s64` 能容纳所有 r2 用例。
- `dec-b16-s32` 失败：部分用例需要超过 32 个槽位。
  - 后端找不到合适的 S，在空的 `min()` 上报错（*"min() arg is an empty sequence"*），而不是给出清楚的信息。
  - 日志已保留。因此固定 S=32 的包无法单独使用。

## 6. 选择：`int8-unified-scored`

按任务说明的优先顺序：

1. **在已加载无状态包的前提下，新增内存占用最低。**
   - 全范围包中：整个序列后（归因测量）`unified-scored` 40.3 MiB、`unified` 41.0 MiB，v1 为 40.0 MiB 再加 chunk 包 13 MiB。
   - 固定 S 的包更轻（约 24 MiB），但各有问题：
     - `dec-b16-s32` 装不下 r2 的上下文。
     - `dec-b16-s64` 装得下，但把上下文加按键限制在 64 个 token，需要 chunk 包或逐 token 预填充（16 个 token 约 100 ms），并且每次调用都填充到 B=16（即使 B=1 也要 3.7–6.5 ms）。
     - `dec-b16-s128` 在 Mac 上每次调用 6.5–8.5 ms，在 13 Pro 上约 13–17 ms，超出目标。
   - `dec-b16-s64` 是唯一能省内存的选项（比 `unified-scored` 少约 16 MiB）。没有选它，因为它不满足第 2 条，也没有 128 token 的范围。
   - S 枚举的 `dec-b16`（54 MiB）比 `unified` 更重。
   - 所有 T=1 的可变或枚举包都被排除（≥ 200 MiB）。
2. **不需要 chunk 包。** 两个 `unified` 变体都一次调用完成 prompt 预填充（2–4 ms）。用它们，键盘只需要无状态包和这一个包。
3. **延迟。**
   - `unified-scored` 在 Mac 上：S 直到 128，K=149 和 K=450 下 B=16 ≤ 4.4 ms、B=8 ≤ 3.5 ms。
   - 按 Mac 约 2 倍的速度换算，13 Pro 上 B=16 S128 约 9 ms，S ≤ 64 时约 7–8 ms，目标为 8 ms。B=8 翻倍后只在 S≈32–64 以内保持 5 ms 以内（Mac 上 3.1–3.5 ms）。需要真机确认。

两个 `unified` 变体之间，内存占用和解码延迟在噪声范围内持平。选择 `unified-scored` 是因为：

- 端到端搜索更快（p50 79 对 106 ms；客户端不再需要每次调用拷贝 16×12,288 个 logits 并做 softmax）；
- 每次调用的输出是 16×(K+1) 个浮点数，而不是 16×12,288。

代价：

- 客户端必须传入每个桶允许的 id 列表（K ≤ 1024；r2 最大为 536）。
- 客户端只拿到这些 id 的对数概率。

`unified`（`$O/int8-unified`，搜索 125/125）是可直接替换的备选，接口为完整 logits，内存相同。

fixture 阶段报告包对无状态最大 1.24e-5（`logs/06-fixture.log`）。

## 7. 客户端资源与发布副本

- `$O/ios-kvbatch-v2/`：
  - `Resources/TinyChineseV1KVBatchV2INT8.mlmodelc`；
  - `Resources/VimeLMManifestZhV1KVBatchV2.json`（格式 `vime_ios_lm_zh_v1_kv_batch_v2`，段 `kv_batch_v2`，记录输出头 `scored` 和可变 token 数）；
  - `KVBatchFixture-*` 和 `manifest.json`。

  v1 资源未动。
- 发布副本：`artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvbatch-v2/`，是 `$O/int8-unified-scored` 的字节相同副本（`diff -r` 无差异，22 MiB）。`RELEASE.json` 和 `SHA256SUMS.txt` 未变。

**Windows 端后续（2026-11-15）。** 结果 ZIP 已与其 SHA256SUMS 核对（251 个文件）。发布副本与 `int8-unified-scored` 字节相同，并以 v2 格式通过 `verify_package`。发布清单现在列出 `ios18-int8-block32-kvbatch-v2/` 的四个文件和 `coreml_manifest_sha256.kvbatch_v2_int8` = `33c90e2a…`；README 描述了 v2 包。`verify_release.py --hashes` 在 374 个文件上通过。Hugging Face 发布在 revision `8dd4f19729e921a6bd6844a79743355343569784`（父 revision `ce35c931…`）原地更新；重新下载该 revision 后通过 `verify_release.py --hashes`。Mac 上对 `shape_spec`/`shapes_of` 的改动按原样并入 `zh_v1_batch2.py`。当固定或枚举形状包没有任何形状适合某次调用时，后端现在会给出清楚的错误，而不是 `dec-b16-s32` 上那样的空 `min()`。

任务说明对 v2 的推理是：v1 的逐层缓存拷贝让 Core ML 分配缓存大小的临时缓冲区。Mac 上的数字不支持这一点：对可变 T 的图，只读缓存没有任何变化（约 40 MiB，与 v1 相同），T=1 的图则差得多。v2 在真机上省下的是 chunk 包，而不是 Core ML 自身的内存。只有固定形状的包更轻，代价见 §6。

## 8. iPhone

按任务说明本会话未做；Vime 用自己的测试宿主测量。真机上要检查：

- T=1 的图是否有同样的堆增长；
- `unified-scored` 在 Mac 上约 40 MiB 的数字，是否对应 Vime 测得的 v1 Core ML 约 27 MiB。

**Vime 的真机结果（2026-11-15），`unified-scored` 对 v1：**

| | iPhone 13 Pro v1 → v2 | iPhone 16 Pro Max v1 → v2 |
| --- | --- | --- |
| Prompt 预填充 8/16/32 token | 21 / 44 / 89 → 4.2 / 4.4 / 6.2 ms | 15 / 41 / 102 → 2.2 / 2.3 / 3.1 ms |
| 每次按键 p50 / p95 / 最大 | 10.3 / 24.9 / 29.8 → 10.4 / 25.7 / 30.1 ms | 6.5 / 16.0 / 20.2 → 6.5 / 15.6 / 19.4 ms |
| 在无状态包之上的增量 | +39.3 → +39.8 MiB | +39.3 → +39.9 MiB |

- VimeML fixture 在 B=16/7/4/1 下都在 3e-4 以内；r2 搜索在全部 125 例中与 v1 INT8 搜索相同且顺序一致；62 个模拟器测试通过。
- v2 只在预填充上更快（5–30 倍）。每次按键的延迟和内存占用与 v1 相同；省下的是不再加载的 chunk 包。早先 v1 +35.8 MiB 的数字来自另一次运行；本次运行测得 v1 为 +39.3 MiB。
- 每次按键剩余时间的大部分在客户端组装桶（13 Pro 上约 6.7 ms）。
- **被掩码的槽位必须是有限值。** 新建的 `MLMultiArray` 不会清零。prompt 未使用的缓存槽位中的 NaN 传到了输出（被掩码的分数仍会乘以 value 缓存，而 0·NaN 是 NaN）。跨形状调用几十次后，所有预填充输出都变成了 NaN。Vime 现在会把这个输入清零（400 轮随机形状无 NaN）。接口文档中"这些槽位可以是垃圾值"的说法已更正为：任意有限值，不能是 NaN 或 Inf。chunk KV 包及其客户端封装同样适用。

## 对 zh_v1_batch2.py 的改动

完整 diff 在 `$O/zh_v1_batch2.py.diff`；改动后的文件包含在 ZIP 中。改动分两部分。

1. **`shape_spec` 的枚举分支。**
   - 构建 (B,S) 对的列表。
   - 缓存为每一对给一个枚举形状。
   - `input_ids` 和 `position` 给相同数量的形状，第 i 个重复第 i 对的 B，这是 iOS 18 的要求。
   - 只有一个批大小时，`input_ids` 和 `position` 改为普通固定形状，因为 `EnumeratedShapes` 至少需要两个。
   - 少于两对时给出清楚的错误。
2. **`shapes_of`。** 识别固定 S 备选包的 `"S fixed <n>"` manifest，让 Core ML 后端以及 `measure`/`search` 阶段正好传入那个 S，而不是把 S 当作可变。

改动后重跑 `check` 并通过（`check-after-shape-fix.json`、`logs/01b-check-after-shape-fix.log`），单元测试也通过。

## 文件

| 路径 | 内容 |
| --- | --- |
| `check.json`、`check-after-shape-fix.json` | PyTorch 检查 |
| `fp32-<name>/`、`int8-<name>/`、`*-align.json` | 包与门槛（6 个变体 + 3 个备选） |
| `memory.json` | `memory` 阶段（12 种配置 × 3 个进程） |
| `memory-attribution/` | 预分配输入的运行、每个包的 `footprint` 和 `vmmap` 文本、V2/V4 检查 |
| `int8-<name>-latency.json`、`latency-interleaved/`、`latency-table.txt` | 延迟 |
| `search-<name>[-chunk]/`、`search-fp32-*/`、`search-table.txt` | 搜索检查 |
| `fixture/`、`ios-kvbatch-v2/` | Vime fixture 与客户端资源 |
| `tools/` | `memory_attribution.py`、`run_latency.sh`、`run_search.sh` |
| `logs/` | 每个阶段，包括两次失败的转换和被拒绝的 scored+chunk 搜索 |
| `zh_v1_batch2.py.diff` | 源码改动 |
