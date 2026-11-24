# vimeml-tiny-zh-v1 句子打分 Core ML 结果（sequence 输出头）—— 2026-11-16

按 [sequence 输出头任务说明](../../mac/mac-zh-v1-kvsequence.md)执行。沿用前几次任务的检出目录（`~/Documents/Sources/VimeML-zh-kvbatch`），fast-forward 拉到 `main` `fcc1750`。

**拉取之前**

- 检出目录里还留着 v2 任务未提交的改动：`zh_v1_batch2.py` 的形状修复和未跟踪的 v2 报告。上游已经收录了这两者（`a1a3a63`）。
- 为了能 fast-forward，把本地副本、diff 和原始的未跟踪报告存到了 `outputs/deployment/mac-20261115-v2-local-backup/`，然后把文件恢复到 `HEAD`。
- 本地没有丢失内容。上游的 v2 报告与 Mac 版只差几行编辑性改动。

**未改动**

- checkpoint、tokenizer、INT8 配方（block32，阈值 2048，47 个矩阵）和打分规则。
- `RELEASE.json` 和 `SHA256SUMS.txt`（仍为 `899ca51a…` / `58096d45…`，记录在 `logs/10-release-meta-before.sha`）。
- `src/vimeml/deployment/zh_v1_batch2.py`（没有转换或对齐问题需要修复）。
- Mac 上没有提交任何内容。

**环境**

- 硬件与系统：Apple M3，16 GiB，macOS 27.3（26C5035r），Xcode 27.3（27C5020n）。
- Python：uv venv 中的 3.11.17，PyTorch 2.7.1，coremltools 9.0，NumPy 2.3.5，sentencepiece 0.2.1。
- Core ML：mlprogram，iOS 18 目标，FP32 计算，`CPU_ONLY`。
- 电源与负载：接交流电，`pmset -g therm` 没有警告。测量期间平均负载约 4。

所有输出在 `outputs/deployment/zh-v1-kvsequence/`（`$O`）下：各阶段日志在 `logs/`，辅助脚本在 `tools/`。没有覆盖任何输出，没有阶段失败。

## 摘要

- **门槛：** 每个阶段一次通过。
  - `check` 列出 `sequence-Tflex` 已 trace 并检查。
  - FP32 对 PyTorch 对齐和 Vime 的 INT8 对无状态包门槛，最大都是 3.34e-5（容差：每个位置 3e-4，每个和 1e-3）。
- **两个问题：**
  1. **B=16、T=16 延迟：** 一次 sequence 调用 **17.5–18.4 ms**。B=16 下链式调用 16 次 `unified-scored` 解码 **54.5–62.2 ms**。对 16–64 个槽位的前缀，**快 3.1–3.4 倍**（Mac p50，在一个进程中交替运行两个包计时）。
  2. **内存随 T 的变化：** **确实随 T 增长**，曲线见下文。以下是预分配输入后的 Core ML 数字，单位为相对"无状态包 + 输入"增加的 MiB。
     - T ≤ 8：与 `unified-scored` 相差约 1 MiB 以内（34–40 对 40.8）。
     - T=16：+9 MiB（49.9），峰值 66.7。
     - T=32：+23 MiB（63.4），峰值 103。
     - T=64：+63 MiB（104），峰值 169。
- **Fixture：** 8 例，39 个候选，13 次调用。
  - 包对无状态 INT8：每个和最大 3e-5。
  - 包对 PyTorch FP32：每个和最大 0.17，来自 INT8 量化。
  - 8/8 例候选排序与 FP32 相同。
- **决定：作为附加资源发布，每次调用限制 T。**
  - **T ≤ 8** 时内存占用符合任务说明"相差几 MiB 以内"的规则。
  - **T=16** 多保留约 9 MiB，峰值多约 25 MiB。
  - 建议上限：**8**；如果 Vime 能接受多出的内存则用 16。
  - 即使拆成 T=8 的调用，B=16 一批 16 token 续写在 Mac 上约 22 ms，16 次解码调用约 55 ms。
  - 未测量：两个批量包同时加载。如果句子打分和搜索共存，这就是发布配置，应在真机上检查。

## 1. PyTorch 检查（`torch-check.json`，`logs/01-check.log`）

| 检查 | 结果 |
| --- | ---: |
| 解码（eager 和 traced） | 最大 5.01e-5 |
| scored 输出头对 log-softmax | 1.91e-6 |
| sequence 输出头逐位置 | 最大 2.19e-5 |
| `score_continuations` 和 | 最大 1.91e-5 |
| trace 的图 | `logits-T1`、`logits-Tflex`、`scored-T1`、`sequence-Tflex`：已 trace 并检查 |

运行输出 `passed`。sequence 输出头的数字与 Windows 端检查一致（每个位置 1.1e-5，每个和 2.7e-5）。

## 2–5. 转换、压缩与对齐

- `convert $O/fp32-seq --tokens flex --head sequence` 构建了 B 1..16、T 1..128、S 1..128（全部为 RangeDim）的包，新增输入 `targets` int32 `[B,T]`（`logs/02-convert-fp32-seq.log`）。
- INT8 包（`int8-seq`，22 MiB）用不变的配方从它压缩得到（`logs/04-compress-int8-seq.log`）。

每项检查的最大绝对误差。FP32 对齐是对 PyTorch FP32；INT8 门槛是对无状态 INT8 包（Vime 的门槛）。每一行容差外的位置或续写都为 0。

| 检查 | 行 | 位置 | FP32 对齐 | INT8 门槛 |
| --- | ---: | ---: | ---: | ---: |
| Sequence，逐位置，B=4（prompt 2–17，续写 1–7） | 4 | 53 | 1.05e-5 | 9.5e-6 |
| Sequence，逐位置，B=1 | 1 | 3 | 1.34e-5 | 5.7e-6 |
| Sequence，逐位置，B=16（填充行，不同位置） | 16 | 217 | 1.53e-5 | 1.43e-5 |
| `score_continuations`：4 次调用 37 个续写，每个和 | — | — | 3.34e-5 | 3.34e-5 |
| **结果** | | | **通过，最大 3.34e-5** | **通过，最大 3.34e-5** |

日志：`logs/03-align-fp32-seq.log`、`logs/05-align-int8-seq.log`；JSON：`fp32-seq-align.json`、`int8-seq-align.json`。

## 6. Mac 延迟

设置：`measure`，已加载无状态包，200 次重复，`CPU_ONLY`。数值为 p50，单位 ms；p95 在 `latency-seq.json`。

- 一次打分调用是 B 行 × T 个续写 token，基于 S 个缓存槽位，`position = S`。
- `measure` 的网格跳过 S + T > 128 的组合，所以每个 T 都没有 S=128。

### Sequence 包，打分调用

| B \ (T, S) | T4 S16 | T4 S64 | T8 S16 | T8 S64 | T16 S16 | T16 S32 | T16 S64 | T32 S16 | T32 S64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.13 | 3.53 | 3.60 | 3.55 | 3.27 | 2.92 | 3.06 | 4.38 | 4.79 |
| 4 | — | — | — | — | 6.37 | 6.28 | 6.65 | — | — |
| 8 | — | — | — | — | 10.10 | 10.15 | 10.60 | — | — |
| 16 | 6.75 | 7.39 | 10.53 | 11.22 | **17.75** | **17.81** | **18.36** | 31.18 | 32.86 |

一次调用完成 prompt 预填充（B=1）：8 token 2.63 ms，16 token 3.03 ms，32 token 4.25 ms。当天重新测量的 `unified-scored`：2.04 / 2.32 / 3.25 ms。

B=16 时开销大致随 B×T 线性增长：每个打分位置约 0.07 ms。这与每个位置都要做输出头矩阵乘和 12,288 个 logits 的 log-softmax 相符。

### `unified-scored`，B=16 解码（`latency-unified-scored.json`，重新测量）

| S | 16 | 32 | 64 | 128 |
| --- | ---: | ---: | ---: | ---: |
| K=149 | 3.10 | 3.21 | 3.67 | 4.04 |
| K=450 | 3.04 | 3.28 | 3.60 | 4.12 |

### 对比：一次 sequence 调用 对 16 次解码调用，B=16，T=16（`latency-seq-vs-decode16.json`）

两个包在无状态包之后加载到同一个进程中，输入预先分配（`tools/compare_seq_vs_decode.py`）。

- **Sequence 侧：** 一次调用，`input_ids`/`targets` 为 `[16,16]`，`position = P`。
- **解码侧：** 链式调用 16 次 `unified-scored`，`[16,1]`，K=149。第 i 次调用位于 `position = P+i`，S = P+i，所以缓存每次调用增长一个槽位。
- **计时：** 只计 `predict`。客户端在解码调用之间写缓存的时间不计入，这只会对解码侧有利。
- **轮次：** 每轮交替两侧；3 次运行，每次 100 轮。

| 共享前缀 P | Sequence 一次调用（每次运行的 p50） | 中位数 | 16 次解码调用（每次运行的 p50） | 中位数 | 倍数 |
| ---: | --- | ---: | --- | ---: | ---: |
| 16 | 17.64 / 17.41 / 17.50 | **17.50** | 54.78 / 53.52 / 54.50 | **54.50** | **3.12×** |
| 32 | 17.90 / 17.63 / 17.69 | **17.69** | 57.28 / 56.52 / 56.79 | **56.79** | **3.21×** |
| 64 | 18.38 / 18.66 / 18.28 | **18.38** | 62.21 / 62.73 / 61.73 | **62.21** | **3.38×** |

这些与单独的 `measure` 运行一致：sequence B16 T16 为 17.8–18.4 ms，16 × `unified-scored` B16 解码时间为 50–65 ms。

更小的 T 用延迟换内存（见第 7 节）：

- B=16、T=8 每次调用 10.5–11.2 ms，两次 T=8 调用约 22 ms。
- 仍比 16 次解码调用好 2.5 倍。

## 7. 内存

### `memory` 阶段（每种配置 3 个全新进程，中位数；`memory.json`，`logs/07-memory.log`）

数值为相对"已加载并运行无状态包之后的进程"增加的 MiB。两个包都运行 v2 的调用序列（T=1）；sequence 包之后再加一次 B=16、T=32、S=64 的打分调用。

| 配置 | 加载 | 预填充 24 | 首次 B16 S32 | 扫到 S=64 | B16 S128 | 打分 B16 T32 S64 | 峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `seq` | 39.0 | 40.3 | 60.4 | 99.2 | 180.0 | 220.2 | 266.6 |
| `unified-scored` | 39.1 | 40.4 | 60.5 | 98.9 | 180.0 | — | 181.5 |

如 v2 报告所示，这个阶段在测量窗口内创建每次调用的 NumPy 输入，所以加载后的增长主要是 Python 端的输入；一对 B=16、S=128 的缓存就是 56 MiB。两个包在 v2 序列上完全相同。最后的打分调用使当前内存占用增加 40 MiB，峰值增加 85 MiB。

### 内存随 T 的变化，输入预分配（`memory-over-t/`，`tools/memory_over_t.py`）

每个包在 3 个全新进程中：

1. 加载并运行无状态包；
2. 分配本次运行的全部输入；
3. 加载包；
4. 预填充 24，然后运行 v2 序列；
5. 仅 sequence 包：B=16、S=64 的打分调用，T 从 1 升到 64，再在 S=16 和 S=96 下做 T=16/32；
6. 全部重复一次。

数值为相对"已加载无状态包 + 已分配输入"增加的 MiB 中位数。"峰值"是到该时点为止的生命周期最大值。

| 时点 | `seq` 占用 | `seq` 峰值 | `unified-scored` 占用 | `unified-scored` 峰值 |
| --- | ---: | ---: | ---: | ---: |
| 加载 | 32.8 | 32.8 | 39.5 | 39.6 |
| 预填充 24 | 34.1 | 36.4 | 40.9 | 41.7 |
| v2 解码序列（T=1，到 B16 S128） | 34.0 | 36.9 | 40.8 | 42.3 |
| 打分 T=1 | 34.0 | 36.9 | | |
| 打分 T=2 | 34.6 | 38.0 | | |
| 打分 T=4 | 36.3 | 41.9 | | |
| 打分 T=8 | 39.7 | 50.1 | | |
| **打分 T=16** | **49.9** | **66.7** | | |
| 打分 T=32 | 63.4 | 103.2 | | |
| 打分 T=64 | 104.0 | 169.2 | | |
| S=16 和 S=96 下 T=16/32 | 90.7 | 182.9 | | |
| 整个序列重复一次 | 90.1 | 195.7 | 40.8 | 42.5 |

解读曲线：

- **T ≤ 8 没有额外开销。** sequence 包的内存占用不超过 `unified-scored` 的约 40 MiB。
  - 两个包加载时 33 对 39.5 MiB 的差异在进程间的波动范围内。在 `memory` 阶段两者加载都是 39 MiB。
- **从 T=16 起，内存占用随见过的最大 T 增长**，约为 `[16,T,12288]` FP32 logits 大小的 1–1.5 倍（T=16 时 12 MiB，T=32 时 24，T=64 时 48）。
  - Core ML 在调用后保留这部分内存。
  - 之后的小调用只归还一部分（104 → 90.7 MiB）。
  - 重复一遍不再增加。
- **峰值上升得更快**，比调用时保留的占用高出约两倍 logits 大小。这是瞬时的 logits 和 log-softmax 缓冲区。
- `footprint` 显示的权重映射与 `unified-scored` 相同（154 MB 映射文件，clean）。与 v2 报告中 T=1 的包不同，权重没有被复制到堆上。额外的内存是 `Malloc Large`，与按 B×T×12,288 分配的缓冲区相符。

**给 Vime 的单次调用上限：**

| T 上限 | 相对 `unified-scored` 的占用 | 峰值 | B=16 调用 |
| ---: | ---: | ---: | ---: |
| 8 | 约 0 | 约 50 MiB | 约 11 ms |
| 16 | +9 MiB | 约 67 MiB | 约 18 ms |

更长的续写可以按任务说明允许的方式拆成多次调用：第二次调用从缓存读取第一次调用的 `new_key`/`new_value`。

## 8. Fixture（`fixture/`、`fixture-summary.txt`、`logs/08-fixture.log`）

`fixture $O/int8-seq $O/fixture` 写出含 8 例的 `sequence.jsonl`。每例包含候选、共享前缀 id、每个候选的续写 id，以及包、无状态 INT8 包和 PyTorch FP32 各自给出的每个候选的和。规则是生产用的 `score_item`。

| 用例 | 候选数 | 调用数 | 最长续写 | 包对无状态 | 包对 FP32 | 排序 = FP32 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 3 | 1 | 4 | 0 | 0.079 | 是 |
| 2 | 3 | 2 | 6 | 2e-5 | 0.091 | 是 |
| 3 | 3 | 1 | 7 | 1e-5 | 0.127 | 是 |
| 4 | 3 | 1 | 8 | 1e-5 | 0.073 | 是 |
| 5 | 3 | 2 | 5 | 2e-5 | 0.170 | 是 |
| 6 | 3 | 1 | 6 | 1e-5 | 0.131 | 是 |
| 7 | 3 | 2 | 16 | 2e-5 | 0.084 | 是 |
| 8 | 18 | 3 | 9 | 3e-5 | 0.153 | 是 |
| **全部** | **39** | **13** | | **最大 2.36e-5** | **最大 0.170** | **8/8（top-1 和完整排序）** |

- 包对无状态的一致性远好于任务说明要求的每个和约 1e-3。
- 与 FP32 的差异（整句 ≤ 0.17 nats）来自 INT8 量化，与 v2 搜索检查中的效应相同。这里没有改变任何排序。
- 3 个候选的用例调用数为 2，表示一次预填充加一次打分调用。

## 9. 客户端资源与发布副本

- `$O/ios-kvsequence/`（`logs/09-package-ios.log`）：
  - `Resources/TinyChineseV1KVSequenceINT8.mlmodelc`；
  - `Resources/VimeLMManifestZhV1KVSequence.json`（`kv_sequence` 段）；
  - `KVSequenceFixture-sequence.jsonl`、`KVSequenceFixture-meta.json` 和 `manifest.json`。

  v1 和 v2 资源未动。
- 发布副本：`artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvsequence/`，是 `$O/int8-seq` 的逐字节副本（`diff -r` 无差异，22 MiB）。`RELEASE.json` 和 `SHA256SUMS.txt` 未变。

**Windows 端后续（2026-11-16）。** 结果 ZIP 已与其 SHA256SUMS 核对（65 个文件）。发布副本与 `int8-seq` 逐字节相同，并以 v2 格式通过 `verify_package`。它的 `weight.bin` 与批量 KV v2 包的逐字节相同。发布清单现在列出 `ios18-int8-block32-kvsequence/` 的四个文件和 `coreml_manifest_sha256.kvsequence_int8` = `c7b9e791…`；README 描述了 sequence 包和单次调用的 T 上限。`verify_release.py --hashes` 在 378 个文件上通过。Hugging Face 发布在 revision `ff3dab5f2b730fb8e9a5bd10addc8ac6414a5b6e`（父 revision `8dd4f197…`）原地更新；重新下载该 revision 后通过 `verify_release.py --hashes`。

## 10. iPhone

按任务说明，本次未做。Vime 应在真机上检查：

- 上面的 T 曲线（至少 T = 8、16、32）；
- `unified-scored` 和 sequence 包**同时加载**时的内存占用。Mac 上没有测；两者是不同的权重文件，预计加载开销大致相加。

## 决定

**把 `int8-seq` 作为句子打分的附加资源发布；搜索继续用 `unified-scored`。**

- 两项对齐都通过，fixture 一致，延迟提升约 3 倍。
- 只有限制每次调用的 T，内存占用才能保持在 `unified-scored` 的"几 MiB 以内"。**建议上限：T ≤ 8**，保留开销约 0，峰值约 +10 MiB。
- 如果能接受约 9 MiB 保留和约 25 MiB 峰值，可以改用 **T ≤ 16**；每批 16 token 少一次调用（18 ms 对约 22 ms）。
- 不限 T（最多 64）要保留约 60 MiB、峰值约 130 MiB，不推荐。

## 对 zh_v1_batch2.py 的改动

无。文件停留在 `fcc1750`。v2 任务的本地改动已经在上游；其备份在 `outputs/deployment/mac-20261115-v2-local-backup/`，不属于本任务。

## 文件

| 路径 | 内容 |
| --- | --- |
| `torch-check.json` | PyTorch 检查 |
| `fp32-seq/`、`int8-seq/`、`fp32-seq-align.json`、`int8-seq-align.json` | 包与对齐 |
| `latency-seq.json`、`latency-unified-scored.json`、`latency-seq-vs-decode16.json` | 延迟 |
| `memory.json`、`memory-over-t/` | 内存（`memory` 阶段；T 曲线，每次运行附 `footprint`/`vmmap` 文本） |
| `fixture/`、`fixture-summary.txt`、`ios-kvsequence/` | fixture 与客户端资源 |
| `tools/` | `run_build.sh`、`run_latency.sh`、`compare_seq_vs_decode.py`、`memory_over_t.py` |
| `logs/` | 各阶段日志 |
