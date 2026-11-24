# vimeml-tiny-zh-v1 批量单步 KV 解码 Core ML 结果 —— 2026-11-15

按[交接任务说明](../../mac/mac-zh-v1-kvbatch.md)执行。检出目录由 `setup_mac.py` 从 `vimeml-zh-kvbatch-mac-20261115.zip` 创建，`main` `a8440c5`（已核对记录的提交），每个载荷文件都与 `handoff-manifest.json` 核对过。源模型 `vimeml-tiny-zh-v1`（checkpoint `44c423a1…`，tokenizer `68bd37c4…`，推理包 manifest `9d9dad19…`）。未改动：checkpoint、tokenizer、INT8 配方（INT8 block32，权重阈值 2048，参数选择相同）、搜索规则。`RELEASE.json` 和 `SHA256SUMS.txt` 未动（SHA256 仍为 `899ca51a…` / `58096d45…`）。没有编写或提交 Vime 端代码。

环境：Apple M3，16 GiB，macOS 27.3（26C5035r），Xcode 27.3（27C5020n），Python 3.11.17（uv venv），PyTorch 2.7.1，coremltools 9.0，NumPy 2.3.5，scikit-learn 1.5.1，sentencepiece 0.2.1，psutil。coremltools 提示 PyTorch 2.7.1 未经测试（2.7.0 经过测试）；没有观察到影响。Core ML 目标：mlprogram，iOS 18，FP32 计算，除非另有说明为 `CPU_ONLY`。接交流电（电池 100%，已充满），`pmset -g therm` 没有热或性能警告，平均负载 2.9–4.1。

所有输出在 `outputs/deployment/zh-v1-kvbatch/` 下；各阶段日志在 `logs/`，探索性辅助脚本在 `tools/`。没有覆盖任何输出。一次 shell 失误（zsh 中 `$P` 变量未拆分）导致第 4 步第一次循环什么都没跑；其日志保留在 `logs/shell-typo/`，循环已重跑。

**没有修改 `src/vimeml/deployment/zh_v1_batch.py`。** 动态槽位数和 one-hot 写入的转换按原样可用，从未需要"修复后重跑 `check`"。

## 摘要

| 步骤 | 结果 |
| --- | --- |
| 1. PyTorch 检查 | 通过：eager 和 traced 各 4 例，最大 5.01e-5（Windows 3.96e-5） |
| 2. FP32 flex | 已转换；对 PyTorch FP32 最大 5.05e-5，507 行，argmax 全部相同 —— 通过 |
| 3. INT8 flex | Vime 对无状态 INT8 门槛**通过**：最大 4.82e-5（上限 3e-4） |
| 4. 固定 B=16 / B=8 | FP32 5.05e-5 / 4.58e-5；INT8 对无状态 4.82e-5 / 3.58e-5 —— 全部通过，包括填充行 |
| 5. FP16 缓存 | 严格门槛**如预期失败**（FP32 最大 7.0e-3，INT8 最大 6.2e-3），argmax 507/507；`compress` 拒绝；INT8 用单独的仅作证据工具构建。CPU 上无速度提升，ANE 无提升 |
| 6. 合并包 | 两个函数都能加载，与独立包逐位一致；权重只存一份（22.96 MB 对 45.59 MB）；Mac 上内存占用节省约 10 MiB，不足一份完整权重 |
| 7. Mac 延迟 | INT8 flex B=16：3.35 ms（S=16），3.76 ms（S=32）；B=8：2.69 / 2.92 ms。固定 16 在 B=16 时不更快，B=1 时要付约 3.3 ms |
| 8. Vime fixture | 16 行，位置 1–9，一次解码调用：包、无状态 INT8 和 PyTorch FP32 的 top-5 相同；对无状态最大 2.48e-5 |
| 9. 搜索检查 | FP32（FP32 预填充）：125/125 排序完全一致，最大 3.8e-5。INT8：与只用无状态 INT8 的搜索完全相同（125/125）；对 FP32 fixture top-1 125/125，文本集合相同 86/125 —— 见 §9 |
| 10. 客户端资源 | 由 **INT8 flex** 构建（`ios-kvbatch/`） |
| 11. iPhone | **未做**（Vime 没有批量测试宿主；本会话未构建） |
| 12. 发布副本 | `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvbatch/` = INT8 flex，字节相同 |

## 接口与两处有意的差异

`vimeml.deployment.zh_v1_batch.KVBatchGraphZh`，一次调用把 B 个独立假设各推进一个 token：

| 名称 | 类型 | 含义 |
| --- | --- | --- |
| `input_ids` | int32 `[B,1]` | 每行一个新 token |
| `position` | int32 `[B]` | 该 token 之前的行缓存长度；`position[b] < S` |
| `key_cache`、`value_cache` | float32（探索变体为 float16）`(9,B,6,S,64)` | 第 b 行 `>= position[b]` 的槽位被忽略；那里的垃圾值被掩码（用 ±30 填充检查过） |
| → `logits` | float32 `[B,12288]` | 只有新 token；不做 softmax |
| → `new_key`、`new_value` | `(9,B,6,64)` | 新 token 的 key/value；调用方把它们写入第 b 行的槽位 `position[b]` |

行之间从不互相注意；填充行（位置 0 的 token 0）不改变真实行（在固定 16 对齐中检查过：15 和 11 个填充行；固定 8 中：7 和 3 个）。

1. **`S` 可变（1..128），而不是固定 128。** 调用方传入 `max(position)+1` 个槽位。B=16 的固定 128 槽缓存每个 28 MiB，每次调用约 56 MiB 输入，而真实的行只是一个短 prompt 加约 20 个 token。Mac 上实测 S 的开销（INT8 flex，B=16）：S=16 时 3.35 ms，S=32 时 3.76，S=64 时 5.45，S=128 时 8.73 —— 固定 128 的约定会让每步时间大致翻倍。
2. **只返回新槽位，不返回完整缓存。** 返回两个完整缓存每次调用要写出与输入一样多的数据，比这一步本身还多。用调用方持有的缓冲区，只拷贝 `(9,B,6,64)` 的槽位；Vime 仍可用自己的缓冲区承载输入（小输出用 `outputBackings`）。

prompt 仍用已发布的 batch-1 chunk 包（或合并包的 `prefill`）：运行一次，把它的缓存行拷进批量缓冲区。不使用 MLState（一个桶里的行，其父节点分布在之前不同的桶中）。

## 1. PyTorch 检查

`check.json`。16 行，每行 3–25 个 token，从不同的步开始（每次调用中位置各不相同）；每一步的 logits 与该行自己的完整前向对比。

| 用例 | 调用数 | 行 | 最大绝对误差（eager = traced） |
| --- | ---: | ---: | ---: |
| B=16，S=max(position)+1 | 30 | 221 | 4.10e-5 |
| B=1 | 3 | 3 | 2.19e-5 |
| B=5，S=128（无垃圾值的完整缓冲区） | 30 | 62 | 3.91e-5 |
| B=16，seed 2026 | 30 | 221 | 5.01e-5 |

## 2–4. 转换、压缩与对齐

INT8 配方：量化 47 个矩阵（gather 与共享输出头共用的 token embedding、位置 embedding、9 × QKV / projection / gate / up / down）；排除 1 个结构常量（零输出头偏置；这个图没有因果掩码常量）。FP32 计算，不量化激活。

| 包 | Batch | 逻辑字节 |
| --- | --- | ---: |
| `fp32-flex` | 1..16（`RangeDim`） | 80,337,051 |
| `fp32-b16` / `fp32-b8` | 固定 | 80,337,661 |
| `int8-flex` | 1..16 | 22,762,761 |
| `int8-b16` / `int8-b8` | 固定 | 22,763,413 |

每个对齐用例的最大绝对误差（容差 atol = rtol = 3e-4；每例容差外 0 行，每行 argmax 相同）：

| 包 | 参照 | B=16 | B=1 | B=5，S=128 | B=16 seed 2026 | 结果 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `fp32-flex` | PyTorch FP32 | 5.05e-5 | 2.57e-5 | 4.20e-5 | 4.72e-5 | 通过 |
| `fp32-b16` | PyTorch FP32 | 5.05e-5 | 2.96e-5（+15 填充） | 4.58e-5（+11 填充） | 4.72e-5 | 通过 |
| `fp32-b8` | PyTorch FP32 | — | 2.96e-5（+7 填充） | 4.58e-5（+3 填充） | — | 通过 |
| `int8-flex` | 无状态 INT8 | 4.24e-5 | 7.2e-6 | 3.43e-5 | 4.82e-5 | **通过** |
| `int8-b16` | 无状态 INT8 | 4.24e-5 | 1.72e-5（+15 填充） | 3.58e-5（+11 填充） | 4.82e-5 | **通过** |
| `int8-b8` | 无状态 INT8 | — | 1.72e-5（+7 填充） | 3.58e-5（+3 填充） | — | **通过** |

按任务说明，B=8 的对齐跳过 16 行的用例。chunk-KV INT8 包与无状态 INT8 相差 4.2e-5 以内；批量包相差 4.8e-5 以内。

## 5. FP16 缓存（探索性，仅作证据）

`fp32-flex-fp16cache` 用 `--cache-dtype float16` 转换。它对 PyTorch FP32 的严格对齐**失败**（`fp32-flex-fp16cache-align.json`，保留）。随后 `compress` 拒绝了它（"First pass alignment for this exact FP32 batch package"；`logs/05-compress-int8-flex-fp16cache.log`）。为了仍能测量 INT8 FP16 缓存包，`tools/compress_evidence_only.py` 套用完全相同的配方，但只记录失败的门槛而不强制执行；结果 `int8-flex-fp16cache` 在 manifest 中带有 `"evidence_only": true`。它没有用于资源或发布（资源脚本也要求通过无状态门槛）。它对无状态 INT8 的对齐同样失败（`int8-flex-fp16cache-align.json`）。

误差分布（`tools/error_distribution.py`；与对齐相同的行，全部 logits）：

| 包 / 参照 | 用例 | 最大 | 均值 | RMSE | p50 | p90 | p99 | p99.9 | Logits > 3e-4 | 容差外行 | Argmax | Top-5 集合 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 FP16 缓存 / PyTorch | B=16 | 4.38e-3 | 4.5e-4 | 6.5e-4 | 3.2e-4 | 1.04e-3 | 2.20e-3 | 3.19e-3 | 52% | 186/221 | 221/221 | 220/221 |
| | B=1 | 6.7e-4 | 1.2e-4 | 1.7e-4 | 8.9e-5 | 3.0e-4 | 4.4e-4 | 5.5e-4 | 10% | 2/3 | 3/3 | 3/3 |
| | B=5，S=128 | 5.49e-3 | 5.5e-4 | 8.5e-4 | 3.7e-4 | 1.23e-3 | 3.44e-3 | 4.49e-3 | 56% | 53/62 | 62/62 | 62/62 |
| | B=16 seed 2026 | 7.03e-3 | 5.3e-4 | 7.3e-4 | 4.0e-4 | 1.16e-3 | 2.29e-3 | 3.75e-3 | 60% | 199/221 | 221/221 | 221/221 |
| INT8 FP16 缓存 / 无状态 INT8 | B=16 | 6.17e-3 | 5.3e-4 | 7.7e-4 | 3.5e-4 | 1.27e-3 | 2.59e-3 | 3.77e-3 | 55% | 195/221 | 221/221 | 221/221 |
| | B=1 | 9.2e-4 | 1.7e-4 | 2.4e-4 | 1.5e-4 | 4.1e-4 | 5.7e-4 | 6.9e-4 | 25% | 2/3 | 3/3 | 3/3 |
| | B=5，S=128 | 3.05e-3 | 4.2e-4 | 5.7e-4 | 3.2e-4 | 9.8e-4 | 1.60e-3 | 2.02e-3 | 52% | 52/62 | 62/62 | 62/62 |
| | B=16 seed 2026 | 5.71e-3 | 5.3e-4 | 7.6e-4 | 3.9e-4 | 1.19e-3 | 2.63e-3 | 3.99e-3 | 59% | 197/221 | 221/221 | 221/221 |

相对 INT8 自身的误差（约 0.4）而言这个误差很小，但处处都超过 3e-4。速度（INT8 FP16 缓存，p50 ms，完整网格单次运行，无状态包以相同计算单元加载）：

| B，S | INT8 flex FP32 缓存，CPU_ONLY | FP16 缓存 CPU_ONLY | FP16 缓存 CPU_AND_NE | FP16 缓存 ALL |
| --- | ---: | ---: | ---: | ---: |
| 1, 16 | 1.90 | 1.67 | 1.75 | 2.56 |
| 4, 32 | 3.08 | 3.12 | 3.05 | 5.43 |
| 8, 32 | 3.51 | 3.97 | 3.78 | 4.21 |
| 16, 16 | 3.88 | 4.14 | 4.03 | 3.76 |
| 16, 32 | 4.22 | 5.60 | 5.24 | 5.01 |
| 16, 64 | 5.45 | 8.35 | 8.12 | 7.71 |
| 16, 128 | 8.73 | 14.37 | 13.69 | 12.86 |

FP16 缓存在 CPU 上没有帮助：输入字节减半，但 FP32 计算的图会把它们转回去，从 B=16、S=32 起反而更慢。`CPU_AND_NE` 与 `CPU_ONLY` 在噪声范围内相同（FP32 计算的程序不在 ANE 上运行）。`ALL` 噪声大且不更好；在 `ALL` 下加载还让进程 RSS 比 `CPU_ONLY` 多约 450 MiB。这与 Vime 在 chunk 包上看到的一致。建议：保留 FP32 缓存和 `CPU_ONLY`。

## 6. 合并包

`merge int8-flex int8-merged`，chunk = 已发布的 `ios18-int8-block32-kvchunk`。`prefill` 和 `decode` 共享一个 `weight.bin`（22,694,592 字节）。

| | 字节 |
| --- | ---: |
| `int8-merged` 包 | 22,962,601 |
| 已发布 chunk 包 + `int8-flex` | 22,830,199 + 22,762,761 = 45,592,960 |

`tools/merged_memory.py` 在每次全新进程中（每种配置 3 次）加载：无状态 INT8 包（运行一次），然后加载两个独立包或合并包的两个函数（`function_name="prefill"` / `"decode"`），运行一次预填充和一次 B=16 解码。两个函数都能加载；输出与独立包完全相同（预填充 logits/keys 和解码 logits/keys 最大绝对差 0）。

| Mac，Python，CPU_ONLY（3 次中位数） | 独立 | 合并 |
| --- | ---: | ---: |
| 无状态 INT8 加载 + 运行，phys_footprint 增量 | 15.7 MiB | 15.2 MiB |
| + 加载两者 + 预填充，phys_footprint 增量 | 63.9 MiB | 54.4 MiB |
| + B=16 解码 | 2.4 MiB | 2.4 MiB |
| 相对无状态的总增量，phys_footprint | **66.2 MiB** | **56.8 MiB** |
| 相对无状态的总增量，RSS | 222.2 MiB | 230.7 MiB |

磁盘字节减半，但在 Mac 上内存占用只省约 9.5 MiB，不到一份权重（21.6 MiB）；加载开销大部分不是权重文件（每个函数的编译程序状态和缓冲区，加上 Python/coremltools 的编译路径）。包含映射权重文件的 RSS 合并后略高。所以"一份而不是两份"对包成立，对 Mac 上的内存只部分成立；需要在真机上确认（键盘扩展的 `phys_footprint`），本次没有做（§11）。

## 7. Mac 延迟

`measure`，20 次预热后 200 次重复，无状态 INT8 同时加载并运行，`CPU_ONLY`。时间为 Python `predict()` 墙钟时间，含输入转换。第一次完整网格（`latency/*-latency.json`）显示 `int8-merged` 比 `int8-flex` 快 10–20%，但它的 `decode` 函数是同一个图，可见单次运行的噪声就有这么大。因此在真实桶会用到的 S 值上，把这些包交替重测了三次（`latency/*-rerun{1,2,3}-latency.json`）；表中为三次 p50 / p95 的中位数。

| B，S | INT8 flex | INT8 固定 16 | INT8 固定 8 | 合并 `decode` |
| --- | ---: | ---: | ---: | ---: |
| 1, 16 | 1.64 / 1.98 | 3.31 / 3.73 | 2.65 / 3.04 | 1.67 / 1.95 |
| 1, 32 | 1.80 / 2.12 | 3.86 / 4.18 | 2.94 / 3.28 | 1.77 / 2.13 |
| 4, 16 | 2.30 / 2.94 | 3.27 / 3.75 | 2.43 / 2.94 | 2.48 / 3.08 |
| 4, 32 | 2.47 / 3.11 | 3.84 / 4.17 | 2.83 / 3.35 | 2.66 / 3.21 |
| 8, 16 | 2.69 / 3.28 | 3.19 / 4.24 | 2.65 / 3.04 | 2.87 / 3.30 |
| 8, 32 | 2.92 / 3.54 | 3.63 / 4.68 | 2.89 / 3.33 | 2.79 / 3.44 |
| 16, 16 | 3.35 / 3.72 | 3.34 / 3.77 | — | 3.25 / 3.73 |
| 16, 32 | 3.76 / 4.25 | 3.67 / 4.18 | — | 3.89 / 4.24 |

完整网格单次运行（p50 ms），更大的 S：

| B，S | INT8 flex | 固定 16 | 固定 8 | 合并 |
| --- | ---: | ---: | ---: | ---: |
| 16, 64 | 5.45 | 4.24 | — | 4.63 |
| 16, 128 | 8.73 | 6.66 | — | 7.05 |
| 8, 64 | 3.94 | 5.44 | 3.13 | 3.41 |
| 8, 128 | 5.15 | 8.59 | 4.22 | 4.77 |

- 固定包的填充开销：固定 16 调用中一个 B=1 行要付完整 16 行一步的时间（S=16/32 时 3.3–3.9 ms，flex 为 1.6–1.8 ms）；固定 8 中 B=1 为 2.7–2.9 ms。
- 在 B=16 和真实的 S 下，flex 与固定 16 在噪声范围内相同。固定 16 在 S≥64 时可能更快（单次运行），但真实桶达不到。
- Mac 目标：B=16 ≤ 8 ms、B=8 ≤ 5 ms 都有很大余量。这些是 Mac 数字；Vime 中 iPhone 13 Pro 每次 batch-1 chunk predict 为 3.8 ms，这里每个 B=1 步约 1.7 ms，所以真机时间预计大约是 Mac 的两倍，仍在目标内，但未经测量。
- 脚本说明：`measure` 也会运行大于固定包批次的 `rows`（如固定 8 上的 "rows 16"），实际按 8 行调用计时却标错了标签。上表排除了这些行；它们仍在 JSON 里。

## 8. Vime fixture

`fixture int8-flex fixture` → `fixture/batch16.jsonl`、`batch16_logits.npz`、`meta.json`。16 个手写的句子前缀（不是基准条目），位置 1–9（8 个不同位置，`slots_needed` 10）；每个用已发布的 chunk 包预填充，然后在一次解码调用中全部推进一步。

- 包对无状态 INT8（完整前向）：最大 2.48e-5。
- PyTorch 批量图对 PyTorch 完整前向：最大 2.48e-5。
- 每行 top-5 id：16 行全部是 包 = 无状态 INT8 = PyTorch FP32，顺序相同。

## 9. 搜索检查

在 125 例 `zh-generation-v1-lm-r2` fixture 上用不变的逐按键位置 beam 搜索（`zh_search.search`，beam 16，top 10，lm 模式）运行 `search`。每次生成平均 14.48 次解码调用、231.68 行（每次调用 16.0 行）。不批量的运行时每次生成要做 231.68 次 batch-1 predict；批量后变成 14.48 次调用。

| 运行 | 预填充 | 排序完全一致 | 文本集合相同 | 最大分数差（阶段输出，按名次） | Mac 每次生成 p50 / p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| `search-r2`（INT8 flex） | 已发布 INT8 chunk | 46 | 86 | 3.48 | 87.5 / 214.6 ms |
| `search-r2-fp32-flex-int8chunk`（按任务说明的命令运行） | 已发布 INT8 chunk | 90 | 110 | 2.18 | 85.8 / 207.1 ms |
| `search-r2-fp32-flex-fp32chunk` | FP32 chunk `tiny-zh-v1-kvchunk-fp32-v1` | **125** | **125** | **3.8e-5** | 85.4 / 191.0 ms |

- `search` 阶段的 `--chunk` 默认是 INT8 chunk 包，所以任务说明里的"FP32 flex"运行仍用 INT8 预填充 prompt，不可能完全一致。完全 FP32 的运行用 `--chunk` 指向 2026-11-11 zh-v1 工作中的 FP32 chunk 包（`~/Documents/Sources/VimeML-zh-v1-coreml/artifacts/deployment/tiny-zh-v1-kvchunk-fp32-v1`，推理包 manifest 同为 `9d9dad19…`，原地读取，未复制）。它完全复现了 fixture：排序完全一致，分数相差 3.8e-5 以内（上限 1e-3）。因此批量图、Core ML 转换和搜索都是正确的。
- 阶段输出的"最大分数差"按名次比较分数。一旦某个文本掉出列表，名次就会错位，所以它夸大了分数误差。

**INT8 归因**（`tools/search_int8_attribution.py` → `search-r2-int8-attribution/`）。用一个从不使用批量包或 chunk 包的后端跑同样的搜索：每一行都是已发布无状态 INT8 包的一次完整前向。

| 对比 | 排序完全一致 | 文本集合相同 | Top-1 相同 | 共同文本最大分数差（每例中位数） | ≤ 0.5 的用例 | 掉出的文本（参照名次） |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 批量 INT8 对 无状态 INT8 搜索 | 125 | 125 | 125 | 2.2e-5 | 125 | 无 |
| 无状态 INT8 对 FP32 fixture | 46 | 86 | 125 | 0.565（0.128） | 124 | 48：第 10 名 ×24，9 ×7，8 ×5，7 ×2，6 ×4，4 ×3，3 ×2，2 ×1 |
| 批量 INT8 对 FP32 fixture | 46 | 86 | 125 | 0.565（0.128） | 124 | 同样的 48 个 |

批量 INT8 包的搜索结果与已发布的无状态 INT8 包完全相同。与 FP32 fixture 的差异来自 INT8 本身。Top-1 从不改变，共同文本的分数相差约 0.5 以内（一例 0.565）。125 例中有 39 例文本集合发生变化，大多在第 9–10 名。第 2–4 名的 7 次掉出是因为 beam 更早地保留了不同的假设，而不是互换。`meta.json` 对真机 INT8 期望"文本集合相同"。用这个 INT8 配方，任何 INT8 运行时（批量与否）都只有 86/125 例满足。Vime 应拿 INT8 对比 INT8（`search-r2-int8-attribution` 或 fixture 的 INT8 行），或接受尾部的集合差异。

## 10. 客户端资源

选定：**INT8 flex**。任务说明的规则是选 flex，除非它明显慢于固定 16。在 B=16 和真实的 S 下，flex 与固定 16 在噪声范围内相同。在更小的桶上 flex 最多快 2 倍，而固定 16 总是要付完整 16 行一步的时间。一个包覆盖 1 到 16 的所有 B。

`package_zh_v1_kvbatch_ios.py --batch int8-flex --alignment int8-flex-align.json --search search-r2 --fixture fixture --output ios-kvbatch`：

- `ios-kvbatch/Resources/TinyChineseV1KVBatchINT8.mlmodelc`（用 Xcode 27.3 `coremlcompiler` 编译，22 MB）
- `ios-kvbatch/Resources/VimeLMManifestZhV1KVBatch.json`：`kv_batch` 段，含资源、编译文件 SHA256、接口、缓存形状/槽位/数据类型、`batch_range` [1,16]、对齐容差/参照/检查、搜索检查、策略
- `ios-kvbatch/KVBatchFixture-*`（第 8 步的 fixture）

现有 zh-v1 资源（`TinyChineseV1INT8`、`TinyChineseV1KVChunkINT8`、`VimeLMManifestZhV1.json`）未动。manifest 的 `search_check` 带着阶段汇总，包括按名次对齐的 3.48。按文本对齐的数字见 §9。

## 11. iPhone

**未做。** 两台测试手机（iPhone 13 Pro、iPhone 16 Pro Max）都已配对可用，但 Vime 没有批量 KV 的测试宿主。逐次 predict 和内存占用的测试工具意味着要写新的客户端代码，并在本会话中签名安装，任务说明把这留给了 Vime。真机上仍待测：两台手机上 B = 1/4/8/16、S = 16/32；在已加载无状态包的情况下，键盘扩展在今天 43 MB 基础上增加的内存；以及合并包在真机上省下的内存是否比 Mac 上的约 9.5 MiB 更多。

## 12. 发布副本

`artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvbatch/` = `outputs/deployment/zh-v1-kvbatch/int8-flex/`（`diff -r` 相同）。`manifest.json` SHA256 `472d148a…`，`weight.bin` `7fa1b5c8…`，`model.mlmodel` `0e8c9e70…`。`RELEASE.json` / `SHA256SUMS.txt` 留给 Windows 端。

**Windows 端后续（2026-11-15）。** 结果 ZIP 已与其 SHA256SUMS 核对（159 个文件），发布副本已与 `int8-flex` 及其包 manifest 核对。发布清单现在列出 `ios18-int8-block32-kvbatch/` 的四个文件和 `coreml_manifest_sha256.kvbatch_int8` = `472d148a…`；README 描述了批量包。`verify_release.py --hashes` 在 370 个文件上通过。Hugging Face 发布在 revision `ce35c931e0102248e484fd93c36100f430067ead`（父 revision `8cc5ef3a…`）原地更新；重新下载该 revision 后通过 `verify_release.py --hashes`。本报告指出的两个脚本问题已在 `coreml_zh_v1_batch.py` 中修复：`measure` 不再运行超过固定包批次的行数；`search` 还会报告 Top-1 一致率、两个列表中都存在的文本的分数差，以及掉出的文本。文档注明了要做完全 FP32 的运行，`--chunk` 需要指向 FP32 chunk 包。

## 内存与占用

- 单独的无状态 INT8（Mac，Python）：phys_footprint +15.6 MiB。
- 再加批量包：一次 B=16 解码在加载之外增加 2.4 MiB。加上已发布的 chunk 包，chunk + 批量在 Mac 上增加 66 MiB（独立）或 57 MiB（合并）phys_footprint。这些是 Python 宿主进程的数字，被 coremltools 的加载路径放大了。它们不是键盘扩展的内存占用，后者仍待测量（§11）。
- 每次调用，B=16 时调用方缓冲区为 2 × 9·16·6·S·64·4 字节：S=16 时 0.84 MiB，S=32 时 1.69 MiB（输入），加上 48 KiB logits 和 2 × 216 KiB 新槽位（输出）。

## 只返回允许 token 的 logits 与 MLState

按任务说明未构建。log-softmax 需要全部 12,288 个 logits，所以只返回允许的 token 并不能跳过输出头矩阵乘。B=16 时 `[B,12288]` 输出为 768 KiB。它的拷贝和客户端 softmax 是否重要，应根据真机数字判断；Mac 上看不出它占主导。MLState 继续不用。

## 文件

| 路径 | 内容 |
| --- | --- |
| `check.json` | 第 1 步 |
| `fp32-{flex,b16,b8}`、`*-align.json` | FP32 包及其对 PyTorch 的对齐 |
| `int8-{flex,b16,b8}`、`*-align.json` | INT8 包及其对无状态 INT8 的门槛 |
| `fp32-flex-fp16cache`、`int8-flex-fp16cache`（仅作证据）、`*-align.json`、`*-errors.json` | 第 5 步 |
| `int8-merged`、`merged-memory/` | 第 6 步 |
| `latency/` | 第 7 步（完整网格和三次交替重测） |
| `fixture/` | 第 8 步 |
| `search-r2*`、`search-r2-int8-attribution/` | 第 9 步 |
| `ios-kvbatch/` | 第 10 步 |
| `tools/` | `compress_evidence_only.py`、`error_distribution.py`、`merged_memory.py`、`search_int8_attribution.py` |
| `logs/` | 各阶段控制台输出；`shell-typo/` 是第 4 步第一次空跑的记录 |
