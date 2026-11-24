# vimeml-tiny-zh-v1 Core ML 整句打分（sequence 输出头）：Mac 任务说明

这是 zh-v1 批量 KV 包的第三个 Mac 任务。它接在 [mac-zh-v1-kvbatch-v2.md](mac-zh-v1-kvbatch-v2.md) 之后；那个任务的报告是 [mac-20261115/zh-v1-kvbatch-v2.md](../reports/mac-20261115/zh-v1-kvbatch-v2.md)。

## 原因

Vime 现在对整句打分：字面转换结果加上几十个近似候选，每个候选与字面结果只在一处不同。它用 token 树共享前缀，并把树的每一层送入 `unified-scored` 包。这个包只返回**最后一个**新 token 的下一 token 分布，所以共享部分之后有 n 个 token 的候选仍要 n 次调用。长句在 Mac 上 p95 达到 310 ms，主要来自顺序调用的次数。

新的 **sequence 输出头**在**每个**新位置返回给定下一 token 的对数概率。这样一个已知的续写一次调用就能打分。共享前缀之后，一批最多 16 个候选只需一次调用，无论它们多长。

## 改动：`zh_v1_batch2.KVBatchGraphZhV2(model, "sequence")`

- 图、输入和缓存规则与 v2 相同。
- **新输入** `targets` int32 `[B,T]`：跟在 `input_ids[b,t]` 后面的 token。
- **输出** `target_logprobs` float32 `[B,T]`：log P(`targets[b,t]` | 第 b 行直到 `input_ids[b,t]`)，在全部 12,288 个 logits 上归一化。
- `new_key`/`new_value` 不变：`(9,B,6,T,64)`。
- **填充。** 长度不同的行用任意 token 和任意目标右填充。填充位置在真实位置之后，因果性使它们不影响真实分数。调用方忽略它们的输出。
- **打分规则**（`zh_v1_batch2.score_continuations`，参照算法）。设 `prefix` 为所有候选共享的 id（BOS + 上下文，在候选分歧处截断）：
  1. 对 `prefix[:-1]` 做一次预填充调用。
  2. 每 16 个候选一次调用：`input_ids = [prefix[-1]] + c[:-1]`，`targets = c`，`position = len(prefix) − 1`，每行读取同一个缓存。
  3. 候选的分数是其 `target_logprobs` 之和。

  这就是 `benchmarks.zh13_lm_scores.score_item` 的生产规则：BOS + 上下文 + 候选，目标从共享前缀开始。
- Sequence 输出头只构建可变 T 版本，不用枚举形状。仍要求 `position[b] + T <= 128`。
- **开销。** 输出头矩阵乘对每个新位置都运行，`[B,T,12288]`。B=16、T=32 时图内有 25 MB 的 logits。要测量；见下文。

## Windows 检查

PyTorch FP32，真实 zh-v1 权重：

| 检查 | 结果 |
| --- | --- |
| 每行 prompt 之后的逐位置对数概率，然后从不同位置做填充的多行调用（B = 4、1、16） | 最大 1.1e-5，3e-4 外 0 个 |
| Traced 图，同样的检查 | 通过 |
| `score_continuations`，37 个 1–12 token 的候选 | 求和相差 2.7e-5 以内；4 次调用 |
| 生产规则（`score_item`）下的 fixture 用例 | 最大 1.9e-5；18 个候选 3 次调用 |
| 现有的解码、预填充和 scored 输出头检查 | 不变（最大 3.96e-5） |

`coremltools` 没有在 Windows 上运行。

## 环境准备

使用之前任务的检出目录（`~/Documents/Sources/VimeML-zh-kvbatch`）：

```bash
cd ~/Documents/Sources/VimeML-zh-kvbatch
git pull --ff-only origin main        # needs the commit that adds the sequence head
source .venv/bin/activate
O=outputs/deployment/zh-v1-kvsequence
S=artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32
U=outputs/deployment/zh-v1-kvbatch-v2/int8-unified-scored
```

所有新输出放在 `$O` 下。绝不覆盖输出。

## 阶段

1. **检查。** `python scripts/zh/deployment/coreml_zh_v1_batch2.py check $O/torch-check.json`。必须打印 `passed`，并列出 `sequence-Tflex` 为已 trace。
2. **转换。** `python scripts/zh/deployment/coreml_zh_v1_batch2.py convert $O/fp32-seq --tokens flex --head sequence`。批大小和槽位保持可变（B 1..16，S 1..128），与 `unified` 相同。
3. **FP32 对齐。** `python scripts/zh/deployment/coreml_zh_v1_batch2.py align $O/fp32-seq $O/fp32-seq-align.json`，对 PyTorch FP32。
   - 它运行逐位置检查（B = 4、1、16）和 `score_continuations` 求和检查。
   - 容差为每个位置 3e-4、每个求和 1e-3。
   - 如果转换或对齐失败，在 `zh_v1_batch2.py` 中修复，重跑 `check`，并记录改动。
4. **压缩。** `python scripts/zh/deployment/coreml_zh_v1_batch2.py compress $O/fp32-seq $O/fp32-seq-align.json $O/int8-seq`。
5. **INT8 对齐。** `python scripts/zh/deployment/coreml_zh_v1_batch2.py align $O/int8-seq $O/int8-seq-align.json --stateless $S`。这是 Vime 的门槛。
6. **延迟。** `python scripts/zh/deployment/coreml_zh_v1_batch2.py measure $O/int8-seq $O/latency-seq.json`。
   - 默认网格为 B 1/4/8/16 × T 4/8/16/32 × S 16/32/64/128（跳过 S + T > 128），加上 8/16/32 token 的 prompt 预填充，同时加载无状态包。
   - 作为对比，也运行 `measure $U $O/latency-unified-scored.json`；v2 任务中可能已经有了。
   - 要比较的数字：B=16、T=16 的一次 sequence 输出头调用，对 B=16 的 16 次 `unified-scored` 解码调用。
7. **内存。** `python scripts/zh/deployment/coreml_zh_v1_batch2.py memory $O/memory.json seq=$O/int8-seq unified=$U`。
   - Sequence 运行在最后增加一次 B=16、T=32、S=64 的打分调用（`after_score_B16_T32_S64`）。
   - 观察 `[B,T,12288]` logits 是否让内存占用超过可变 T 包的约 40 MiB。
8. **Fixture。** `python scripts/zh/deployment/coreml_zh_v1_batch2.py fixture $O/int8-seq $O/fixture`。
   - 它写出 `sequence.jsonl`：8 个用例，每个 3–18 个候选。每个用例有共享前缀 id、每个候选的续写 id，以及来自包、无状态 INT8 包和 PyTorch FP32 的期望求和。
   - 预期包对无状态每个求和相差约 1e-3 以内。
9. **客户端资源。**

   ```bash
   python scripts/zh/deployment/package_zh_v1_kvbatch_ios.py --batch $O/int8-seq --alignment $O/int8-seq-align.json \
       --fixture $O/fixture --output $O/ios-kvsequence
   ```

   它写出 `Resources/TinyChineseV1KVSequenceINT8.mlmodelc` 和 `VimeLMManifestZhV1KVSequence.json`（段 `kv_sequence`），以及 `KVSequenceFixture-*`。v1 和 v2 资源保持不动。
10. **发布副本。** 把 `$O/int8-seq` 拷贝到 `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvsequence/`。不要编辑 `RELEASE.json` 或 `SHA256SUMS.txt`。
11. **iPhone。** 不在本会话中。Vime 用自己的测试宿主测量。

## 决策规则

- 满足以下条件时**发布** sequence 包：
  - 两次对齐都通过；
  - fixture 一致；
  - 其内存占用与 `unified-scored` 相差在几 MiB 以内。
- **如果大 T 时内存占用增长很多**，报告随 T 变化的曲线。Vime 可以限制每次调用的 T（例如 16），把更长的续写拆成两次调用；缓存规则允许这样做。
- **不要替换 `unified-scored`。** 搜索继续使用它。Sequence 包是用于整句打分的额外资源。

## 报告与回传

按 v2 报告的格式写 `docs/zh/reports/mac-<date>/zh-v1-kvsequence.md`。内容应包括：

- 环境；
- 每个阶段，包括作为证据保留的失败；
- 对齐表；
- 延迟表，含第 6 阶段的对比；
- 内存表；
- fixture 摘要；
- 决定。

带回一个附 SHA256SUMS 的 ZIP，包含：

- `$O`，空间不够时可不带 `fp32-seq`；
- 报告；
- 发布副本；
- `ios-kvsequence/`；
- 对 `src/vimeml/deployment/zh_v1_batch2.py` 的任何改动，或说明没有改动。

不要在 Mac 上提交。
