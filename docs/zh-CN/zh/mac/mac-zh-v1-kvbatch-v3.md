# vimeml-tiny-zh-v1 批量 KV v3（共享前缀、精简打分、单一包）：Mac 任务说明

这是 zh-v1 批量 KV 包的第四个 Mac 任务。它接在 [mac-zh-v1-kvsequence.md](mac-zh-v1-kvsequence.md) 之后；那个任务的报告是 [mac-20261116/zh-v1-kvsequence.md](../reports/mac-20261116/zh-v1-kvsequence.md)。

## 原因

Vime 在 16 Pro Max 上的运行峰值为 65.7 MB，而键盘扩展有硬性内存上限，在 6 GB 内存的 iPhone 上更低。目前 Vime 最多加载三个包：无状态包、`kvbatch-v2`（`unified-scored`）和 `kvsequence`。每个包都持有一份自己的权重。在权重之外，有三件事占内存：

1. **Prompt 被拷贝到每一行。** 一个 16 行的 beam 桶在全部 16 个缓存中都带着同样的 BOS + 上下文。每个 token 27 KB，48 token 的 prompt 每行 1.3 MB，每次 B=16 调用 21 MB，`key_cache` 和 `value_cache` 各一份。
2. **Sequence 输出头会生成 `[B,T,12288]` 的 logits。** B=16、T=16 时图内有 12.6 MB，而且它还返回打分从不使用的 `new_key`/`new_value`。在 Mac 上内存随 T 增长：T=16 时 +9 MiB，T=64 时 +63 MiB。
3. **独立的包持有独立的权重副本。** v1 kvbatch 任务表明，合并的多函数包只存一份权重（磁盘上 22.96 MB 对 45.59 MB），在 Mac 上省了约 9.5 MiB。

激活量化和低于 INT8 的权重量化不在 20M 的范围内。

## 改动：`vimeml.deployment.zh_v1_batch3`

**一个多函数包**，三个函数共享同一份权重：

| 函数 | 输入 | 输出 |
| --- | --- | --- |
| `logits` | `input_ids` [B,T]、`position` [B]、`key_cache`/`value_cache` (9,B,6,S,64)、`prefix_length` [1]、`prefix_key`/`prefix_value` (9,1,6,P,64) | 最后一个新 token 的 `logits` [B,12288]、`new_key`/`new_value` (9,B,6,T,64) |
| `scored` | 同 `logits`，加上 `allowed` [K] | `logsumexp` [B]、`allowed_logits` [B,K]、`new_key`/`new_value` |
| `score` | `input_ids` [B,T]、`targets` [B,T]、`prefix_length`、`prefix_key`/`prefix_value` | 只有 `target_logprobs` [B,T] |

**共享前缀。**
- Prompt（BOS + 上下文）作为 `prefix_key`/`prefix_value` 只传一次，批大小为 1。
- 每一行通过广播读取它。没有任何行持有副本。
- 每行自己的缓存只保存该行在 prompt 之后新增的 token。
- `position[b]` 计数这些 token。绝对位置是 `prefix_length + position[b] + t`。
- 注意力对三部分用一个 softmax：前缀、该行的缓存和新 token。
- `>= prefix_length` 的前缀槽位和 `>= position[b]` 的行槽位被掩码。与之前一样，它们必须是有限值。

**精简打分。**
- `score` 没有行缓存，也没有 KV 输出。每行只读取共享前缀。
- 它从不构建 `[B,T,12288]` 的对数概率。它对 logits 计算 logsumexp，并从共享输出头的对应行取出目标 logit。
- 使用 `--vocab-chunks 4` 时，输出头分四个 3072 宽的块运行，维护滑动的最大值和求和。这把最大的中间结果限制在 `[B,T,3072]`。
  - 分块切片可能在包内复制一份输出头权重。测量大小和峰值；见下文。
- 整句打分规则与 [mac-zh-v1-kvsequence.md](mac-zh-v1-kvsequence.md) 相同，以 prompt 调用的 KV 作为前缀（`zh_v1_batch3.score_continuations`）。

**搜索。**
- `zh_v1_batch3.BatchBackendV3` 接入 `zh_search`。Prompt 调用的 `new_key`/`new_value` 成为前缀；搜索的其余部分不变。
- 单元测试中排序与 v2 后端相同。

形状是可变的：B 1..16，T、S 和 P 1..128，且 `P + S + T <= 128`。包的目标为 iOS 18，FP32 计算，`CPU_ONLY`。

在 iOS 18 上加载：`MLModelConfiguration.functionName = "logits" | "scored" | "score"`，从同一个编译后的包为每个函数创建一个 `MLModel`。权重应在它们之间共享；测量是否如此。

## Windows 检查

PyTorch FP32，真实 zh-v1 权重（`check`）：

| 检查 | 结果 |
| --- | --- |
| 共享前缀之后的 `logits` 和 `scored`：B = 16、1、5（T = 3）、带 64 token 前缀的 16；每个被掩码槽位填 ±30 垃圾值 | 通过，eager 和 traced 最大 3.24e-5 |
| `score`，74 个续写，前缀 9 个和 1 个 token | 通过 |
| `score` 用 4 个词表块（traced） | 每个求和最大 2.14e-5 |
| 单元测试（`tests/test_deployment_zh_v1_batch3.py`）：trace 后形状可变、输入不被修改、搜索与 v2 后端相同 | 5/5 |

`coremltools` 没有在 Windows 上运行。

## 环境准备

```bash
cd ~/Documents/Sources/VimeML-zh-kvbatch
git pull --ff-only origin main        # needs the commit that adds zh_v1_batch3
source .venv/bin/activate
O=outputs/deployment/zh-v1-kvbatch-v3
S=artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32
C="python scripts/zh/deployment/coreml_zh_v1_batch3.py"
```

所有新输出放在 `$O` 下。绝不覆盖输出。

## 阶段

1. **检查。** `$C check $O/torch-check.json`。必须打印 `passed`。
2. **转换。** 生成四个 FP32 包：

   ```bash
   $C convert $O/fp32-logits --function logits
   $C convert $O/fp32-scored --function scored
   $C convert $O/fp32-score --function score
   $C convert $O/fp32-score-c4 --function score --vocab-chunks 4
   ```

   如果转换失败，在 `zh_v1_batch3.py` 中修复，重跑 `check` 和单元测试（`python -m unittest tests.test_deployment_zh_v1_batch3`），并记录改动。
3. **合并 FP32。** 生成两个变体：

   ```bash
   $C merge $O/fp32-merged logits=$O/fp32-logits scored=$O/fp32-scored score=$O/fp32-score
   $C merge $O/fp32-merged-c4 logits=$O/fp32-logits scored=$O/fp32-scored score=$O/fp32-score-c4
   ```

   记录打印出的包字节数，与独立包对比。
4. **FP32 对齐**，对 PyTorch FP32。检查每个函数，被掩码槽位填垃圾值。容差为每个值 3e-4、每个求和 1e-3。

   ```bash
   $C align $O/fp32-merged $O/fp32-merged-align.json
   $C align $O/fp32-merged-c4 $O/fp32-merged-c4-align.json
   ```

5. **压缩。** 按函数分别运行。门槛是包含该包本身的合并 FP32 对齐。

   ```bash
   for f in logits scored score; do $C compress $O/fp32-$f $O/fp32-merged-align.json $O/int8-$f; done
   $C compress $O/fp32-score-c4 $O/fp32-merged-c4-align.json $O/int8-score-c4
   ```

6. **合并 INT8 并对无状态包对齐。** 这是 Vime 的门槛。

   ```bash
   $C merge $O/int8-merged logits=$O/int8-logits scored=$O/int8-scored score=$O/int8-score
   $C merge $O/int8-merged-c4 logits=$O/int8-logits scored=$O/int8-scored score=$O/int8-score-c4
   $C align $O/int8-merged $O/int8-merged-align.json --stateless $S
   $C align $O/int8-merged-c4 $O/int8-merged-c4-align.json --stateless $S
   ```

   - 把 `package_bytes` 与已发布 INT8 包之和（`ios18-int8-block32-kvbatch-v2` + `-kvsequence`）对比。
   - 如果合并包不是大约一份权重，指出哪些常量被重复了。`-c4` 可能多带一份输出头。
7. **延迟。** `$C measure $O/int8-merged $O/latency.json`。如果保留 `-c4`，也对它重复一次。
   - 它在同样的负载上把 v3 与已发布的 v2 包交替运行。网格为 prompt 16/48 × 行自身 token 1/8/16 × 打分 T 8/16，B = 16。
   - v2 把 prompt 拷贝进每一行（S = prompt + 行自身）；v3 只传一次。
   - 预期解码调用不会更慢。打分调用应比 `kvsequence` 快，因为它没有 KV 输出，也不做完整的 log-softmax。
8. **内存。** 这是主要问题。

   ```bash
   $C memory $O/memory.json v3=$O/int8-merged v3c4=$O/int8-merged-c4 --without-stateless
   ```

   - 每种配置在全新进程中运行，各三次，报告中位数。
   - 每轮负载为：48 token 的 prompt；B = 16 的 16 个解码步，行自身缓存从 1 增长到 16；一次 `logits` 调用（仅 v3）；一次 B = 16、T = 16 的打分调用。共 20 轮。
   - 配置：
     - `v2-shipped`：无状态 + `kvbatch-v2` + `kvsequence`，即 Vime 目前的运行方式；
     - `v3` 和 `v3c4`，带无状态包；
     - `v3-alone` 和 `v3c4-alone`，不带无状态包。此时由 `logits` 函数代替无状态包。
   - 报告 `load_mib`、`peak_mib`、`after_rounds_mib` 以及第一轮之后的增长。
   - 还要跑打分的 T 曲线：`--score-tokens 32` 和 `--score-tokens 64`，作为额外输出（`memory-T32.json`、`memory-T64.json`）。这能看出 `-c4` 是否让曲线变平。
9. **搜索。** `$C search $O/int8-merged outputs/vime-fixtures/zh-generation-v1-lm-r2/cases.jsonl $O/search`。
   - 预期的 `vs_v2_package`：全部或几乎全部用例排序完全一致。差异只允许出现在接近并列的相邻项之间。
   - `vs_fp32_fixture` 应与 v2 任务的数字一致。
10. **Fixture。** `$C fixture $O/int8-merged $O/fixture`（如果选了 `-c4` 就用它）。它写出：
    - `step.jsonl` 和 `step.npz`：一个共享 prompt 之后的 16 行，一次 `scored` 和一次 `logits` 调用；
    - `score.jsonl`：kvsequence 任务的 8 个整句打分用例，通过 v3 调用运行。

    预期包对无状态每个值相差约 3e-4 以内，每个求和 1e-3 以内。
11. **客户端资源。**

    ```bash
    $C ios $O/int8-merged $O/ios-kvbatch-v3 --alignment $O/int8-merged-align.json --search $O/search --fixture $O/fixture
    ```

    它写出 `Resources/TinyChineseV1KVBatchV3INT8.mlmodelc` 和 `VimeLMManifestZhV1KVBatchV3.json`（段 `kv_batch_v3`），以及 `KVBatchV3Fixture-*`。现有资源保持不动。
12. **发布副本。** 把选定的 INT8 合并包拷贝到 `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvbatch-v3/`。不要编辑 `RELEASE.json` 或 `SHA256SUMS.txt`。
13. **iPhone。** 不在本会话中。Vime 用自己的测试宿主测量。

## 决策规则

- 满足以下条件时**发布 v3**：
  - 所选变体的两次对齐都通过；
  - 搜索与 v2 一致；
  - fixture 一致；
  - `v3-alone` 或 `v3` 在加载时和峰值时都明显比 `v2-shipped` 用更少内存。
- 只有当 `-c4` 降低的打分峰值超过它额外增加的包大小，且延迟与 `-c1` 相差约 10% 以内时，才**选择 `-c4`**。
- **无状态包。** 如果 `v3-alone` 覆盖了 Vime 使用无状态包的场景（不带前缀和行缓存的 `logits` 函数就是同一次前向），要报告出来。这样 Vime 就可以去掉无状态资源。

## 给 Vime 的说明（不属于本会话）

- Prompt 作为 `prefix_*` 只传一次，每行自身的 token 放在该行自己的缓存中。不要把 prompt 拷贝进各行。
- 整句打分用 `score`，并像之前一样限制每次调用的 T（8 或 16）。
- 在大调用之前检查 `os_proc_available_memory()`，并据此确定 B 和 T。这一点在 6 GB 设备上尤其适用。

## 报告与回传

按之前报告的格式写 `docs/zh/reports/mac-<date>/zh-v1-kvbatch-v3.md`。内容应包括：

- 环境；
- 每个阶段，包括作为证据保留的失败；
- 包大小，合并对独立、对已发布的 v2 两个包；
- 对齐表；
- 延迟表；
- 内存表，包括 T 曲线；
- 搜索和 fixture 摘要；
- 关于 v3、`-c4` 和无状态包问题的决定。

带回一个附 SHA256SUMS 的 ZIP，包含：

- `$O`，空间不够时可不带 FP32 包；
- 报告；
- 发布副本；
- `ios-kvbatch-v3/`；
- 对 `src/vimeml/deployment/zh_v1_batch3.py` 或 `scripts/zh/deployment/coreml_zh_v1_batch3.py` 的任何改动，或说明没有改动。

不要在 Mac 上提交。
