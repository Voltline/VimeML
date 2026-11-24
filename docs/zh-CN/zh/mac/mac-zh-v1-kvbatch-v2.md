# vimeml-tiny-zh-v1 批量 KV 解码 v2：Mac 任务说明

这是批量 KV 包的第二个 Mac 任务。第一个是 [mac-zh-v1-kvbatch.md](mac-zh-v1-kvbatch.md)，其报告是 [mac-20261115/zh-v1-kvbatch.md](../reports/mac-20261115/zh-v1-kvbatch.md)。

Vime 已在真机上集成 v1 包，延迟达标：每次按键 p50 在 iPhone 13 Pro 上 35 ms，在 16 Pro Max 上 23 ms。内存没有达标。加载批量包并调用一次后，测试进程从 48 MiB 涨到 89 MiB。其中 Vime 自己的 KV 缓冲区占 13.5 MiB；剩下约 27 MiB 是 Core ML。合并包在真机上什么也没省（43.7 MiB，两个独立包为 40.5 MiB）。

## 改动：`zh_v1_batch2.KVBatchGraphZhV2`

1. **只读缓存。**
   - v1 用 `cache*(1-write) + key*write` 把每层的新 key 合并进缓存。这在每层都会对每个缓存做一次完整的 `(B,6,S,64)` 拷贝：18 个缓存大小的临时缓冲区，Core ML 可能按最大的可变形状分配它们。
   - v2 分别注意缓存（掩码 `>= position[b]` 的槽位）和新 token，再用一个 softmax 归一化项把两者合并。不存在缓存大小的中间结果。
   - Mac 上的运行没有证实这一点：可变 T 包和 v1 一样用约 40 MiB，T=1 的图用得多得多。见 [v2 报告](../reports/mac-20261115/zh-v1-kvbatch-v2.md)。
   - 被掩码的槽位可以是任意有限值，但不能是 NaN 或 Inf：被掩码的分数仍会乘以 value 缓存，而 0·NaN 是 NaN。Vime 在真机上发现了这一点（新建的 `MLMultiArray` 不会清零）；调用方要把新的缓存缓冲区清零。这适用于所有 zh-v1 KV 包，包括 chunk 包。
   - `S` 只需 `>= max(position)`（且至少为 1）。v1 的规则 `max(position)+1` 仍然可用。
2. **每行 T 个新 token**（`input_ids [B,T]`，彼此之间因果）。
   - T=1 是解码步。B=1、T = prompt 长度就是 prompt 预填充，所以不再需要 chunk 包。
   - 只返回最后一个新 token 的 logits。`new_key`/`new_value` 为 `(9,B,6,T,64)`。
3. **可选的 scored 输出头。**
   - 搜索桶中的每一行都在同一个按键位置，所以所有行共享一份允许 token 列表：r2 fixture 上中位数 149 个 id，p90 449，最大 536。
   - 输入：`allowed` int32 `[K]`（1..1024）。
   - 输出：对全部 12,288 个 logits 的 `logsumexp` `[B]`，以及 `allowed_logits` `[B,K]`。log P = `allowed_logits − logsumexp`。
   - 输出头矩阵乘仍完整运行。省下的是 `[B,12288]` 输出和客户端 softmax。

Windows 检查（PyTorch FP32，真实 zh-v1 权重）：

| 检查 | 结果 |
| --- | --- |
| 解码，eager 和 traced | 最大 3.96e-5 |
| 预填充，然后从不同位置做多 token 调用 | 最大 2.9e-5 |
| Scored 输出头对 log-softmax | 1.9e-6 |
| r2 搜索，logits 输出头和 scored 输出头 | 125/125 排序完全一致 |

Core ML 端的代码路径在 Windows 上用 PyTorch 图支撑的假 `predict` 跑过：可变、枚举和固定形状、预填充、scored 输出头和搜索。`coremltools` 本身没有在 Windows 上运行。

## 环境准备

使用第一个任务的现有检出目录（`~/Documents/Sources/VimeML-zh-kvbatch`）。其载荷和第一个任务的输出原地保留。

```bash
cd ~/Documents/Sources/VimeML-zh-kvbatch
git pull --ff-only origin main        # needs the commit that adds zh_v1_batch2.py
source .venv/bin/activate
```

第一个任务的两个输出会再次用到：`outputs/deployment/zh-v1-kvbatch/int8-flex/`（v1 基线）和已发布的 chunk 包。所有新输出放在 `outputs/deployment/zh-v1-kvbatch-v2/`（下文记作 `O`）下。绝不覆盖输出。

## 变体

所有变体都用 `python scripts/zh/deployment/coreml_zh_v1_batch2.py convert $O/fp32-<name> <flags>` 构建：

| 名称 | 参数 | 回答的问题 |
| --- | --- | --- |
| `dec-flex` | （默认：B 1..16，T=1，S 1..128，logits） | 仅靠只读缓存能否解决内存问题？ |
| `dec-enum` | `--batch 1,4,8,16 --slots 16,32,64,128` | Vime 的要求：用枚举形状代替 RangeDim |
| `dec-b16` | `--batch 16 --slots 32,64,128` | Vime 的第二个选项 |
| `scored` | `--head scored` | 输出更小，不需要客户端 softmax |
| `unified` | `--tokens flex` | 预填充和解码在一个包里（不需要 chunk 包） |
| `unified-scored` | `--tokens flex --head scored` | 两者兼有 |

`dec-enum` 和 `dec-b16` 同时在多个输入上使用枚举形状。如果 coremltools 拒绝，把错误作为证据保留，并退回到固定形状：

- B=16，S 固定为 32、64 或 128（`--batch 16 --slots 32` 等）；
- 报告每个固定包的内存。

对每个变体：

1. `align $O/fp32-<name> $O/fp32-<name>-align.json`：对 PyTorch FP32，容差 3e-4。如果转换或对齐失败，在 `zh_v1_batch2.py` 中修复，重跑 `check`，并记录改动。
2. `compress $O/fp32-<name> $O/fp32-<name>-align.json $O/int8-<name>`。
3. `align $O/int8-<name> $O/int8-<name>-align.json --stateless artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32`：Vime 的门槛。

## 任务

1. **PyTorch 检查。** 运行 `check $O/check.json`。必须通过，与 Windows 上一样。
2. **按上述方式构建全部六个变体并做门槛检查。**
3. **内存**，这是主要问题。

   ```bash
   python scripts/zh/deployment/coreml_zh_v1_batch2.py memory $O/memory.json \
     v1=outputs/deployment/zh-v1-kvbatch/int8-flex  v1+chunk=outputs/deployment/zh-v1-kvbatch/int8-flex+chunk \
     dec-flex=$O/int8-dec-flex  dec-enum=$O/int8-dec-enum  dec-b16=$O/int8-dec-b16  scored=$O/int8-scored \
     unified=$O/int8-unified  unified-scored=$O/int8-unified-scored  dec-flex+chunk=$O/int8-dec-flex+chunk
   ```

   - 每种配置在 3 个全新进程中运行。与键盘中一样，先加载并运行无状态 INT8 包。
   - 该阶段记录 `phys_footprint` 及其生命周期峰值（`proc_pid_rusage` V4）：
     - 无状态包之后；
     - chunk 包之后（`+chunk`）；
     - 加载之后；
     - 24 token 预填充之后（可变 T 包）；
     - 第一次 B=16、S=32 调用之后；
     - 扫过 B 1/4/8/16 × S 16/32/64 之后；
     - B=16、S=128 之后。
   - 检查一次 V4 的 `phys_footprint` 字段是否等于第一份报告使用的 V2 值。
   - 如果能看出来（例如对运行中的进程用 `vmmap --summary` 或 `footprint`），说明剩余的内存去了哪里。
4. **延迟。** 对每个 INT8 变体运行 `measure $O/int8-<name> $O/int8-<name>-latency.json`。
   - 解码网格：B ∈ {1,4,8,16} × S ∈ {16,32,64,128}，同时加载无状态包。
   - Scored 变体在 K=149 和 K=450 下测量。
   - 8/16/32 token 的 prompt 预填充：可变 T 包一次调用，其他包每次调用一个 token。
   - 在 S=16/32 下把 `int8-dec-flex` 和 v1 `int8-flex` 交替测量三次，使 v1→v2 的差异不是噪声。v1 用 `coreml_zh_v1_batch.py measure` 测量。
5. **搜索。** 对每个 INT8 变体运行 `search $O/int8-<name> outputs/vime-fixtures/zh-generation-v1-lm-r2/cases.jsonl $O/search-<name> --reference-batch outputs/deployment/zh-v1-kvbatch/int8-flex`。
   - Prompt 用包本身预填充。对非可变 T 的变体，传 `--prompt chunk` 改用 chunk 包。
   - 预期的 `vs_reference_batch`：v1 INT8 批量运行，它等于无状态 INT8 搜索。全部或几乎全部用例排序完全一致；差异只出现在接近并列的相邻项之间。
   - Scored 输出头只改变归一化项在哪里计算，所以应与 logits 输出头表现相同。
   - 还要运行 `fp32-dec-flex` 和 `fp32-unified`。它们的 prompt 现在也是 FP32，所以 `vs_fp32_fixture` 应完全一致，分数在 1e-3 以内。
6. **选择。** 选出要发布的 INT8 变体。按以下顺序优先：
   1. 在已加载无状态包的情况下新增内存占用最低；
   2. 不需要 chunk 包；
   3. 延迟在 Vime 目标内：13 Pro 上 B=16 ≤ 8 ms、B=8 ≤ 5 ms（Mac 大约快 2 倍）。

   在报告中说明取舍。然后：

   ```bash
   python scripts/zh/deployment/coreml_zh_v1_batch2.py fixture $O/int8-<chosen> $O/fixture
   python scripts/zh/deployment/package_zh_v1_kvbatch_ios.py --batch $O/int8-<chosen> --alignment $O/int8-<chosen>-align.json \
       --search $O/search-<chosen> --fixture $O/fixture --output $O/ios-kvbatch-v2
   ```

   它写出 `Resources/TinyChineseV1KVBatchV2INT8.mlmodelc` 和 `VimeLMManifestZhV1KVBatchV2.json`（段 `kv_batch_v2`）。v1 资源保持不动。
7. **发布副本。** 把选定的 INT8 包拷贝到 `artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvbatch-v2/`。不要编辑 `RELEASE.json` 或 `SHA256SUMS.txt`。
8. **iPhone。** 不在本会话中。Vime 用自己的测试宿主测量。

## 报告与回传

按第一份报告的格式写 `docs/zh/reports/mac-<date>/zh-v1-kvbatch-v2.md`。内容应包括：

- 环境；
- 每个阶段，包括作为证据保留的失败；
- 覆盖所有配置的内存表；
- 延迟表；
- 搜索结果；
- 选择及原因。

报告中不出现基准文本。

带回一个附 SHA256SUMS 的 ZIP，包含：

- `outputs/deployment/zh-v1-kvbatch-v2/`，空间不够时可不带 FP32 包；
- 报告；
- 发布副本；
- `ios-kvbatch-v2/`；
- 对 `src/vimeml/deployment/zh_v1_batch2.py` 的任何改动，或说明没有改动。

不要在 Mac 上提交。
