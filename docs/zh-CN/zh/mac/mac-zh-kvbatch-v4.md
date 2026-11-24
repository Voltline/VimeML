# vimeml-tiny-zh 批量 KV v4（`prefix` 函数、FP16 KV）：Mac 任务说明

这是一个自成一体的 Mac 任务。它接在 v3 任务（[mac-zh-v1-kvbatch-v3.md](mac-zh-v1-kvbatch-v3.md)，报告 [mac-20261118/zh-v1-kvbatch-v3.md](../reports/mac-20261118/zh-v1-kvbatch-v3.md)）之后；不需要那个会话目录中的任何内容。它使用 zh-v1 的权重。vimeml-zh-v2-20m 的结构相同，所以之后换掉推理包，就能用它的权重重建选定的变体。

## 原因

Vime 在 iPhone 13 Pro 和 iPhone 16 Pro Max 上分析了已发布的 v3 包（`TinyChineseV1KVBatchV3INT8.mlmodelc`）（iOS 18，仅 CPU，以 2 ms 间隔采样 `phys_footprint` 及其内核峰值）。两台设备结果一致：

| 加载顺序 | 第一个函数之后 | 第二个之后 | 然后对 16 个长句打分 |
| --- | --- | --- | --- |
| `score`，然后 `scored` | 稳定 +1.4 MB（瞬时 +20.7 MB） | **稳定 +27.3 MB，不释放** | 峰值 72.2–72.8 MB |
| `scored`，然后 `score` | 稳定 +1.8 MB，但有 **+48.1 MB 的瞬时峰值**（总计 77.9 MB） | +0.1 MB | 稳定约 31.7 MB |

客户端用 `scored` 加上一个占位的 `allowed: [0]` 构建共享前缀 KV，并随按键逐步扩展它。这次调用不需要输出头，也不需要 logsumexp。由此引出两个问题：

1. **没有输出头的函数能否避开多出的 27 MB？** 权重是文件映射的，由 v3 的三个函数共享（v3 报告：一个 23.1 MB 的包）。多出的 27 MB 大约是一份权重，所以它可能来自 Core ML CPU 后端按函数重新打包权重，那样 `prefix` 函数也未必能避开。下面的加载顺序矩阵在 Mac 上测试这一点。Vime 在 iPhone 上重复测试。
2. **FP16 KV 输入输出是否损失准确率？** 它让前缀 KV（FP32 下每 token 27 KB）和每个行缓存减半。在 Windows 上，PyTorch 把每个 KV 输入输出都舍入到 FP16 后，每个 logit 最多差 3.6e-3，每个句子求和最多差 1.8e-3，beam 搜索在全部 125 个 fixture 用例中给出相同排序（排序完全一致 125/125，top 1 125/125）。FP16 计算是第三个变体；这里第一次测试。

## 改动：`vimeml.deployment.zh_v1_batch4`

v3 的三个函数（`logits`、`scored`、`score`，不变；见 [v3 任务说明](mac-zh-v1-kvbatch-v3.md)）加上：

| 函数 | 输入 | 输出 |
| --- | --- | --- |
| `prefix` | `input_ids` [1,T]、`prefix_length` [1]、`prefix_key`/`prefix_value` (9,1,6,P,64) | 只有 `new_key`/`new_value` (9,1,6,T,64) |

- 没有最后的 norm、没有输出头、没有 logsumexp。Token t 位于 `prefix_length + t`。
- 第一次调用：`prefix_length` 为 0，带一个零槽位。之后的调用做扩展：把 `new_key`/`new_value` 追加到有效槽位之后（`zh_v1_batch4.prefix_call`）。
- `score_continuations(..., prefill="prefix")` 在整句打分中使用它。

三个变体，每个都包含全部四个函数，并按已部署的 INT8 block-32 配方压缩：

| 变体 | KV 输入/输出 | 计算 | 容差（每个值 / 每个求和） |
| --- | --- | --- | --- |
| `fp32kv` | FP32 | FP32 | 3e-4 / 1e-3（v3 的） |
| `fp16kv` | FP16 | FP32 | 1e-2 / 1e-2 |
| `fp16kv-fp16c` | FP16 | FP16，规约、exp/log 和 norm 除外 | 1e-1 / 2.5e-1（只用来发现转换错误） |

KV 张量是 `key_cache`、`value_cache`、`prefix_key`、`prefix_value`、`new_key` 和 `new_value`。Logits、`logsumexp`、`allowed_logits` 和 `target_logprobs` 保持 FP32。对 fp16 变体，起决定作用的门槛是搜索检查，而不是容差。

Windows 检查（`check` 阶段，zh-v1 权重）：eager 和 traced FP32 最大 3.24e-5；FP16 舍入的 KV 最大 3.57e-3；单元测试 `tests/test_deployment_zh_kvbatch4.py` 5/5。`coremltools` 没有在 Windows 上运行。

## 环境准备

交接 ZIP 中是不在 git 里的文件：zh-v1 发布的子集（推理包、无状态 INT8、v3 INT8）、读音词典和生成 fixture。先按其 `SHA256SUMS` 核对。

```bash
git clone https://github.com/Voltline/VimeML.git VimeML-kvbatch4 && cd VimeML-kvbatch4
git log --oneline -1                     # needs the commit that adds zh_v1_batch4
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e . torch==2.7.1 coremltools==9.0 numpy sentencepiece==0.2.1
unzip /path/to/kvbatch4-handoff.zip -d .   # creates artifacts/ and outputs/ at these paths:
#   artifacts/releases/vimeml-tiny-zh-v1-hf-v1/{inference,coreml/ios18-int8-block32,coreml/ios18-int8-block32-kvbatch-v3}
#   outputs/zh-readings/dict/vime_pinyin.dict.yaml
#   outputs/vime-fixtures/zh-generation-v1-lm-r2/{cases.jsonl,meta.json}
O=outputs/deployment/zh-kvbatch-v4
C="python scripts/zh/deployment/coreml_zh_kvbatch4.py"
python -m unittest tests.test_deployment_zh_kvbatch4 tests.test_deployment_zh_v1_batch3
```

v3 任务使用 Python 3.11、PyTorch 2.7.1、coremltools 9.0 和 sentencepiece 0.2.1。如果 `pip install -e .` 装了别的版本，记录下来。每个阶段都拒绝覆盖输出。

## 阶段

1. **检查。** `$C check $O/torch-check.json` 必须打印 `passed`。
2. **构建**三个变体（转换、合并、对 PyTorch 对齐、压缩、合并、对无状态 INT8 包对齐）：

   ```bash
   for v in fp32kv fp16kv fp16kv-fp16c; do $C build $O --variant $v 2>&1 | tee $O-build-$v.log; done
   ```

   - 每个变体写出 `$O/<variant>/`：`src-<function>`、`src-merged`、`src-merged-align.json`、`int8-<function>`、`int8-merged` 和 `int8-merged-align.json`。
   - 如果某次转换失败（例如在 FP16 输入/输出类型或 FP16 op 选择上），在 `zh_v1_batch4.py` 中修复，重跑 `check` 和单元测试，重建该变体，并记录改动。
   - 如果某个变体的源对齐失败，停止该变体。它的失败和对齐文件就是结果。
   - 记录包字节数。四个函数应仍只存一份权重，与 v3 一样约 23 MB。
3. **加载顺序（内存），主要问题。**

   ```bash
   $C loadorder $O/loadorder.json v3=artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvbatch-v3 \
     fp32kv=$O/fp32kv/int8-merged fp16kv=$O/fp16kv/int8-merged fp16kv-fp16c=$O/fp16kv-fp16c/int8-merged
   ```

   - 每次运行是一个全新进程。在 `start` 之前导入 coremltools 并构建输入，模型从进程外编译好的 `.mlmodelc` 加载。
   - 顺序：`score>scored`、`scored>score`、`score>logits`、`logits>score`、`score>prefix`、`prefix>score`、`score>prefix>scored`。包中没有的函数所在的顺序会跳过。
   - 负载为 10 轮。每轮是一个 48 token 前缀和 8 次单 token 扩展，若已加载 `prefix` 就用它，否则用 `logits`，再否则用 `scored`；然后仅在加载了 `scored` 时做 16 次 B=16 beam 步；然后一次 B=16、T=16 的 `score` 调用。
   - 按包和顺序报告每次加载后的稳定值、`load_peak`、`after_rounds` 和 `peak`。
   - 问题是 Mac 能否复现 Vime 看到的现象：`score>scored` 比 `scored>score` 多保留约 27 MB。然后检查 `score>prefix` 或 `score>logits` 能否避开它。
4. **延迟。** 比较同样的四个包：

   ```bash
   $C measure $O/latency.json v3=artifacts/releases/vimeml-tiny-zh-v1-hf-v1/coreml/ios18-int8-block32-kvbatch-v3 \
     fp32kv=$O/fp32kv/int8-merged fp16kv=$O/fp16kv/int8-merged fp16kv-fp16c=$O/fp16kv-fp16c/int8-merged
   ```

   它测量一个 48 token 前缀、一次单 token 扩展、一次 B=16 beam 步和一次 B=16、T=16 的打分调用。v3 通过 `scored` 构建前缀。
5. **搜索**，对每个构建成功的变体：

   ```bash
   $C search $O/fp32kv/int8-merged $O/search-fp32kv
   for v in fp16kv fp16kv-fp16c; do $C search $O/$v/int8-merged $O/search-$v --reference $O/fp32kv/int8-merged; done
   ```

   - `fp32kv` 与 v3 比较。预期 125 例中全部或几乎全部排序完全一致；差异只允许出现在接近并列的相邻项之间。
   - fp16 变体与 `fp32kv` 和 FP32 fixture 比较。
6. **Fixture**，对每个通过搜索的变体：`$C fixture $O/<variant>/int8-merged $O/fixture-<variant>`。
7. **客户端资源**，对每个有 fixture 的变体：

   ```bash
   $C ios $O/<variant>/int8-merged $O/ios-<variant> --alignment $O/<variant>/int8-merged-align.json \
     --search $O/search-<variant> --fixture $O/fixture-<variant>
   ```

   Vime 在 iPhone 上测量这些资源；Mac 的结果不单独决定变体。

## 决策规则

这些规则在任何结果出来之前就已确定。最终选择使用 Vime 在 iPhone 上对同样顺序测得的数字。

- **`prefix` 函数。** 如果在 iPhone 上 `score>prefix`（两种顺序）的稳定内存占用比 `score>scored` 低至少 10 MB，且加载峰值不更高，就保留它。如果 `score>logits` 能省下同样多，客户端改用 `logits`，去掉 `prefix`（少一个函数）。如果没有任何顺序能避开多出的内存，原因在 Core ML 的加载而不在输出头，去掉 `prefix`。
- **`fp16kv`。** 满足以下条件时采用：其 INT8 无状态对齐通过；对 `fp32kv` 的搜索在全部 125 例中 top 1 相同，至少 120 例排序完全一致，其余只是接近并列的互换；fixture 的分数顺序在每个用例中都一致；beam 负载下 iPhone 峰值更低。
- **`fp16kv-fp16c`。** 只有满足 `fp16kv` 的所有条件，并且在 iPhone 上比 `fp16kv` 更快或更轻时才采用。转换或对齐失败是一个结果，不是需要绕过的东西。
- **本任务不再增加变体。** INT4 权重、ANE 和其他布局不在范围内。

## 报告与回传

按之前报告的格式写 `docs/zh/reports/mac-<date>/zh-kvbatch-v4.md`。内容应包括：

- 环境；
- 每个阶段，失败作为证据保留，以及任何代码改动；
- 每个变体的包大小（合并对独立）；
- 对齐表；
- 加载顺序表；
- 延迟表；
- 搜索和 fixture 摘要；
- Mac 数字对每条决策规则说明了什么。

带回一个附 SHA256SUMS 的 ZIP，包含：

- 不含 `src-*` 包的 `$O`；
- 构建日志；
- 报告；
- `ios-*/`；
- 对 `zh_v1_batch4.py` 或 `coreml_zh_kvbatch4.py` 的任何改动（以 patch 形式），或说明没有改动。

不要在 Mac 上提交或推送。

## 给 Vime 的说明（不属于本会话）

- 用 `ios-*` 资源在两台 iPhone 上重复加载顺序矩阵：同样的七种顺序和同样的负载。
- 对每种顺序，报告每次加载后的稳定值、加载瞬时峰值和峰值。
- 对 fp16 变体，KV 以 Float16 `MLMultiArray` 传入。Logits、对数概率和句子分数保持 Float32。
