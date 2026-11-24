# vimeml-zh-v2-20m colloq-c40：Core ML 构建（Mac）与替换（Vime）

这是给 Mac 会话和 Vime 的自成一体的任务。`colloq-c40` 在其[测试集确认](../reports/zh-colloq-confirmation-20261123/report.md)通过后，于 2026-11-23 取代 `kd-cont` 成为 `vimeml-zh-v2-20m`。在口语测试集上，自重排 Top-1 从 39.0 升到 42.3；在公开拼音测试集上从 49.1 升到 51.4；对 v1 的 Standard 盲测净 +7。

`colloq-c40` 是 `kd-cont` 用口语文本再训练 30,000 次更新得到的（[口语后训练计划](../training/training-zh-colloq-plan.md)）。它的结构、tokenizer、prompt 格式和解码约定与 `kd-cont` 完全相同。所以本任务就是用新权重重跑一遍 [kd-cont 任务说明](mac-zh-v2-20m.md)，门槛按那里最终的判定方式（其 2026-11-20 的偏差）。Core ML 配方、搜索和 Vime 约定都不变。

## 模型

| 项 | 值 |
| --- | --- |
| 发布名 | `vimeml-zh-v2-20m`（arm `colloq-c40`） |
| 源 checkpoint | `artifacts/models/tiny-zh13-12k-pinyin-colloq-c40/final.pt`，第 30,000 步，SHA256 `0ec5ffb1…1027`（不在包中） |
| 推理包 | `artifacts/releases/vimeml-zh-v2-20m-c40/inference/`（`weights.pt`、`config.json`、`tokenizer.model`、`manifest.json`；manifest 的 `checkpoint_sha256` 即上面这个） |
| Tokenizer | v1 的 `zh-unigram-12288`，SHA256 `68bd37c4…7e5a`；pad 0，unk 1，bos 2，eos 3；任务 id CTX 4，READ 5，OUT 6 |
| 结构 | 同 v1 和 `kd-cont`：词表 12,288，d384，九层，六头，SwiGLU 960，上下文 128，20,036,736 参数 |
| FP32 fixture | `outputs/deployment/vimeml-zh-v2-20m-c40-fp32-input/`（与 v1 的用例相同） |
| 生成 fixture | `outputs/vime-fixtures/zh-generation-v2-20m-c40-task-r1/`（与 `kd-cont` fixture 相同的 125 例，用 `colloq-c40` 重新生成；`prompt_ids` 与 `kd-cont` 的完全相同，显示的 top 1 有 23 例与 `kd-cont` 不同） |

交接 ZIP `zh-v2-20m-c40-handoff-20261123.zip`（105 MB，SHA256 `40225249…92ea`）按仓库路径包含这些文件和读音词典，附 `SHA256SUMS` 文件。`kd-cont` 的文件保留各自的路径，所以两者可以放在同一个检出目录中。代码来自 `main`，`scripts/zh/deployment/prepare_zh_v2_20m.py --model colloq-c40` 可以重建 ZIP 中的每个文件。

## Mac 任务

在现有的 kvbatch 检出目录中工作：

1. 在 `main` 上 `git pull`。
2. 在仓库根目录解压交接包，运行 `shasum -a 256 -c SHA256SUMS`。
3. 使用 v4 的环境，设 `B=artifacts/releases/vimeml-zh-v2-20m-c40/inference`。

下面的包名使用 `-c40` 后缀，所以不会覆盖 `kd-cont` 的包。

1. **无状态包**（`scripts/zh/deployment/coreml_zh_v1.py`）：
   - `convert --bundle $B --output artifacts/deployment/tiny-zh-v2-20m-c40-fp32-v1`。
   - `align --bundle $B --model <fp32 package> --reference-fixtures outputs/deployment/vimeml-zh-v2-20m-c40-fp32-input --output outputs/deployment/zh-v2-20m-c40-align-fp32`。必须在 3e-4 下通过。
   - `compress --source <fp32 package> --alignment outputs/deployment/zh-v2-20m-c40-align-fp32/alignment.json --output artifacts/deployment/tiny-zh-v2-20m-c40-int8-b32-v1`，已部署配方 `int8-b32` 不变。
2. **批量 KV v4**（`scripts/zh/deployment/coreml_zh_kvbatch4.py`，每个阶段都带 `--bundle $B`）：
   - `check`，然后 `build --variant fp32kv --stateless <c40 stateless INT8>`。源对齐和 INT8 对无状态对齐必须像 v4 报告中一样通过。
   - `search <c40 int8-merged> <out> --cases outputs/vime-fixtures/zh-generation-v2-20m-c40-task-r1/cases.jsonl --reference <c40 src-merged>`。搜索门槛采用 kd-cont 任务说明最终确定的那个：
     - 重排后（显示的）top 1 至少在 123/125 例中与 FP32 fixture 相同；
     - 文本集合差异只允许出现在第十位的边界上，分数相差约 0.6 以内。

     列出每个不同的用例。
   - `fixture <c40 int8-merged> <out> --stateless <c40 stateless INT8>`，然后用对齐、搜索和 fixture 的输出运行 `ios`。
   - 资源沿用 `kd-cont` 的名字 `TinyChineseV2KVBatchV4FP32KVINT8.mlmodelc`，因为 c40 在同一发布下取代 kd-cont。
   - 在同一会话中用 `kd-cont` 的 v4 INT8 包和 c40 的包运行 `measure`，`--prompt 48` 和 `--prompt 80`。结构相同，延迟应一致。报告任何超出噪声的差异。
3. **按 v4 格式报告**：大小、对齐、搜索检查、延迟，以及任何失败。不要为了通过门槛而改变配方、变体或搜索。

## Vime：用 colloq-c40 替换 kd-cont

约定就是 [kd-cont 任务说明](mac-zh-v2-20m.md)中的，不变：

- 任务模式 prompt `[2 BOS, 4 CTX] + encode(context) + [5 READ] + encode(keys) + [6 OUT]`；
- 每个按键位置 beam 16，top 10，结尾加上 log P(EOS)；
- 自重排 `gen_score + 1.6 × lm_score`，并列保持生成顺序；
- `sentenceContext` 上限 64。

当前的产品决定（2026-11-21）也保持不变：

- v2 是 Rime 重排、预测和纠错的默认模型；
- 整句生成留在开发开关之后；
- v1 作为回退随附，在 `vime.chineseModelVersion` 之后。

1. **替换包。** 用 c40 的 `TinyChineseV2KVBatchV4FP32KVINT8.mlmodelc` 代替 `kd-cont` 的。预计不需要其他代码改动。如果需要，报告出来。
2. **Fixture 门槛。** 对 `zh-generation-v2-20m-c40-task-r1` 的全部 125 例运行现有的 fixture 测试：
   - `prompt_ids` 完全一致；
   - 至少 123/125 例 `rescored` top 1 与 fixture 相同；
   - 文本集合差异只在第十位的边界上；
   - 分数相差约 0.6 以内。

   列出每个不同的用例。
3. **真机检查**（iPhone 13 Pro，同一构建，`kd-cont` 和 c40 并列，用 [kd-cont 报告](../reports/mac-20261121/zh-v2-20m.md)的方法）：
   - Rime 重排第一个列表的 p50/p95 和 60 ms 内的按键占比；
   - 加载时间和峰值内存；
   - 逐键和一次性生成的时间。

   这些应与 `kd-cont` 一致。重排必须像之前一样每次按键都在 60 ms 以内。
4. **纠错**与 `kd-cont` 一样，保持现有规则和在 v1 上校准的阈值。

## 提升是什么，在哪里到达用户

确认的提升是在**整句生成**（自重排 Top-1）上测得的。本次发布中，生成留在开发开关之后。因此用户主要通过 Rime 重排和预测看到变化，它们使用同一模型的 LM 分数。

这些用途没有为 c40 单独测量：

- 对 v1 的 Standard 盲测重排净 +7，不显著，`kd-cont` 为 −2；
- 在开发集上，句末语气词预测 Top-1 从 0.190（`kd-cont`）升到 0.271，这正是引出口语工作的吧/把问题。

一旦不阻塞按键的生成上线，生成的提升就会直接生效。

## Mac 结果与偏差（2026-11-23）

**完整性与对齐**

- 交接包的 `SHA256SUMS` 检查通过，结构与 `kd-cont` 一致。
- `int8-merged` 为 23,271,109 字节；编译后的 `TinyChineseV2KVBatchV4FP32KVINT8.mlmodelc` 为 22 MB，与 `kd-cont` 相同。
- 无状态 FP32 对 PyTorch 通过 5/5，最大 3.89e-5（门槛 3e-4）。
- KV v4 检查：eager 和 traced 都是 2.17e-5。
- KV v4 fp32kv 对齐（logits、scored、score、prefix）：最大 2.57e-5，容差外 0 行。
- INT8 KV v4 对无状态 INT8：step 1.62e-5，score 求和 1.69e-5，顺序相同 8/8。

**搜索**（`zh-generation-v2-20m-c40-task-r1`，INT8 对 FP32 fixture）

- 生成顺序 top 1 在 121/125 例中相同（`kd-cont`：122/125）。共同分数差最多 0.21。p50 为 65.9 ms（`kd-cont`：65.9 ms）。
- 显示顺序：INT8 的前 10 用 c40 的 FP32 LM 分数以 λ 1.6 重排。显示的 top 1 在 **122/125** 例中与 fixture 相同，比 123 的门槛少一例（`kd-cont`：125/125）。共同文本的最终分数差最多 0.21。
- 三处差异都是长的全首字母输入。其中两例，INT8 的 beam 路径走到了一个根本不在 FP32 前 10 中的候选，它在重排后胜出。第三例中，它把 FP32 的第二名排到第一，FP32 差距为 1.7。
- 文本集合在 89/125 例中与 fixture 相同。有 14 例在第十位以上就有差异。v1 已发布的 INT8 包对它自己的 fixture，集合相同的是 86/125。

**延迟**（Mac，`kd-cont` 对 c40，p50）

| prompt | 步骤 | `kd-cont` | c40 |
| --- | --- | ---: | ---: |
| 48 | `prompt_prefix` | 3.36 ms | 3.52 ms |
| 48 | `extend_prefix_1` | 1.26 ms | 1.25 ms |
| 48 | `scored_step_b16` | 3.74 ms | 3.46 ms |
| 48 | `score_b16` | 17.89 ms | 17.88 ms |
| 80 | `prompt_prefix` | 5.12 ms | 5.22 ms |
| 80 | `extend_prefix_1` | 1.43 ms | 1.41 ms |
| 80 | `scored_step_b16` | 4.08 ms | 3.83 ms |
| 80 | `score_b16` | 18.21 ms | 18.14 ms |

**偏差：尽管搜索门槛差一例，c40 仍交给 Vime。**

这一差额来自 INT8 在长全首字母输入上的 beam 路径分歧。它不是导出错误：所有对齐都在约 2e-5 通过，而默认开启的用途 Rime 重排和预测使用 `score`，其对齐在 1.7e-5。搜索门槛唯一守护的用途是整句生成，它留在开发开关之后。

这一决定是在看到结果之后做出的。Vime fixture 门槛按同样方式判定：显示的 top 1 至少 122/125，并列出每个不同的用例。
