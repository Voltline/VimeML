# vimeml-ja-v2-20m：Core ML 构建（Mac）与真机测试（Vime）

[English version](../../../ja/mac/mac-ja-v2-20m.md)

这是给 Mac 会话和 Vime 的自成一体的任务，对应 [ja-gen 计划](../training/training-ja-gen-plan.md)第 5 阶段。第 4 阶段的 [test 确认](../reports/ja-gen-confirmation-20261128/report.md)通过后，`colloq-c40` 作为 `vimeml-ja-v2-20m` 发布，替换现在部署的 V4.2（AzooKey N-best + V4.2 重排）。kana test 宽松 Top-1 从 510 升到 528（p 0.0024）；口语 test 从 509 升到 537。

确认的是一个完整的方式，而不只是一个模型：

- AzooKey N-best 20 与模型生成的前 10 合并；
- 合并后的每个候选都用同一模型打两个分，按 `gen_score + 1.6 × lm_score` 排序。

本任务要在手机上复现这个方式，并测出它的延迟和内存。

模型块与 `vimeml-zh-v2-20m` 相同（12K 词表，d384，九层，上下文 128），所以 Core ML 流程就是中文 v2 的：

- [mac-zh-v1-coreml.md](../../zh/mac/mac-zh-v1-coreml.md) 中的无状态 INT8 包；
- [mac-zh-kvbatch-v4.md](../../zh/mac/mac-zh-kvbatch-v4.md) 中的批量 KV v4 包（`fp32kv`，带 `prefix`）。

不同的是三处：权重和 tokenizer 是日文的；搜索按平假名读音位置分桶，用读音索引；显示列表合并 AzooKey 候选。

## 模型

| 项 | 值 |
| --- | --- |
| 发布名 | `vimeml-ja-v2-20m`（arm `20m-colloq-c40`） |
| 源 checkpoint | `artifacts/models/ja-kana-20m-colloq-c40/final.pt`，SHA256 `19b04ddf…d15f`（不在包中） |
| 推理包 | `artifacts/releases/vimeml-ja-v2-20m/inference/`（`weights.pt`、`config.json`、`tokenizer.model`、`manifest.json`） |
| Tokenizer | `ja-unigram-12288`，SHA256 `61d71460…aa53`；pad 0，unk 1，bos 2，eos 3；任务 id CTX 4，READ 5，OUT 6；有字节回退 |
| 结构 | 词表 12,288，d384，九层，六头，SwiGLU 960，上下文 128 |
| 读音索引 | `outputs/ja-gen-stage2/kana-index.json`，SHA256 `b60c485d…de5b`：平假名读音 → 允许的 token id |
| FP32 fixture | `outputs/deployment/vimeml-ja-v2-20m-c40-fp32-input/`（logits 与 KV 逐步） |
| 生成 fixture | `outputs/vime-fixtures/ja-generation-v2-20m-c40-task-r1/`（`cases.jsonl` + `meta.json`，125 例） |

生成 fixture 的 125 例：

- 120 例按有无左文 × 长度档分层，从 `ime-kana-ja-v1` 开发集抽取，各带开发集导出的 AzooKey N-best 20；
- 5 例手写会话句，只有生成，没有 AzooKey 列表；
- test 没有读取。

每例记录：

- 生成的前 10（`top`，含 `beam_score`）和仅生成的显示顺序 `rescored`；
- 合并列表 `merged`：每个候选的 `gen_score`、`lm_score`、`final_score`，以及算这两个分所用的全部 token id（`gen_prompt_ids`、`gen_output_ids`、`lm_ids`、`lm_from`）；
- 显示顺序 `shown`。

`meta.json` 写明搜索和合并的约定。

交接 ZIP 按仓库路径包含上面的文件和读音表，附 `SHA256SUMS` 文件（文件名、大小与 SHA256 见下文“交接包”）。代码来自 `main`；`scripts/ja/deployment/prepare_ja_v2_20m.py` 可以重建 ZIP 中的每个文件。

## Mac 任务

在现有的 kvbatch 检出目录中工作：

1. 在 `main` 上 `git pull`。
2. 在仓库根目录解压交接包，运行 `shasum -a 256 -c SHA256SUMS`。
3. 使用中文 v4 的环境，设 `B=artifacts/releases/vimeml-ja-v2-20m/inference`。

包名都用 `tiny-ja-v2-20m-` 前缀，不会覆盖中文的包。

1. **无状态包**（`scripts/zh/deployment/coreml_zh_v1.py`）：
   - `convert --bundle $B --output artifacts/deployment/tiny-ja-v2-20m-fp32-v1`。
   - `align --bundle $B --model artifacts/deployment/tiny-ja-v2-20m-fp32-v1 --reference-fixtures outputs/deployment/vimeml-ja-v2-20m-c40-fp32-input --output outputs/deployment/ja-v2-20m-align-fp32`。必须在 3e-4 下通过。
   - `compress --source artifacts/deployment/tiny-ja-v2-20m-fp32-v1 --alignment outputs/deployment/ja-v2-20m-align-fp32/alignment.json --output artifacts/deployment/tiny-ja-v2-20m-int8-b32-v1`，已部署配方 `int8-b32` 不变。
   - 跳过 `evaluate`：它是中文 Standard 评估。
2. **批量 KV v4**（`scripts/zh/deployment/coreml_zh_kvbatch4.py`，每个阶段都带 `--bundle $B`）：
   - `check`，然后 `build --variant fp32kv --stateless artifacts/deployment/tiny-ja-v2-20m-int8-b32-v1`，输出放在 `artifacts/deployment/tiny-ja-v2-20m-kvbatch-v4`。源对齐和 INT8 对无状态对齐必须像中文 v4 报告中一样通过。
   - `fixture <int8-merged> <out> --stateless artifacts/deployment/tiny-ja-v2-20m-int8-b32-v1`。这是 KV 管线的通用 fixture（`prefix` 一次与两次调用、`logits`、`scored`、`score`），用的是中文 v4 的例句。日文 tokenizer 有字节回退，这些例句都能原样还原；它只检查 KV 管线，不检查日文质量。
3. **日文搜索检查**（`scripts/ja/deployment/coreml_ja_v2_20m.py`，替代中文的拼音 `search` 阶段）：

   ```bash
   python scripts/ja/deployment/coreml_ja_v2_20m.py search <int8-merged> outputs/deployment/ja-v2-20m-search-int8 \
       --bundle $B --reference <src-merged>
   ```

   它对每例做三项检查：
   - **generation**：用包的 `logits` 跑按读音位置分桶的受限 beam 16、前 10，与 fixture 的 `top` 比较。
   - **scores**：用 fixture 自己的 token id，以包的 `score` 给 fixture 的合并列表打分，与 `gen_score` / `lm_score` 比较，再按这些分排出显示顺序，与 `shown` 比较。
   - **pipeline**：包自己生成的前 10 与该例的 AzooKey 列表合并，用包打分，即手机上的完整流程。比较显示的 top 1 与 `shown[0]`，仅生成的 top 1 与 `rescored[0]`。

   门槛（INT8 对 FP32 fixture，按中文 v2 最终采用的判定方式，在看到结果之前写定）：
   - scores：显示的 top 1 至少 123/125 例与 `shown[0]` 相同；共同候选的分数差约 0.6 以内。
   - pipeline：显示的 top 1 至少 123/125 例与 `shown[0]` 相同。
   - generation 只报告，不设门槛：用户看到的是合并后的列表，生成顺序的接近并列互换和第十位边界的集合差异在中文 v2 上已经见过。

   列出每个不同的用例（`summary.json` 的 `differing_cases`）。本地已用 PyTorch FP32 KV 图跑过同一脚本（`--torch`）：三项都是 125/125，分数最大差 1.9e-4（见下文“交接包”）。
4. **iOS 资源**：用对齐、日文搜索和 fixture 的输出运行 `coreml_zh_kvbatch4.py ios`：

   ```bash
   python scripts/zh/deployment/coreml_zh_kvbatch4.py ios <int8-merged> <out> --bundle $B \
       --alignment <int8-merged-align.json> --search outputs/deployment/ja-v2-20m-search-int8 --fixture <fixture out>
   ```

   资源名为 `TinyJapaneseV2KVBatchV4FP32KVINT8.mlmodelc`，manifest 为 `VimeLMManifestJaKVBatchV4-fp32kv.json`，与中文资源可以并存。
5. **延迟**：在同一会话中用中文 `vimeml-zh-v2-20m`（c40）的 v4 INT8 包和日文的包运行 `measure`，`--prompt 48` 和 `--prompt 80`。结构相同，延迟应一致；报告任何超出噪声的差异。日文搜索检查的 `p50_search_ms` / `p50_score_ms` 也一并报告（Mac Python，含输入转换）。
6. **报告**：按中文 v4 格式报告大小、对齐、搜索检查、延迟，以及任何失败。不要为了通过门槛而改变配方、变体、搜索、λ 或候选数。

## Vime：用 vimeml-ja-v2-20m 替换 V4.2

现在的日文流程是 AzooKey N-best 加 V4.2 重排。新流程用一个模型做两件事：生成候选，以及给 AzooKey 和生成的候选统一打分。V4.2 的资源保留，作为回退。

1. **资源**：
   - `TinyJapaneseV2KVBatchV4FP32KVINT8.mlmodelc` 和它的 manifest；
   - 推理包的 `tokenizer.model`，起一个不与 V4.2 tokenizer 冲突的资源名；
   - `kana-index.json`，读音 → token id；可以在构建时转成更紧凑的格式，但内容必须一致。
2. **输入**：
   - `reading`：当前未确定的假名转成平假名（片假名转平假名，保留长音符和标点）；
   - `context`：左文，最多 64 字，与中文 `sentenceContext` 相同。
3. **生成**（`vimeml.deployment.ja_search.search_constrained`，`meta.json` 的 `search` 一节）：
   - Prompt：`[2 BOS, 4 CTX] + encode(context[-64:]) + [5 READ] + encode(reading) + [6 OUT]`。左文 token 从左侧截掉，使 prompt 在 128 中至少留出 2 个位置。
   - 左文部分不依赖读音：每个左文用 `prefix` 建一次 KV，然后每次读音变化时用 `[READ] + encode(reading) + [OUT]` 扩展。
   - 搜索按已消耗的平假名字数分桶。每个桶内：
     - 只走读音索引允许的 token（`reading[pos:end]` 在索引中，且 `end` 之后的读音仍可覆盖）；
     - 按 token id 去重，保留 beam 16；
     - 一次批量 decode。
   - 分数是对整个词表归一化的 log-prob，可以用 `scored` 并把要走的 token 与 EOS 放进 `allowed`。
   - 读音消耗完时加上 log P(EOS = 3)。按解码后的文本去重，取最高分，保留前 10。
   - `prompt_ids` 要与 fixture 精确核对。
4. **合并与打分**（`kana_gen_ja.merge_and_rank`，`meta.json` 的 `merge` 一节）：
   - 候选集合：AzooKey 整段读音转换的前 20 个候选，按 AzooKey 顺序；再接生成的前 10，按生成顺序；按字符串精确去重。
   - `gen_score`：给每个候选（包括 AzooKey 的）打任务模式分数。前缀是 prompt（左文按该候选的输出长度重新截断，使 prompt + 输出 − 1 ≤ 128），续写是 `encode(text) + [3]`；用 `prefix` / `scored` 建前缀，用 `score` 求和（与 `zh_v1_batch3.score_continuations` 相同）。
   - `lm_score`：与中文 `VimeLanguageModel.scores(context:candidates:)` 相同：`BOS + encode(context + text)` 一次联合调用，从与 `BOS + encode(context)` 的公共前缀之后开始求和。
   - 排序：`gen_score + 1.6 × lm_score` 降序，并列保持合并顺序；只有一个分数的候选排在两个分数都有的之后，没有分数的排最后。
   - fixture 给出每个候选的 token id 和分数。
5. **显示时机**（计划第 5 阶段，沿用中文 2026-11-21 的产品决定）：生成不阻塞按键。
   - 每次按键先显示 AzooKey 前 20 按同样分数重排后的列表。这一步不需要生成，每个候选两次打分。
   - 停顿后或在后台完成生成，把生成的候选合并进来，按上面的规则重新排序后更新列表。
   - 用户在生成完成前确定转换时，用当时的重排列表。
6. **Fixture 门槛**（全部 125 例）：
   - `prompt_ids` 完全一致；
   - `merged` 中每个候选的 `gen_prompt_ids`、`gen_output_ids`、`lm_ids`、`lm_from` 完全一致；
   - 用 fixture 的合并列表打分时，显示的 top 1 至少 123/125 例与 `shown[0]` 相同，分数相差约 0.6 以内；
   - 用手机自己的生成加 fixture 的 AzooKey 列表时，显示的 top 1 至少 123/125 例与 `shown[0]` 相同。

   列出每个不同的用例。
7. **真机测试**（iPhone 13 Pro 和 iPhone 16 Pro Max，同一构建中 V4.2 与 v2 并列，键盘扩展中逐键端到端，用中文 [v2 报告](../../zh/reports/mac-20261121/zh-v2-20m.md)的方法）：
   - 重排列表（第 5 步第一步）：p50/p95，以及 60 ms 内的按键占比；
   - 完整流程（生成 + 合并 + 打分）：逐键和停顿后一次性的 p50/p95；
   - 加载时间、峰值内存，以及键盘扩展的内存余量；
   - 与 V4.2 重排的同项数字并列。

   发布条件：13 Pro 上的重排列表与 V4.2 一样每次按键都在 60 ms 以内。完整流程只报告，按第 5 步在后台运行。
8. **回退**：V4.2 资源保留，放在一个同时切换模型和流程的开关之后（命名沿用中文的 `vime.chineseModelVersion` 风格），审核后发现问题时不需要重新构建。

## 不做的事

- 不重新训练，不改 λ、beam、前 10 或 AzooKey 候选数。
- 不读 test 或 blind。fixture 只来自开发集。
- 不为了通过门槛而改变配方、变体或搜索。不通过就如实报告；是否偏离由 VimeML 一侧在看到结果后记录。

## 回传

- **Mac**：
  - 报告（中文）；
  - `outputs/deployment/ja-v2-20m-align-fp32/`、KV v4 的 `*-align.json`、`outputs/deployment/ja-v2-20m-search-int8/`（`summary.json`、`cases.jsonl`）、`measure` 的输出；
  - `ios` 输出中的 manifest。

  这些打成一个 ZIP，不要提交到 VimeML。
- **Vime**：
  - fixture 测试输出；
  - 真机测量表；
  - 改动的提交号（在 Vime 仓库中）。

## 交接包

`ja-v2-20m-c40-handoff-20261128.zip`，97,917,499 字节，SHA256 `05d6f57c…9325`。共 10 个文件和 `SHA256SUMS`，由 `prepare_ja_v2_20m.py`（默认 `--model colloq-c40`）的 `weights`、`bundle`、`fixture`、`package` 四个阶段构建，构建环境为 Windows PyTorch CPU FP32。

- **权重**：源 checkpoint `19b04ddf…`，第 30,000 步，20,036,736 参数。
  - `deployment.pt` 为 `16e798c2…`，不含优化器状态。
  - 推理包 manifest 为 `325c6aa8…`。
  - KV 逐步对无状态最大差 1.9e-5。
- **生成 fixture**（`cases.jsonl` `9cf4c102…`）：
  - 125 例，生成回退 0，LM 打分回退 0。
  - 合并列表每例 10–30 个候选，中位数 18。
  - 120 个开发集例中有 11 例显示的 top 1 来自生成，不在 AzooKey 列表中。
- **与第 3.3 阶段结果核对**：120 个开发集例与第 3.3 阶段判定用的 `colloq-c40` 开发集结果逐条比较：
  - 显示顺序 120/120 完全相同；
  - 生成的前 10 也是 120/120 相同；
  - `gen_score` 最大差 1.8e-4。
- **搜索检查脚本自检**：`coreml_ja_v2_20m.py search --torch`，PyTorch FP32 KV 图：
  - generation、scores、pipeline 三项都是 125/125；
  - 分数最大差 1.9e-4。
