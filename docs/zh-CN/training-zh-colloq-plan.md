# vimeml-zh-v2-20m 口语后训练计划

2026-11-22 起草，训练前提交（预注册）。本文是参考文本，英文版见 [docs/training-zh-colloq-plan.md](../training-zh-colloq-plan.md)。

v2-20m 候选 `kd-cont` 在口语上偏书面：例如“这么说吧”打 `ba` 时会补成“把”。本计划只改训练数据：在 `kd-cont` 的配方上混入口语文本再训练，从模型侧改善口语句的预测和转写；客户端规则不在本计划内。

## 数据

口语数据通过 `[[data.extra]]` 进入训练，构建器、样本规则、分词器和 Rime 读音都与 r1 拼音数据相同。打标和质量筛查只用同一个大模型（DeepSeek 4.1 Flash）。export-r2 不用。

**export-r1 口语段落**（[data-zh-colloq-r1.toml](../../configs/data-zh-colloq-r1.toml)，store `zh-colloq-r1-12k`，manifest sha256 `1dea7399…`）

- [zh_colloq.py](../../scripts/corpus/zh_colloq.py) 先用语气词规则预筛，再用 LLM 标注训练两个分类器（字 1–3 元组哈希特征，逻辑回归）：
  - `p_chat`：段落是真人聊天；
  - `p_spoken`：段落是口语（含小说对白）。
- 训练集取 train 划分里 `p_chat ≥ 0.7` 的段落，共 383 万段、9340 万字。另按哈希抽取 `p_spoken ≥ 0.8` 且 `p_chat < 0.5` 的段落 73 万段、3670 万字，多为小说对白。
- 留出集按文档哈希划分（`zh-colloq-heldout`，2%），只用于评测，不进训练。
- 每段一条样本：30% 纯文本，70% 拼音。

**外部聊天**（[data-zh-colloq-chat.toml](../../configs/data-zh-colloq-chat.toml)，store `zh-colloq-chat-12k`）

- 来源：豆瓣对话、MMChat（微博）、贴吧，经 [zh_chat_external.py](../../scripts/corpus/zh_chat_external.py) 规则清洗，并按对话划出 2% 留出集。
- 每条 turn 都过一遍大模型质量筛查（仇恨、辱骂、色情、政治、违法、垃圾等低质量内容），只保留判为 ok 的。没筛到的一律视为不通过。筛查输入已打乱，所以只筛了一部分的来源也是随机子集。
- `zh_colloq.py export-chat` 按行数导出：豆瓣 80 万、MMChat 60 万、贴吧 12 万。贴吧筛后只有 5.1 万条，重复约 2.35 倍，每份副本用不同的对话键。
- 每条 turn 一个样本，上下文只取 turn 内部，与部署一致：30% 纯文本，70% 拼音。
- 这个 store 的 manifest sha256 在构建后、训练前填入配置；这一步不算偏离。

**口语部分的构成**（按样本）：r1 聊天约 52%，r1 口语约 10%，豆瓣 20%，MMChat 15%，贴吧 3%。c20 下 r1 store 各 kind 最多读 0.7 遍，聊天 store 不到 1 遍。

## 训练臂

四个臂都从 `kd-cont` final（step 30,000）开始，沿用第 3 轮配方：

- 继续训练 30,000 次更新；
- 200M 教师，α = 0.5，T = 1；
- 峰值 LR 1e-4 → 1e-5，warmup 1000；
- seed `20261122`。

每次更新 512 个样本，每个 kind 的总数与 `kd-cont` 相同（plain 160 / full 96 / mixed 128 / initials 96 / error 32），因此 `per_kind` 归一化不变。

| 臂 | 每次更新的口语样本（r1 口语 + 聊天） | 口语上的损失 | 配置 |
| --- | --- | --- | --- |
| `colloq-ctrl` | 0 | — | [train-zh-pinyin-colloq-ctrl.toml](../../configs/train-zh-pinyin-colloq-ctrl.toml) |
| `colloq-c10` | 52（32 + 20），约 10% | 蒸馏 | [train-zh-pinyin-colloq-c10.toml](../../configs/train-zh-pinyin-colloq-c10.toml) |
| `colloq-c20` | 104（64 + 40），约 20% | 蒸馏 | [train-zh-pinyin-colloq-c20.toml](../../configs/train-zh-pinyin-colloq-c20.toml) |
| `colloq-c10-hard` | 同 c10 | 只用交叉熵 | [train-zh-pinyin-colloq-c10-hard.toml](../../configs/train-zh-pinyin-colloq-c10-hard.toml) |

- `colloq-ctrl` 回答“再训 30,000 步本身带来什么”。r3-ctrl 和 r3-ctrl-s2 是同一配方的另外两个 seed，用作 seed 噪声参考。
- c10 和 c20 中，r1 基础混合按比例缩小，给口语样本腾出位置。
- c10-hard：教师没见过这类文本，其软标签可能把口语拉回书面。所以在这个臂里，两个口语目录设 `ce_only = true`，只用标签的交叉熵；r1 基础部分仍然蒸馏。训练器改动见 `f32a36b`。没有 `ce_only` 来源时，批次与改动前逐字节相同，测试覆盖了这一点。

## 评测

全部在 development 上进行，不读 test 和 blind。

**主指标**：口语整句拼音转写，评测集为 `ime-pinyin-zh-colloq-v1`（[prepare_pinyin_colloq_zh.py](../../scripts/benchmarks/prepare_pinyin_colloq_zh.py)）。

- 记录格式和流程与公开拼音集相同：Vime 搜索（每个按键位置 beam 16），取前 10 个候选，用模型自身分数重排（λ = 1.6），看重排后的 Top-1。
- 上下文只用同一条消息里光标前的部分（infield），看不到前几轮对话。
- 来源与题数（均来自评测专用划分）：

  | 来源 | 题数 | 取自 |
  | --- | ---: | --- |
  | LCCC | 600 | test |
  | r1 口语 | 600 | 留出集 |
  | 豆瓣 | 300 | 留出集，筛查通过 |
  | MMChat | 300 | 留出集，筛查通过 |
  | 贴吧 | 200 | 留出集，筛查通过 |

- 子句 4–20 字，约一半以语气词结尾。读音只保留三种工具一致的。
- 先与所有训练语料做重叠扫描（Aho-Corasick，最短 8 字），命中的题目删去，然后冻结。
- development 和 test 各一半；test 留给确认运行。
- 显著性检验：按句子簇做配对符号翻转检验。

**参考指标**（只描述，不参与决策）：

- 联想 track A Top-3 中 CPED（电视剧对白）那一层单独报告；
- 语气词联想：用口语集里以语气词结尾的题，在语气词前的位置联想，允许单字候选（`min_chars = 1`），报告 Top-1 和 Top-3。

**守门条件**：

1. 公开拼音集 development 自重排 Top-1 不显著低于 `kd-cont`（句子簇符号翻转，p < 0.05）；
2. Standard 全拼重排 Top-1 不显著低于 v1。

联想（track A–D）和纠错只描述，不参与决策。

## 决策

1. 在 c10、c20、c10-hard 中取主指标最高的一臂。它替换 `kd-cont` 需同时满足：
   - 主指标显著高于 `colloq-ctrl`；
   - 两个守门条件都通过。
2. 如果 `colloq-ctrl` 本身显著高于 `kd-cont`，只说明多训有用，不算口语数据的效果。
3. c10-hard 对 c10 回答“口语上要不要教师”。如果 c10-hard 胜出，以后的口语数据都按只用交叉熵处理。
4. 如果没有臂满足条件，`kd-cont` 不变，口语问题回到客户端（传入已上屏文字、允许单字候选）和评测侧处理。
5. 胜出的臂与 `kd-cont` 架构相同，导出和延迟不变；它要按 v2-20m 原有流程走一遍 test 确认和 Mac 对齐。

## 不在本计划内

- 人工出题的口语评测（以后再做）；
- 纠错数据、800M 教师；
- 客户端规则和敏感词拦截。

## 偏离

（暂无。）
