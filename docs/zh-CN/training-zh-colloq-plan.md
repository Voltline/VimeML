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
- 这个 store 的 manifest sha256 在构建后、训练前填入配置（`d97cb59d…`）；这一步不算偏离。训练集样本数：纯文本 318,457，全拼 311,428，混拼 310,867，首字母 311,656，纠错 97,667。

**口语部分的构成**（按样本）：r1 聊天约 52%，r1 口语约 10%，豆瓣 20%，MMChat 15%，贴吧 3%。c20 下 r1 store 各 kind 最多读 0.7 遍，聊天 store 纯文本读约 1.13 遍（构建时丢掉不足 8 个 token 的纯文本窗口，聊天短句多），其他 kind 最多 0.97 遍。

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
  | r1 口语 | 600 | 留出集，且不在 zh13 的 P1 池内 |
  | 豆瓣 | 300 | 留出集，筛查通过 |
  | MMChat | 300 | 留出集，筛查通过 |
  | 贴吧 | 200 | 留出集，筛查通过 |

- 子句 4–20 字，约一半以语气词结尾。读音只保留三种工具一致的。
- 20M 和 200M 教师的预训练只用了 P1，r1 拼音数据也从 P1 抽取，所以 r1 的题只取 P1 以外的文档。
- 先与所有训练语料做重叠扫描（Aho-Corasick，最短 8 字），命中的题目删去，然后冻结。r1 的题在扫描时跳过它自己所在的文档，其他文档照常计入。
- development 和 test 各一半；test 留给确认运行。
- 冻结结果（2026-11-22，`frozen-r1`，manifest sha256 `f94a76df…`）：重叠扫描删去 455 题（r1 176、贴吧 117、LCCC 96、MMChat 55、豆瓣 11），主要是贴吧抓取数据里同一楼层在不同帖子里重复、MMChat 和 LCCC 同出自微博，以及 r1 的网页转载。development 770 句（LCCC 252、r1 211、豆瓣 146、MMChat 118、贴吧 43），其中 458 句以语气词结尾、143 句有句内上下文；test 775 句。
- 显著性检验：按句子簇做配对符号翻转检验。

**参考指标**（只描述，不参与决策）：

- 联想 track A Top-3 中 CPED（电视剧对白）那一层单独报告；
- 语气词联想：用口语集里以语气词结尾的题，在语气词前的位置联想，允许单字候选（`min_chars = 1`），报告 Top-1 和 Top-3（[predict_colloq_zh.py](../../scripts/benchmarks/predict_colloq_zh.py)）。

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

## 第一轮结果（2026-11-22）

口语集 development 自重排 Top-1（770 句，4552 个输入），p 为对 `colloq-ctrl` 的句子簇符号翻转检验：

| 臂 | Top-1 | 对 ctrl | p |
| --- | ---: | ---: | ---: |
| `kd-cont` | 39.6 | −0.2 | 0.54 |
| `colloq-ctrl` | 39.8 | — | — |
| `colloq-c10` | 40.9 | +1.2 | 0.0008 |
| `colloq-c20` | 41.9 | +2.1 | 0.00005 |
| `colloq-c10-hard` | 41.1 | +1.3 | 0.0009 |

- c20 对 c10 +0.9（p 0.0044）；c10-hard 对 c10 +0.2（p 0.64），口语上去掉教师没有可测的收益。
- 守门：
  - 公开拼音集自重排 Top-1：`kd-cont` 54.8，ctrl 54.6，c10 55.7，c20 56.1（+1.3，p 0.031），c10-hard 55.8；
  - Standard 对 v1：四臂都通过（c20 净 +5，p 0.48）。
- 参考指标：
  - 语气词联想 Top-1：`kd-cont` 0.190，ctrl 0.194，c10 0.223，c20 0.225，c10-hard 0.229；
  - 联想 track A 的 CPED 层 Top-3：16.4 → 17.5（c20）。
- 按决策第 1 条，c20 胜出，替换 `kd-cont` 成为候选。ctrl 不显著高于 `kd-cont`，所以收益来自口语数据。c20 > c10，说明口语占比还没到头，因此先做第二轮，再做 test 确认。

## 第二轮（修订，2026-11-23 训练前提交）

第一轮的口语数据在 c20 下已接近读一遍，要试更高的占比就得补数据。本节修订“数据”一节里“只用同一个大模型”和“export-r2 不用”两条，只适用于第二轮的新 store；第一轮的 store 和结果不变。

**数据**

- **export-r2 口语段落**（[data-zh-colloq-r2.toml](../../configs/data-zh-colloq-r2.toml)，store `zh-colloq-r2-12k`，manifest sha256 `d23dd329…`）
  - 在 export-r2 的一部分文件上选段：体积 1.4 GB 以上的 54 个文件里每 4 个取 1 个，共 14 个。分类器和阈值与 r1 相同。
  - `p_chat ≥ 0.7` 的 229 万段全部进入候选，另按哈希抽取 43.7 万段小说式对话。
- **第二轮聊天**（[data-zh-colloq-chat2.toml](../../configs/data-zh-colloq-chat2.toml)，store `zh-colloq-chat2-12k`，manifest sha256 `3cc52ba1…`）
  - 豆瓣 part00–05 和 MMChat 的全部训练集都已筛完。
  - 按行数导出：豆瓣 240 万、MMChat 100 万、贴吧 12 万（贴吧筛后 5.1 万条，重复约 2.35 倍），共 352 万条。
  - 它包含第一轮聊天 store 的全部来源，在 c30/c40 里取代 `zh-colloq-chat-12k`。
- **质量筛查**：标准与第一轮相同。
  - r2 段落量大，由两个大模型分头筛：DeepSeek 4.1 Flash 从输入的一端开始，DeepSeek chat 从另一端开始，两端相遇即停。
  - 交界处有 9,430 条被两个模型都筛过，判定一致率 96%。只要有一个模型判不通过就丢弃。
  - 停机时还没筛到的 8,802 条（0.44%）按不通过处理。
  - 聊天只用 DeepSeek 4.1 Flash。
- 筛后：r2 训练集 258.8 万段（聊天 218.8 万、小说式对话 40.0 万），验证集 4.4 万段。
- **评测防重叠**：口语集冻结之后才加入的训练文本，只要包含口语集（development 和 test）任一题的重叠模式，就删掉这条训练行，题目不动。
  - 重叠模式与冻结时的扫描相同：目标所在的汉字串和整条消息，各自不短于 8 字。
  - r2 删了 6 段，聊天训练集删了 2 条 turn。
- 训练样本数：

  | store | 纯文本 | 全拼 | 混拼 | 首字母 | 纠错 |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | `zh-colloq-r2-12k` | 669,244 | 382,468 | 381,816 | 381,170 | 121,505 |
  | `zh-colloq-chat2-12k` | 789,883 | 717,466 | 719,141 | 718,950 | 227,424 |

**训练臂**

四个臂都从 `kd-cont` final 开始，用第一轮的配方，每个 kind 的总数不变。r 类份额在 r1 和 r2 两个 store 间按大小分配，约 64 : 36。

| 臂 | 每次更新的口语样本 | 构成（r1 + r2 + 聊天） | seed | 配置 |
| --- | --- | --- | --- | --- |
| `colloq-c30` | 156，约 30% | 60 + 36 + 60（chat2） | 20261122 | [train-zh-pinyin-colloq-c30.toml](../../configs/train-zh-pinyin-colloq-c30.toml) |
| `colloq-c40` | 208，约 40% | 81 + 47 + 80（chat2） | 20261122 | [train-zh-pinyin-colloq-c40.toml](../../configs/train-zh-pinyin-colloq-c40.toml) |
| `colloq-c20-s2` | 同 c20 | 64 + 0 + 40（第一轮聊天） | 20261123 | [train-zh-pinyin-colloq-c20-s2.toml](../../configs/train-zh-pinyin-colloq-c20-s2.toml) |
| `colloq-c40-s2` | 同 c40 | 同 c40 | 20261123 | [train-zh-pinyin-colloq-c40-s2.toml](../../configs/train-zh-pinyin-colloq-c40-s2.toml) |

- c40 下，各口语 store 各 kind 读得最多的是 chat2 纯文本，约 0.95 遍。
- 与 c20 相比，c30/c40 同时改了口语占比和数据（多了 r2 和第二轮聊天）。所以这一比较回答的是“更多口语数据、更高占比”合起来的效果。
- 口语全部蒸馏，因为第一轮 c10-hard 没有收益。

**评测**：与第一轮相同（主指标、参考指标、两个守门条件），只读 development。

**决策**（比较对象是第一轮的 `colloq-c20`，seed 20261122）：

1. c40 和 c40-s2 都显著高于 c20，且都通过两个守门条件，则 c40 取代 c20 成为候选；
2. 否则，若 c30 显著高于 c20 且通过守门条件，则 c30 取代 c20；
3. 否则 c20 保持候选。
4. c20-s2 对 c20、c40-s2 对 c40 的差作为本轮的 seed 噪声带报告，只描述，不参与决策。
5. 最终候选按 v2-20m 原有流程做 test 确认（口语集 test 一半、公开拼音集）和 Mac 对齐。

## 第二轮结果（2026-11-23）

口语集 development 自重排 Top-1，p 为对第一轮 `colloq-c20` 的句子簇符号翻转检验：

| 臂 | Top-1 | 对 c20 | p |
| --- | ---: | ---: | ---: |
| `colloq-c20`（第一轮） | 41.9 | — | — |
| `colloq-c20-s2` | 41.6 | −0.2 | 0.52 |
| `colloq-c30` | 41.8 | −0.1 | 0.85 |
| `colloq-c40` | 42.3 | +0.4 | 0.34 |
| `colloq-c40-s2` | 42.2 | +0.3 | 0.36 |

表中“对 c20”一列是逐句配对后的差，所以和两边 Top-1 直接相减可能差 0.1。

- seed 噪声带：c20-s2 对 c20 −0.2（p 0.52），c40-s2 对 c40 −0.1（p 0.85）。
- 守门：
  - 公开拼音集自重排 Top-1 对 `kd-cont`（54.8）：c20-s2 55.7（+0.9，p 0.11），c30 56.4（+1.6，p 0.0063），c40 56.3（+1.5，p 0.011），c40-s2 56.2（+1.4，p 0.035），都不低于 `kd-cont`；
  - Standard 对 v1：c20-s2 净 −1，c30 +3，c40 0，c40-s2 +3，都不显著。
- 参考指标：语气词联想 Top-1，c20 0.225，c20-s2 0.238，c30 0.258，c40 0.271，c40-s2 0.260。
- 决策：c40、c40-s2、c30 都没有显著高于 c20，按第 3 条，c20 保持候选。口语占比从 20% 提到 30–40%，并加入 r2 和第二轮聊天，主指标上没有可测的收益；语气词联想有提升，但它只是参考指标。

## 不在本计划内

- 人工出题的口语评测（以后再做）；
- 纠错数据、800M 教师；
- 客户端规则和敏感词拦截。

## 偏离

（暂无。）
