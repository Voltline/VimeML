# V4-C1 首批转换数据（2026-10-15）

首批转换数据已在本地完成制作与格式/来源检查，包含 **2,841,003 条训练、18,956 条验证样例**。数据采用双词典自动标注，属于 silver 标签；正式后训练尚未启动。[英文报告](../data-v4-c1.md)。

## 实际规模

| 项目 | train | validation |
| --- | ---: | ---: |
| 样例 | 2,841,003 | 18,956 |
| OUT＋EOS 监督 token/遍历 | 11,486,846 | 75,686 |
| 平均 reading 字符 | 6.765 | 6.674 |
| 平均表记字符 | 5.186 | 5.104 |
| 平均结构序列 token（完整 READ 视图） | 23.284 | 22.649 |
| 纯假名目标 | 568,200 | 3,791 |
| 固定空 READ 训练视图 | 284,062 | 0 |
| 左文裁剪 | 564 | 2 |
| FineWeb 来源 | 2,814,085 | 18,794 |
| Tatoeba 来源 | 26,918 | 162 |

首批从 300 万训练、2 万验证的选择结果继续做保守质量过滤，未用其他候选回填，因此实际数量低于最初目标；不表示原语料已无更多合格样例。按首轮配置每次更新转换 256 条、最多遍历 12 次，对应 133,176 次更新、137,842,152 个转换监督 token。普通 LM 重放另外计数，12 次转换遍历不是 12 轮原始 LM 语料。

## 固定处理规则与隔离

- 仅使用原 train.txt/validation.txt；tokenizer 仍为原 8K 模型，SHA256 `296a2bdf4f4862be8bda848ed1cd261f0b0fd43064181fad5a513c6ce73081b5`。两份原始 TXT 在交付时再次全文件核对，保持原身份。没有新增语料或读取 test/blind。
- 在母句中按原文偏移做 fugashi/UniDic 与 Sudachi 共同边界、词汇 reading 比对；目标 1–24 表记字、1–48 reading 字，两路均不超过 6 词素。保存原表记，不改成词典原形；助词保留键盘正字法。双路 reading 字段是规范化结果，原始逐词 feature 不声称全部保存。
- 每个母句最多选一条，按该句可用的 reading 长度/左文长度/含汉字桶采样；限制纯假名目标≤20%。这不是全局各桶等量抽样。单字和活用词素片段可保留，但保守排除了紧邻其他汉字的单字名词，避免罕见复合词/人名碎片的共同误读。
- 排除分析器标出的目标专名、OOV、无可靠共同 reading/边界的片段和不支持的目标字符；不声称分析器能识别全部专名。原文自身的错字及剩余多音读法仍可能污染标签。
- 按母句 hash 隔离原 train 与全部原 validation，再排除与冻结转换验证集相同的完整字段三元组或结构化 token 序列。已知非 blind 开发/回归数据只用于完整样例身份去重，不进入训练，不按常见表记单独拉黑。文档 ID 已不可恢复，只能保证原句级及精确任务样例隔离，不能宣称文档或语义完全无重叠。
- 原 validation 母集已经用于基础 LM 选模，本次转换 validation 不是对整个研发流程完全未见的独立测试集。

## 编码、检查与质量边界

`BOS <CTX> context <READ> reading <OUT> surface EOS`，各字段分别编码，结构 ID=4/5/6，最多 129 个序列 token，对应 128 个输入/预测位置。只裁左文，保留完整 READ/OUT。labels 只 shift 一次，只监督 OUT 与 EOS；空 READ 约 10% 的训练视图单独保存标志，原 reading 不丢失。

最初约 300 万条已逐条核查母句/源 ID/字符 span/hash、重新编码、loss mask、空 READ 视图与 split 隔离。最终词汇过滤只删除行，所有保留记录、token 及 mask 标志均原样复制，检查由已通过的父产物继承；再次固定最终文件 SHA256、offset/count 和纯假名上限，没有为删除行重复全部词典标注。Windows DataLoader 可按路径重新打开 mmap，避免序列化整份 token 数组。

Codex AI 抽看 200 条分层样例，发现并隔离了“家”标成「や／け」、岩石棒式的单字名词复合边界、从春也中截取春等模式；过滤会同时舍弃部分正确的复合片段，以换取首批数据的保守性。提供最终 200 条分层复核样本及完整母句。没有母语人工审核，也不从 200 条抽查推算总体准确率。

## 产物

目录：`artifacts/posttrain-data/v4-c1/`（大文件不纳入 Git）。

| 文件 | 用途 |
| --- | --- |
| manifest.json / acceptance.json / delivery.json | 最终身份、继承检查链、交付来源核对 |
| `train.tokens.bin` / `validation.tokens.bin` | uint16 token 数组 |
| *.offsets.bin / *.out-start.bin / *.reading-dropout.bin | uint64 序列 offset、uint8 目标起点/视图标志 |
| *.records.jsonl | 原句 hash、TXT 行/字节位置、sourceID、span、条件字段与规范化读音 |
| *quarantine.jsonl / 历史 manifest 和 acceptance | 排除记录与各次制作的审计依据 |
| stratified-review-200.json | 最终分层复核包，附完整母句 |
| preparation-source / refinement-source | 与记录 SHA256 一致的实际处理源码快照 |

最终 manifest SHA256：`3eb58a9dba365e96f094bfb22950c66a9d06cddeb1cb7f5e4db2e78bf59532c0`。

## 训练配置

本地容量记录与已准备的 AutoDL 训练配置见[后训练设计](posttraining-v4.md)。
