# 中文联想基准 ime-predict-zh-v1

2026-11-11 冻结，版本 `ime-predict-zh-v1-r1`。按[预注册计划](predict-zh-plan.md)构建：四个来源，每个来源 1,200 段对话，development 和 blind 各 600 段。标签就是原文，没有人工标注；前缀补全轨道的读音是自动标注，只保留三种读音工具一致的片段。构建脚本进 Git，题目文本只放在本地被忽略的目录里。

blind 在冻结后保持关闭，只在最终桌面模型的确认计划里打开一次。development 用于计划里的所有判定。

## 来源

| 来源 | 版本 | 类型 | 许可 | 候选对话 | 泄露命中（剔除） | 入选对话 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| CrossWOZ | `df82c9f` | 任务对话，用户发言 | Apache-2.0 | 5,012 | 674（13.4%） | 1,200 |
| RiSAWOZ | `1e2d3be` | 任务对话，用户发言 | MIT | 10,200 | 650（6.4%） | 1,200 |
| KdConv | `653db76` | 知识闲聊，不含知识属性的发言 | Apache-2.0 | 3,500 | 1,563（44.7%） | 1,200 |
| CPED | `1e4b81c` | 电视剧台词，闲聊 | Apache-2.0（仓库许可） | 11,835 | 106（0.9%） | 1,200 |

- CrossWOZ、RiSAWOZ、KdConv 与[重排基准](standard-ime-zh.md)版本相同。进过 `ime-standard-zh-v1` 草稿的对话（每个来源 1,000 段，覆盖它的 development 和 blind）不作为候选。
- CPED 取自 GitHub 仓库 `scutcyr/CPED` 的 `data/CPED/{train,valid,test}_split.csv`，没有经过千言平台。仓库许可是 Apache-2.0，README 没有单独的数据条款，只要求使用时引用论文（arXiv:2205.14727）。台词出自已播出的电视剧，原剧本的版权不在这份许可里，所以文本只在本地使用，不随仓库或任何交接包分发。

### 闲聊来源的选择

按计划的顺序检查，结果如下：

| 候选 | 第 1 条（许可） | 第 2 条（不属于 LCCC，不在训练清单） | 第 3 条（泄露 ≤ 2%） | 结论 |
| --- | --- | --- | --- | --- |
| NaturalConv | 满足（仅限非商业研究，不得再分发） | 满足 | 1,202 / 19,919 = 6.0% | 不用 |
| DuLeMon | 不接受（见[偏离记录](predict-zh-plan.md#偏离记录)第 1 条） | — | 未扫描 | 不用 |
| CPED | 满足 | 满足 | 106 / 11,835 = 0.9% | **选用** |

选择只看了许可文本、来源清单和扫描的汇总数字，没有看任何模型分数。

## 泄露过滤

- 方法同 v1 的[泄露检查](../reports/zh-v1-20261110/ime-benchmark.md#泄露检查)：每条发言只保留汉字，取全部 16 字 n-gram，在训练导出 `export-r1/train`（60 个文件，45,925,410 篇文档，同样只保留汉字）里逐字查找。哈希只用于预筛，命中都经过逐字确认。
- 有任何命中的对话整段剔除，在抽样之前完成。过滤只用了 n-gram 匹配，没有人读 blind 文本。
- KdConv 命中率高。它的对话常引用影视、音乐、旅游的百科知识，这类句子在网页里也常见，可能是主要原因（没有逐条核查）。剔除后剩 1,937 段，够抽 1,200 段。
- **CPED 的扫描灵敏度较低**：台词普遍很短，11,835 段里只有 5,281 段至少有一条发言长到能取出 16 字 n-gram。其余对话没法用这种方法查。

## 题目

抽样规则按计划执行：seed 20261111，每个来源先按 seed 打乱，依次取到 1,200 段可用的对话；再按 `sha256(seed/对话 id)` 排序，交替分到 development 和 blind。实现细节：

- **可用发言**：人类一方的发言（任务型对话只取用户发言；KdConv 不取引用知识属性的发言），长 4 到 120 字，汉字至少占 60%，不含链接、标记或替换字符。
- **左文**：同一段对话里此前的全部发言（双方都算），加上当前发言里已经上屏的部分。
- **轨道 A**：每段对话从全部合格位置中按 seed 取 3 个。位置在两个汉字之间，其后到下一个非汉字字符至少还有 2 个汉字，并且其后 6 个字不落在已标注的实体或槽位值上。真实后文到下一个非汉字字符为止，平均 5.3 字。
- **轨道 B**：轨道 A 抽到的位置所在的发言，每条完整模拟一次。
- **轨道 C**：每段对话 1 题，目标是 2 到 6 字、以标点或发言末尾结束的片段，取法同重排基准的简拼轨道。读音由 pypinyin（整句）、pypinyin（按 jieba 分词）和 g2pM 三者一致时才收；整个小句里有任何一处不一致就换下一个片段。
- 有 4 段 KdConv 和 12 段 CPED 对话因合格位置不足 3 个被跳过，1 段 KdConv 对话因没有可用的补全片段被跳过。

## 规模

| 来源 | 轨道 A（dev / blind） | 轨道 B 发言（dev / blind） | 轨道 C（dev / blind） |
| --- | ---: | ---: | ---: |
| CrossWOZ | 1,800 / 1,800 | 1,523 / 1,507 | 600 / 600 |
| RiSAWOZ | 1,800 / 1,800 | 1,475 / 1,468 | 600 / 600 |
| KdConv | 1,800 / 1,800 | 1,489 / 1,478 | 600 / 600 |
| CPED | 1,800 / 1,800 | 1,526 / 1,519 | 600 / 600 |
| 合计 | 7,200 / 7,200 | 6,013 / 5,972 | 2,400 / 2,400 |

development 和 blind 没有共用的对话。

## 解码与模拟的实现细节

以下细节在任何模型出分之前确定，记入计划的[偏离记录](predict-zh-plan.md#偏离记录)第 2 条。代码在 `src/vimeml/benchmarks/predict_zh.py`，单元测试 `tests/test_predict_zh.py` 用穷举对照检查了解码器（beam 不设上限时，结果和枚举所有 token 序列完全一致）。

**左文怎么喂给模型**

- 每个模型按自己训练时的格式拼接左文。vimeml-tiny-zh 训练时每条发言是一个独立的 `BOS … EOS` 序列，所以此前每条发言各拼成 `BOS 发言 EOS`，当前发言是 `BOS 已上屏部分`。外部模型用换行分隔发言。
- 放不下时从最早的发言整条丢弃，并给解码留出 12 个 token。当前发言本身过长时只保留它的末尾。

**解码**

- **token 回退**：已上屏部分以汉字结尾、且最后一个 token 全是汉字时，把这个 token 退回，要求每个假设先重新拼出它。这样模型可以把最后一个字和后文合成一个 token 来续写，而不被分词边界卡住。退回的字不计入候选。
- beam 宽度 16，每步每个假设最多取 16 个扩展，再整体保留分数最高的 16 个。只允许全是汉字的 token，新生成的字最多 6 个。
- 每个落在 token 边界、新字数在 2 到 6 之间的假设都是候选；同一个字符串有多条路径时取分数最高的一条。分数是各 token 的 FP32 log-prob 之和。
- 按分数取前 5 个互不为前缀关系的候选；分数相同按字符串排序。轨道 A 的置信度是第 1 个候选的分数。

**轨道 C 的按键约束**

- 读音表取自 rime-ice 的 `8105.dict.yaml` 和 `41448.dict.yaml` 单字表（每个字的全部读音，ü 写作 v）。
- 只打了第一个音节：候选的第一个字必须有一个读音等于这个音节，后面的字不限，候选 2 到 6 字。
- 打了每个字的声母：候选字数必须等于按键数，第 i 个字必须有一个读音以第 i 个字母开头（zh、ch、sh 按 z、c、s 处理）。
- 约束在展开时检查，不相容的假设直接丢弃。

**轨道 B 的模拟**

- 输入单位是 jieba 分词（精确模式，关闭 HMM）得到的词。没有点选联想时，用户打完到下一个词尾为止的内容。
- 每个汉字的按键数是它的拼音字母数，按 pypinyin 对每段连续汉字整体标注；非汉字字符各记 1 键。
- 发言开头还没有上屏任何内容，不给联想；之后每次上屏都给。联想的左文是此前的发言加上当前发言已上屏的部分。
- 展示分数不低于阈值的前 3 个候选。其中有当前剩余内容的前缀时，用户点选最长的一个（1 键），点选可以停在词中间；否则按上一条继续输入。
- 阈值网格是负无穷，加上该模型轨道 A 第 1 候选分数的 5%、10%……95% 分位数，共 20 个；在 development 上取省键率最高的一个，并列时取较高的阈值。
- 省键率按全部发言的按键总数计算（不是逐条平均），同时报告展示次数、点选次数和点选率。

**指标**

- 轨道 A：命中候选的平均字数，指 Top-3 内排名最高的命中候选的字数，在有 Top-3 命中的题目上平均。
- 各轨道的 Top-k 同时报告按题目的总体命中率（micro）和四个来源的等权平均（macro）。主指标轨道 A Top-3 以 micro 为准。

## 评测入口

```bash
python -X utf8 scripts/benchmarks/predict_ime_zh.py run --out outputs/benchmarks/ime-predict-zh-v1/runs/<名字>
python -X utf8 scripts/benchmarks/predict_ime_zh.py score --run outputs/benchmarks/ime-predict-zh-v1/runs/<名字>
```

`run` 先核对四个冻结文件的 SHA256，只读 development，结果写成每条一行的 jsonl，并在 `meta.json` 里记录模型和分词器的 SHA256、解码参数、rime-ice 版本和读音表的 SHA256、jieba 和 pypinyin 的版本。

## 文件

冻结版本在 `outputs/benchmarks/ime-predict-zh-v1/frozen-r1/`（本地，被忽略）：

| 文件 | SHA256 |
| --- | --- |
| `items-a.jsonl` | `493fc4702b9dc37b1f645fcfdcc3efa515f840adcef7e1423c3d233b21eea87f` |
| `messages-b.jsonl` | `2640d6b573fb2e45996a7b6556a2aeafe9af3e42e96603588947ba75fb7a1c71` |
| `items-c.jsonl` | `dc88bc5f0a7e1ed2e3c0ee2e9ae7841ca2712e13a34151e293e2b205df21fb2e` |
| `manifest.json` | `002dc9a39113e1831b5d2352187a079e15705770310dc81de6e06a87e68f6415` |

`manifest.json` 记录了 seed、来源、泄露扫描的汇总、构建统计、各文件的条数和 SHA256。从同样的原始数据重新构建一次，三个题目文件逐字节一致。

## 构建入口

```bash
# 泄露扫描：四个原有候选来源一次，CPED 单独一次
python -I -X utf8 scripts/benchmarks/prepare_predict_ime_zh.py scan outputs/benchmarks/ime-predict-zh-v1/leak-scan-r1
python -I -X utf8 scripts/benchmarks/prepare_predict_ime_zh.py scan outputs/benchmarks/ime-predict-zh-v1/leak-scan-r2-cped --sources cped
# 构建
python -I -X utf8 scripts/benchmarks/prepare_predict_ime_zh.py build outputs/benchmarks/ime-predict-zh-v1/frozen-r1 \
  --scan outputs/benchmarks/ime-predict-zh-v1/leak-scan-r1 outputs/benchmarks/ime-predict-zh-v1/leak-scan-r2-cped \
  --sources crosswoz risawoz kdconv cped
```

原始数据放在 `datasets/ime-standard-zh-v1/raw/`（前三个来源）和 `datasets/ime-predict-zh-v1/raw/CPED/`。第一次扫描的四个来源里包括 NaturalConv；它没有入选，构建时不读取。
