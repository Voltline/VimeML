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
