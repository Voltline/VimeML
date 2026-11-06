# Standard Chinese IME benchmark

2026-11-06。抽样、逐条 AI 专家审核、人工抽查和 Rime 真实候选都已完成，冻结版本为 `ime-standard-zh-v1-r1`：development 1,456 条，blind 1,445 条。全拼是主轨道；简拼是单独轨道，只收 2–6 字的目标。标签等级为 `ai_expert_reviewed`，不宣称母语人工 gold。脚本和协议进 Git，题目文本、候选和审核记录只放在本地被忽略的目录里。

blind 还没有开启。第一次使用前，必须先提交一份预注册计划，写明比较的模型和评分规则。

## 数据组成与角色

| 来源 | 类型 | 许可 | Development | Blind |
| --- | --- | --- | ---: | ---: |
| CrossWOZ (`df82c9f`) | 任务对话，用户发言 | Apache-2.0 | 478 | 479 |
| RiSAWOZ (`1e2d3be`) | 任务对话，用户发言 | MIT | 493 | 483 |
| KdConv (`653db76`) | 知识闲聊，不含知识属性的发言 | Apache-2.0 | 485 | 483 |
| 合计 | | | 1,456 | 1,445 |

每个来源每个 split 都按目标长度（2–3、4–6、7–12 字）分层，再分别报告。主汇总是各来源独立的 micro，加上三个来源等权的 macro。来源配额和长度分层是预先固定的实验设计，不代表真实键盘流量的比例。

三个来源都是公开数据集，随 GitHub 仓库发布；它们也可能被网页抓取后进入训练语料。重叠筛查见下文。

## 抽样与读音

- 固定 seed 为 20261106，每个来源抽 1,000 条草稿。每段对话最多抽一条，对话在来源内部按 seed 打乱后分成 development 和 blind。
- 给定左文只取当前消息里目标之前的部分（最多 64 字），不拼接之前的消息。
- 目标是一个小句的结尾：一段连续汉字，起点在 jieba 词边界，终点是标点或消息末尾，长 2–12 字。
- 目标不覆盖任何已标注的实体或槽位值，包括 CrossWOZ 的 dialog_act/user_state、RiSAWOZ 的 user_actions/belief_state，以及 KdConv 的实体名和短属性值。
- 目标不以依附左侧的助词开头（的、了、呢……）。同一组（左文，目标）只出现一次；每个来源里，同一目标最多出现 3 次。
- 读音由 pypinyin 按整句给出。与 g2pM 不一致的字记在 `reading_conflicts`，交给审核。
- 键位：全拼的 ü 写作 `v`；简拼取每个音节的首字母，只收 2–6 字的目标。
- 抽样过程不读取任何 LM 分数、候选列表或候选难度。

## 审核与冻结

**AI 专家审核**
- 3,000 条草稿全部逐条审过，审核时看不到候选，也看不到模型分数。
- 审核看的是给定左文、目标、后文几个字和读音。记录在 `expert-review-r1/specs/chunk-*.txt` 和 `chunk-*.jsonl`，每条一行理由。
- 删除的情况：
  - 目标里有没标注的专名，包括分店名、店名、景点名、人名、作品名、地标和街道名；
  - 目标边界切开了一个词，例如“无烟房”“它家”“看会（儿）”；
  - 目标或原文有错字，且没法恢复，例如“周边店餐馆”“这首菜”。
- 读音修正：表示时长的“长”读 chang，表示“必须”的“得”读 dei，“查查”读 cha，等等。
- 可接受写法只收同音、同长度的写法：
  - 他家/它家（指店家），以及左文里没有指代对象时的他/她；
  - 其他/其它，补语“的”/“得”，预定/预订，订/定，问句里的那个/哪个。
- 这些可接受写法不声称穷尽。本版每题最多 3 个。

**人工抽查**
- AI 拿不准的问题交给了项目负责人，通过本地网页逐条决定。决定记录在 `expert-review-r1/human-review-r1.json`。
- “谁”统一读 shei，共 50 条。
- 城市和地区名（苏州、台湾、通州、姑苏区、崇文门、东直门内大街）不再一律删除，按条决定：34 条里保留 26 条。
- 6 条单独拿不准的条目里，删除 2 条。
- 人工抽查不等于母语全量复核。

**冻结**
- `freeze` 有四项前置要求：
  - 每条草稿恰好有一条审核记录；
  - 不能留下 `unsure`；
  - 人工备注必须先并进审核记录；
  - 输出目录必须是新的。
- 结果：保留 2,864 条，修正读音 37 条，删除 99 条，其中人工覆盖了 28 条。
- 被删的题不补样。
- 有 87 条的读音和草稿不同，包括“谁”的统一。
- 136 条带可接受写法。
- 简拼轨道 1,954 条。
- manifest 记录了以下标签字段：
  - `ai_expert_reviewed=true`
  - `human_spot_review=true`
  - `native_chinese_review=false`
  - `labels_formal_gold=false`
- 已冻结的版本不再修改。要修正就另建版本、说明改了什么，并让所有比较的模型使用同一个标签版本。

## 候选池

**引擎**
- librime 1.17.0（Windows 预编译包 `33e7814`）加 rime-ice（`99662ba`），由 `rime_candidates.py` 通过 ctypes 调用。
- 为了让引擎固定、只依赖输入，做了三处修改：
  - 关掉用户词典，前一次查询不会影响后一次的排序；
  - 去掉 Lua 的日期、时间、农历、UUID、数字、计算器翻译器和英文翻译器；
  - 关掉 emoji 和繁体转换。
- 其余配置不变，包括 rime-ice 词典、含简拼的拼写运算和重排候选的 Lua 过滤器。

**候选池构成**
- 先取 Rime 对整串输入的候选，只保留全是汉字、且字数等于目标长度的，按 Rime 顺序排列。
- 不足 20 个时，用 `rime_lattice.py` 在 rime-ice 词典上做 k-best 补足。补进来的候选标记 `origin=lattice`。
- 候选池里不会注入参考答案。

**参考答案的命中情况**（命中指目标本身或任一可接受写法出现在池中）

| 轨道 | split | 条数 | Rime 第一候选命中 | 仅 Rime 候选命中 | 候选池（20）命中 |
| --- | --- | ---: | ---: | ---: | ---: |
| 全拼 | development | 1,456 | 1,001 (.688) | 1,043 (.716) | 1,420 (.975) |
| 全拼 | blind | 1,445 | 1,045 (.723) | 1,065 (.737) | 1,407 (.974) |
| 简拼 | development | 990 | 142 (.143) | 338 (.341) | 403 (.407) |
| 简拼 | blind | 964 | 147 (.152) | 321 (.333) | 401 (.416) |

简拼本身歧义很大，20 个候选的池子只能覆盖约四成参考答案。没命中的题仍计入分母，所以简拼轨道 Top-1 的上限约为 .41。这一轨道只和自身比较，不并入全拼汇总。

Rime 第一候选命中这一列是引擎自身的基线，不是模型分数，不算消耗 blind。

## 重叠筛查

用 `screen_overlap_zh.py` 在 Vime 1.3 训练语料里查题目原文的逐字重叠。扫描范围：

- stage-a-v2 的 21 个非删除分片，包括 keep 和 review 两部分，范围偏保守；
- LCCC v2 的 train 和 review。

每题查两种模式，用 Aho-Corasick 做精确子串匹配：

- 包含目标的整段连续汉字，至少 8 字，对周围标点的差异不敏感；
- 整条消息，至少 8 字。

更短的片段多是常用短语（苏州旅游），不做筛查。这种方法查不出改写或近似重复，因此不能宣称完全无污染。

扫描正在进行，结果和处理方式待补。

## 评分与盲测使用

- 模型只对候选池重排。主评分是给定左文条件下候选的 logP 之和，用 FP32，不加 EOS，不截断。分数并列时保持候选池原顺序。
- 指标按 [IME 指标 v2](metrics-v2.md) 报告。如果某条出现窗口或 roundtrip 问题，这一条回退到候选池原顺序。参考答案不在池中的题保留在分母里。
- development 可以反复用于调试和选型。
- blind 只能在两个前提下使用：标签已经冻结，并且已经提交一份预注册计划，写明全部比较模型和评分规则。
- 第一次开启 blind 后要记录消耗。之后它可以作为固定的 reference test 继续比较，但不能再叫新盲测，也不能用它的分数继续调参，再宣称是独立验证。

## 构建入口

原始来源放在 `datasets/ime-standard-zh-v1/raw/`，引擎在 `datasets/ime-standard-zh-v1/engine/`。产物在 `outputs/benchmarks/ime-standard-zh-v1/` 下：`draft-r1/`、`expert-review-r1/`、`frozen-r1/`、`rime/`。

```powershell
.\.venv\Scripts\python.exe -I -X utf8 scripts/benchmarks/prepare_standard_ime_zh.py extract --out outputs/benchmarks/ime-standard-zh-v1/draft-r1
.\.venv\Scripts\python.exe -I -X utf8 scripts/benchmarks/prepare_standard_ime_zh.py freeze --human outputs/benchmarks/ime-standard-zh-v1/expert-review-r1/human-review-r1.json --out outputs/benchmarks/ime-standard-zh-v1/frozen-r1
.\.venv\Scripts\python.exe -I -X utf8 scripts/benchmarks/rime_candidates.py deploy
.\.venv\Scripts\python.exe -I -X utf8 scripts/benchmarks/rime_candidates.py query <F>/rime-full.jsonl <F>/cand-full.jsonl
.\.venv\Scripts\python.exe -I -X utf8 scripts/benchmarks/rime_lattice.py index
.\.venv\Scripts\python.exe -I -X utf8 scripts/benchmarks/rime_lattice.py pool <F>/rime-full.jsonl <F>/cand-full.jsonl <F>/pool-full.jsonl --track full
.\.venv\Scripts\python.exe -I -X utf8 scripts/benchmarks/screen_overlap_zh.py <F>/items.jsonl <F>/overlap <corpus files>...
```

简拼轨道把 `full` 换成 `abbrev`。已完成的产物不重复构建；参数一改就换新的输出目录。
