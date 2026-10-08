# Standard Japanese IME benchmark

2026-10-08。清洗、分组、重叠筛查、1,500条新来源的逐条AI专家审核、Mac真实候选导入与首次固定模型评分已完成，冻结版本为`ime-standard-ja-v1-ai-expert-r1`。新来源保留1,453条；[首次比较](standard-ime-results-20261008.md)包含V2.1与V3 A的development、blind及旧网页回归。标签等级为`ai_expert_reviewed`，不宣称母语人工gold。代码与协议可以进入Git，受限制的文本、候选和审核文件保存在本地忽略目录。

## 数据组成与角色

| 来源 | Development | Blind | Legacy regression | 用途 |
| --- | ---: | ---: | ---: | --- |
| WRIME v2 | 483 | 475 | — | 真人SNS短文本、口语与日常表达 |
| JMultiWOZ | 245 | 250 | — | 人工任务对话中的用户输入 |
| 旧FineWeb2 Edu | — | — | 500 | 已暴露的旧能力回归诊断 |
| 合计 | 728 | 725 | 500 | 冻结版共1,953条 |

新来源分别报告Top-1/5、MRR、MinCER、候选召回、纠正/改坏和回退。主要汇总为独立来源micro与两个来源等权macro；旧FineWeb单列。跨三来源总分只能作为次要描述，不能替代独立来源结果。来源规模和长度分层是预先固定的实验设计，不声称代表真实键盘流量占比。

WRIME按作者分组，原草案development/blind各30位作者；JMultiWOZ按用户与完整对话分组，草案两组分别包含32/30位用户。一位作者最多20条，每段JMultiWOZ对话最多一条；审核剔除不跨组补样。操作者wizard没有互斥，话题也没有互斥。来源独立采集不等于能证明公开文本从未进入网页训练库。

旧FineWeb来自已经清洗、完成候选导出并反复评分的2000条development，仅抽取500条FineWeb样本；旧1000条blind没有参与。原输入、参考及真实候选保持不变，500条旧参考仍为草稿，不因纳入新协议就成为正式gold。

## 清洗、读音与重叠

固定seed20261008，来源版本与原始行/对话ID保存在manifest和逐条provenance。WRIME v2为35,000条，JMultiWOZ为4,246段对话、61,186个双方发言；本基准只提取JMultiWOZ的USER发言，不引入任务描述、槽位、数据库或其他内部字段。

使用NFC和控制字符清理，保留口语、假名/汉字选择与标点。排除URL、提及账号、HTML和脱敏占位符；从原消息提取连续日语转换区间，给定左文只来自当前消息已经输入的前缀，不拼接此前消息。转换区间为4–40字符，读音为4–64字符，左文最多64字符；保存原消息及精确span。数字、Latin与emoji不作为这版的转换目标，难读音/未知词没有自动补造答案，覆盖限制单独记录。

草案要求UniDic与Sudachi正字法读音一致，且目标独立读音与原区间中的读音相同，边界满足自动词法切分。双词典仍有共同误读、误切，专家审核在此基础上修正或剔除。固定来源配额内按有无左文、读音长度轮转抽样，不读取LM分数、候选命中率或候选难度。规范化消息及相同输入/左文不跨新split重复。

8,000条候选对旧train文本25,185,368行、V3 prepared train 1,563,989行各顺序扫描一次；共筛查约5.42GB文件，排除4条匹配候选。方法为NFKC/去标点后完整消息包含，或首尾12字符锚点提出的相似度≥0.85匹配；检查消息长度至少24字符。短常用表达、锚点被改写的近重复、语义改写与不可访问原网页没有全面排除，不能宣称绝对无污染。V3扫描包括未实际训练的已准备块，排除范围偏保守。没有大文件SHA256。

冻结1,953条已核对ID、CLI映射、来源组隔离和全部参考的V2 tokenizer roundtrip，失败为0；含BOS最长development 56、blind 63、regression 35 tokens。真实候选召回为development 630/728、blind 650/725；两模型的新评分均无窗口或roundtrip回退。兼容性诊断不用于筛选题目。

## 标签与版本冻结

全部1,500条新来源在无LM得分、无候选排名的条件下逐条审阅给定左文、目标与读音：剔除47条，修正50条读音，538条纳入多个自然表记，另修正4条同读音的原文错字。剔除项包括无法确认的姓名/简称、跨词或日期读音的边界以及无法恢复的错字，保留逐条理由，不按模型得分补样。

原建议列表没有直接批量采纳。词内替换可能生成“有りがとう”“と事ん”或把否定应答“いえ”换成“言え”；本版改用词法边界与给定语义约束，记录逐条补充形式。多读音字的明确专家判断可覆盖词典默认读法；可接受表记不声称穷尽，预先设定每题上限256，本版实际最多32个。

审核页面隐藏模型输出及候选排名。确认给定左文下的读音、合理转换边界和可接受表记；无法仅靠给定输入消歧时，增加真实可接受解或隔离样本，不能用原文未提供的右文强行确定唯一答案。原作者表记与其他自然表记都应评估。AI初审和母语人工审核分别记录，不能把AI标记为母语人工审核。

`freeze`要求每条有明确决定、审核者与说明。`--review-mode ai-expert`接受显式AI专家记录，保持`native_japanese_review=false`、`labels_formal_gold=false`，同时记录`labels_frozen=true`与标签等级；`human-native`保留真正母语人工gold入口。隔离项及原因保留。冻结题目生成独立release ID；修正已冻结版本必须另建版本、说明变更，并让所有比较模型使用同一标签版本。

WRIME为CC BY-NC-ND 4.0，派生题库公开发布需确认许可；JMultiWOZ为CC BY-SA 4.0。包内保留原LICENSE、来源与论文归属。本次只准备本地研究与同一所有者的私有跨设备交接，不把文本ZIP提交GitHub。

## 固定评分与盲测复用

使用现有固定AzooKey converter、字典和n-best20，关闭typo与Zenzai。模型仅重排实际候选，不注入参考。主评分为context+candidate联合分词公共前缀后的完整词表logP sum，FP32、不加EOS、不截断、并列保持原候选顺序；这是token后缀评分代理。整条候选出现窗口/roundtrip问题时回退原序，召回失败保留完整分母。量化模型单独报告，不混入FP32主表。

development可反复用于调试和选型；未审核状态只有显式草稿诊断入口。blind要求已审核冻结的标签与预先登记的全部模型、评分规则。首次开启后记录consumption，后续可以作为固定reference test继续比较，但不再称为新盲测，也不能用其分数继续调参后宣称独立验证。需要新的泛化证据时，以相同协议采集新作者/对话并另建盲测版本。

旧回归候选直接复用，评分缓存逐条核对ID、读音、左文、参考与候选内容/次序。500条缓存结果为V2.1 374/500、V3 A 368/500，仅为历史草稿回归诊断，没有重复推理。首次新来源blind已按评分前登记的两模型完成，后续使用属于已暴露reference test。

## 构建与交接入口

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/benchmarks/download_standard_ime.py
.\.venv\Scripts\python.exe -X utf8 scripts/benchmarks/prepare_standard_ime.py
.\.venv\Scripts\python.exe -X utf8 scripts/benchmarks/standard_ime.py review-ui --output handoff/ime-standard-ja-v1-review
.\.venv\Scripts\python.exe -X utf8 scripts/benchmarks/standard_ime.py package --output handoff/ime-standard-ja-v1-mac-draft
```

现有输入复用相同快照，完成产物不重复构建；改变采样参数使用新的output。数据根目录`artifacts/benchmarks/ime-standard-ja-v1-draft/`，原始来源`datasets/ime-standard-ja-v1/`，筛查与核验报告`outputs/benchmarks/ime-standard-ja-v1/`。

当前Mac交接包为`handoff/ime-standard-ja-v1-mac-expert.zip`，包含审核后728 development / 725 blind输入、固定版本脚本与`README_CN.md`。原草案包保留追溯，本轮不再使用。旧FineWeb500条真实候选复用，无需Mac重导。候选收集不进行LM评分：只改已冻结参考时可保留候选、重算参考rank并保存原始导出；读音、左文或数量变化则重新导出。

```bash
bash export_azookey.sh /path/to/AzooKeyKanaKanjiConverter
zip -r azookey-results.zip azookey-results
```

```powershell
# 当前AI专家版已冻结；freeze示例只用于记录构建方法，不覆盖已完成产物。
.\.venv\Scripts\python.exe -X utf8 scripts/benchmarks/standard_ime.py freeze --reviews outputs/benchmarks/ime-standard-ja-v1/expert-review-r1 --output artifacts/benchmarks/ime-standard-ja-v1-ai-expert-r1 --version ime-standard-ja-v1-ai-expert-r1 --review-mode ai-expert --change-note "Pre-score AI expert review"
.\.venv\Scripts\python.exe -X utf8 scripts/benchmarks/standard_ime.py import --prepared artifacts/benchmarks/ime-standard-ja-v1-ai-expert-r1 --archive handoff/ime-standard-ja-v1-azookey-results.zip --output artifacts/benchmarks/ime-standard-ja-v1-candidates

# Development示例；单独选择输出，不覆盖历史成绩。
.\.venv\Scripts\python.exe -X utf8 scripts/benchmarks/evaluate_standard_ime.py --benchmark artifacts/benchmarks/ime-standard-ja-v1-candidates/development --checkpoint artifacts/models/tiny-ja-v3-a-chat05-d320-l6/best.pt --output outputs/ime-eval/standard-v1-v3-a-development

# 评分前登记固定比较模型，再运行blind并传入该plan；本轮plan已完成，不重复创建。
.\.venv\Scripts\python.exe -X utf8 scripts/benchmarks/standard_ime.py plan --benchmark artifacts/benchmarks/ime-standard-ja-v1-candidates --checkpoint artifacts/models/tiny-ja-v2.1-e16k-d320-l6-extend5/best.pt artifacts/models/tiny-ja-v3-a-chat05-d320-l6/best.pt --output outputs/benchmarks/ime-standard-ja-v1/blind-plan.json
```

盲测命令使用`evaluate_standard_ime.py --evaluation-plan ...`；已开启版本的后续固定reference test在新plan使用`--reference-test`，成绩明确注明已暴露。旧回归可传`--cached-scores`复用完全一致的FP32缓存。常规`evaluate_ajimee.py`会拒绝新格式，避免绕过版本/盲测入口；这些约束是工作流程记录，不是对手工访问文件的加密隔离。
