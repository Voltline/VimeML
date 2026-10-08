# V3数据实验

2026-10-08。V3 A与[B联合重训](reports/v3-20261008/b-plan.md)均已完成训练、评测和归档。B沿用V2架构随机初始化，旧V2语料＋当前8分片已清洗抽样池＋两套聊天，4轮、有效聊天tokens 5%，实际40.02亿tokens。B按Standard development选定第2轮step132915（507/728），与V2.1相比净增2条；固定参考533/725、旧网页回归364/500，未形成稳定替换收益。结果见[A实验报告](reports/v3-20261008/a-evaluation.md)、[B实验报告](reports/v3-20261008/b-evaluation.md)与[A独立来源IME比较](benchmarks/standard-ime-results-20261008.md)。当前仍保留V2.1部署。

## 目标与固定项

重点新增真人日语文字聊天，扩大日常文体覆盖。A的新训练语料不采用Edu评分筛选，也未主动混入旧Edu语料；新的B联合重训明确加入旧V2语料。完整版网页语料本身仍含教育类文本，是否减轻文体偏向需要任务评测验证。

首轮保持V2的12,537,920参数、context128与16K tokenizer，复用现有部署接口。旧模型、原验证集、IME候选池与标签保持冻结。

## 来源选择

| 来源 | 已核对信息 | 实验角色 |
| --- | --- | --- |
| [RealPersonaChat](https://github.com/nu-dialogue/real-persona-chat) | 公开版13,583对话、408,619发言；真人文字聊天；CC BY-SA 4.0 | 重点新增内容 |
| [MRMP](https://github.com/nu-dialogue/multi-relational-multi-party-chat-corpus) | 公开版960对话、100,680发言；初次见面与家庭关系中的三人文字聊天；CC BY-SA 4.0 | 与RealPersonaChat合用 |
| [JpnMix](https://huggingface.co/datasets/AdaMLLab/JpnMix) | FineWeb2、HPLT2、CulturaX、C4、FinePDFs五个来源；日语规则过滤与跨来源去重；许可依各来源 | 优先评估`minhash_deduped`作为通用主干 |
| [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | 非Edu的`jpn_Jpan` train；已经过基础质量过滤；日语主子集卡片列出约667GB磁盘体积；ODC-By 1.0及原来源条款 | 独立网页基线或JpnMix不足时补充，按预算抽样 |
| [HPLT2 cleaned](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned) | 日语`jpn_Jpan`；主要来自Internet Archive，另含Common Crawl；卡片CC0指数据包装，不覆盖原网页版权；官方已推荐HPLT3 | 第一轮通过JpnMix引入，不再单独叠加 |

两套聊天公开版合计509,299条发言。原始论文收集量与清理后的公开量分别记录；token统计使用冻结的V2 tokenizer。

### 规模估计

2026-10-08读取[Hub子集元数据](https://datasets-server.huggingface.co/size?dataset=AdaMLLab%2FJpnMix)与文件清单：`minhash_deduped`为114,105,526篇文档、990个Parquet文件，下载247,104,321,870 bytes（247.10GB / 230.13GiB），解码后的Arrow逻辑数据量723,870,629,346 bytes（723.87GB），后者不是必需的同时驻留内存。仓库页面约549GB是三个重叠子集合计，不能作为单独MinHash子集大小。

下载实测：RealPersonaChat为9,358,593字符、4,580,047文本tokens；MRMP为1,163,485字符、642,759文本tokens。合计10,522,078字符、31,329,948 UTF-8 bytes、5,222,806文本tokens，不含BOS/EOS，未清洗、去重和划分split。原始JSON包含元数据，文件体积另计。统计见`outputs/data-research/v3-chat-measured-scale.json`。全量JpnMix合并聊天的下载体积仍约247GB，清洗、缓存和token store另占磁盘。

JpnMix没有提供当前16K tokenizer的精确token计数。用旧语料约5.415文本bytes/stored token的比例外推，其全量可能为约1300亿token；Arrow中含ID、source与结构开销，网页文体和实际清洗也会改变比例，仅作为数量级估计，不能用于确定步数。原始统计与估计口径存`outputs/data-research/v3-source-scale.json`。

### 首批下载与采样建议

首批16个Parquet已下载：共4,055,145,819 bytes（4.06GB）、1,981,909篇文档。990个文件大小中位数约256MB。固定数据revision，将按名称排序的文件清单等分16段，每段用seed42随机选一个；详细清单存`outputs/data-research/v3-jpnmix-file-selection.json`。文件位置分层并不保证来源或文体平衡。

本批`source`实测为FineWeb2 568,044篇、CulturaX 514,707篇、HPLT2 499,158篇、C4 400,000篇，均为单来源标签；未见FinePDFs，不能声称此批覆盖了卡片列出的全部五个来源。后续如需PDF文本对照，另行定位并补充，不因卡片来源数自动扩大此次下载。

原始数据统一存`datasets/v3/`，完整来源版本、清单和核对结果存`datasets/v3/download-manifest.json`与`outputs/data-research/v3-download-report.json`。Parquet核对采用预期大小、footer/schema和source列；聊天核对采用ZIP解压与JSON文本统计，无重复全量SHA256。CLI使用`hf download`与`curl`，下载内容均在Git忽略目录。

本次将参与处理的分片减半至8个，原始下载2,037,650,383 bytes、971,513篇文档；其余分片保留。清洗后网页train可用1,351,303,468 prediction pairs，按随机文档优先级抽取349,384篇、499,999,997 prediction pairs。当前没有按来源设置配额，网页train的token占比为C4 28.39%、CulturaX 22.60%、FineWeb2 22.97%、HPLT2 26.05%；来源标签与实测数量均记录，不强制五个来源各占20%。

文档随机抽取覆盖整份文件，不只截取前N行。先保持文档分组并处理跨池重复/评测重叠，再切分train/validation/test；按token预算选完整文档，保留句间关系，训练窗口遵守context128。两套聊天单独存池、全部保留符合规则的发言与完整会话分组，训练权重单独配置，不依原始文件大小合并后均匀采样。采样与划分记录独立token量，重复曝光仅记入训练预算，不算新增语料。

JpnMix的`quality_filtered`未完成去重，`minhash_deduped`保留去重后的多来源文档，`matched`仅保留至少两个来源共同收录的文档。为保留覆盖，优先选择`minhash_deduped`，`matched`仅作后续可选对照。混合五个来源不等于混合五类文体，它仍主要是网页/PDF文本。

JpnMix已经包含FineWeb2和HPLT2；直接叠加原数据会改变来源权重并引入重复。单独补充时必须统计新增内容与跨池重复。相关[论文v2](https://arxiv.org/html/2512.18834v2)的主要训练对照为阿拉伯语、印地语、土耳其语，不能据此认定日语IME已有优势。

## 数据处理与训练安排

1. 两套聊天保留原始发言、标点与会话顺序，仅提取文本；不将人格资料、时间戳、内部ID或脱敏占位符作为自然语言训练目标。短句和不以句号结尾的发言不能沿用网页规则直接删除。
2. 网页使用非Edu质量清洗，保留来源标签；JpnMix抽样覆盖不同文件与来源组合，避免只读取文件开头。不做大模型润色或默认合成扩写。
3. 按完整文档/对话切分train、validation、test，跨池去重；排除与冻结评测文本的可识别重叠。会话邻近发言可构成上下文，但聊天历史与键盘左侧已输入文本不同，发言独立训练也保留。
4. 聊天是主要实验变量，不要求成为全程token多数。先进行广覆盖混合训练，再用短阶段适配提高聊天采样权重；权重与阶段长度根据实际独立token量和累计曝光次数确定，避免小池随大语料反复训练几十轮。
5. A从V2.1 extend5 best step40000加载权重，使用新优化器。原B预备方案仅用新语料随机初始化，未执行；当前B改为旧＋新语料联合重训，架构与tokenizer相同，不再承担纯替换旧预训练语料的对照。

先完成来源统计和可用样本抽查，再冻结采样配置与预算。首选组合为两套真人聊天加JpnMix；完整版FineWeb2保留独立对照，不与JpnMix重复计入。第一轮不另加Wikipedia、Tatoeba或zenz。

## 构建入口

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/corpus/prepare_v3.py
```

默认使用16个已下载分片的偶数位置，共8个，约2.04GB；两套聊天全部进入清洗流程。NFC/控制字符处理、明显网页噪声剔除和宽松日语比例过滤，不做Edu评分、API审核或润色。网页按句界尽量组合成不超过1024字符的段落块，超长单句按字符分块；聊天独立发言保留，不在本次引入跨说话人的上下文拼接。

完整文档或对话按seed42划分98/1/1；精确块不跨split，网页块精确去重，聊天同split内保留重复频次。旧validation和冻结IME的较长参考仅作精确块排除，不声称完成近重复/子串污染审计，不读取blind答案或评分。默认网页train上限5亿prediction pairs（含EOS，不含BOS），先完整编码8个分片，再以文档随机优先级选择，不截取文件前缀；validation/test不受train预算截断。

处理采用SQLite按来源事务提交，重复同一命令自动续跑；参数或输入元数据变化要求另选`--output`，不覆盖冻结V1/V2。构建入口没有API调用、训练启动或全量文件SHA256。

产物根目录`artifacts/token-data/corpus-v3-half/`：`mixed/`为通用与聊天合并池，`chat/`为独立聊天适配池，`indexes/{mixed,chat}/`为context128索引，`progress.json`记录阶段；格式兼容现有V2训练器。`staging.sqlite`用于续跑与完整文档抽样，暂保留。`--web-train-tokens`可调整通用预算，`--parquets 16`可启用全部已下载分片；训练配比仍单独确定。

参数变化时同时指定新的`--output`，默认输出只能复用相同参数与输入。同一命令续跑会复用已完成的来源、token store和索引。

### 实际构建结果

以下prediction pairs均为文本tokens加EOS，不含BOS；两个池的聊天是同一份数据，不计为额外独立语料。

| 池 / split | 文本序列 | Prediction pairs | context128窗口 |
| --- | ---: | ---: | ---: |
| mixed / train | 1,563,989 | 505,598,379 | 4,933,649 |
| mixed / validation | 33,787 | 13,466,346 | 124,100 |
| mixed / test | 35,341 | 14,206,605 | 130,652 |
| chat / train | 496,266 | 5,598,382 | 496,266 |
| chat / validation | 4,986 | 60,530 | 4,986 |
| chat / test | 5,097 | 60,396 | 5,097 |

两池连同索引占3.62GB；续跑暂存数据库另占11.73GB，训练迁移无需携带该数据库。基础mixed train的聊天自然占比约1.11%，正式训练采用下述5%虚拟采样映射。

构建时核对文件大小、offset及预测对数量；完成后以现有加载器检查六个split的首、中、末序列与窗口，并将保存文本重新编码，确认与token IDs一致。只校验必要的小型manifest/index签名，没有全量语料SHA256或回归测试。统计与核对结果存`outputs/data-research/v3-preparation-report.json`。

## 正式训练采样与A配置

分来源抽查24篇文档/对话后，发现上游过滤仍保留商品目录和价格清单。`filter_v3_noise.py`以至少4个价格或6个商品目录标记、同时句末标点稀疏为条件，排除49,205个网页训练块。聊天不变；验证和测试保留原来的广覆盖分布。规则不使用Edu或话题评分，也不声称完成一般质量认证，其他导航/网页噪声仍可能存在。

`prepare_v3_mixture.py`复用已编码文本，每轮将保留的网页窗口使用一次，聊天训练窗口以打乱后的完整循环重复采样，按前缀裁剪后的有效prediction pairs控制5%配额。映射仅保存窗口ID，不复制token文件；validation/test从不进入训练映射。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/corpus/filter_v3_noise.py
.\.venv\Scripts\python.exe -X utf8 scripts/corpus/prepare_v3_mixture.py --noise-filter artifacts/training-data/corpus-v3-noise-filter --output artifacts/training-data/corpus-v3-chat05-filtered-c128
```

| 轮次 | 有效prediction pairs | 聊天prediction pairs | 聊天有效曝光 | batch512更新 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 476,415,525 | 23,820,773 | 4.54次 | 12,574 |
| 2 | 476,435,522 | 23,821,772 | 4.54次 | 12,572 |

A配置`configs/train-v3-a.toml`：从V2.1 extend5 best step40000加载权重，新AdamW；2轮、25,146次更新；LR1e-4余弦降至1e-5，warmup500。保留12,537,920参数、context128、BF16、batch512、前缀裁剪概率0.30与compiled backbone。按每轮平均有效tokens/更新归一化损失，避免短聊天batch因逐batch等权被额外放大；这不意味着AdamW下参数更新贡献可以精确分解为5%。

每轮采样长度略有差别，训练器按冻结映射逐轮计算更新数，恢复时保留epoch和batch cursor。A已核对两轮全部映射的实际有效/裁剪前token覆盖、网页窗口不重复、真实样本裁剪长度和断点重放。原两轮B配置未执行，`configs/train-v3-b.toml`现已改为[4轮旧＋新联合重训](reports/v3-20261008/b-plan.md)。

新机器4090D短测batch512约621k有效tokens/s，峰值分配14,831MiB。只测compiled模式时报告汇总发生错误，测量记录已保留，汇总代码已修复，未重复成功的80次更新。该短测使用逐batch token归一化；正式训练使用上述固定更新尺度。正式100–500步约617k–625k tokens/s，GPU瞬时利用率95%，仅代表开始阶段。原始记录和正式进度位于本地忽略目录`outputs/training-v3/`。

正式运行使用`screen vimeml-v3-a`与W&B online。每轮报告新mixed、独立聊天和冻结旧语料validation，以及原AJIMEE/development；初始旧语料BPC为3.0802681、聊天BPC为4.3924849，两者不能跨文本直接比较。先完成A评测，再决定B，训练期间配置和代码保持冻结。

## 评测与证据

旧validation和新网页/聊天validation分开报告；不同文本上的BPC不直接比较。冻结IME开发池复用候选与标签，另建未参与训练的聊天输入开发样本以观察口语收益；原2000条标签保留草稿说明，1000条blind不用于调参或模型选择。

选择模型依据IME质量、聊天域收益与旧能力变化；仅报告BPC下降不足以证明键盘体验改善。A/B报告实际训练token、聊天曝光次数和GPU耗时；A已有预训练预算单独列明，不声称仅新增预算相同就等于总算力相同。

核验限于来源版本、字段、统计、必要抽样与针对性任务评测。复用已有冻结产物依据，不重复全库SHA256、冒烟或回归测试。
