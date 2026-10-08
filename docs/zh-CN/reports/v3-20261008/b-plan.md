# V3 B：旧语料与新增语料联合重训

2026-10-08。北京时间17:01在新AutoDL启动联合重训：沿用V2架构、随机权重初始化，合并旧V2训练池、当前8个JpnMix分片的已清洗抽样池与两套真人聊天。A及V1/V2产物保持冻结。原先仅用新语料跑两轮的B预备配置未执行，已被本方案替代。

训练、评测与模型归档已完成，逐轮结果与模型选择见[B评测](b-evaluation.md)。以下保留启动时冻结的实验设计与运行记录。

## 数据范围

旧V2池包含FineWeb2 Edu与Tatoeba，原train为556,462,657 prediction pairs。新池复用A的8个分片、文档抽样和分组split，原train为505,598,379 prediction pairs，其中网页499,999,997、聊天5,598,382。没有启用剩余8个下载分片，也没有把全部8个原始分片中的未抽样文档追加进来。

两池的tokenizer均为冻结V2 16K词表，不重新清洗、编码或训练tokenizer。来源按实际数据计数，JpnMix已包含FineWeb2/HPLT2/CulturaX/C4，不能把这些再算成额外独立数据集。新池原49,205个价格/目录训练块继续排除。

合并前，对旧、新validation/test与新benchmark原消息做一次训练文本重叠筛查；跨池完全相同的网页块保留旧副本，排除新副本，聊天同split重复频次保留。筛查使用NFKC/空白规范化后的精确文本身份及长文本首尾锚点提出的包含匹配，不做全文件SHA256。不同分句/段落边界、锚点外的重叠和语义改写仍可能遗漏，不宣称已完成完整文档近重复审计。

所有原文档/对话的split归属不变，训练映射只索引train。WRIME和JMultiWOZ不作为训练来源；benchmark原消息仅用于排除重叠，不根据模型答错样本或后验答案修改训练筛选。

## 采样与训练

复用两份token store，以虚拟全局窗口ID联合读取，不复制token文件。每轮保留的非聊天窗口各出现一次，聊天按打乱后的完整循环重复到裁剪后有效tokens的5%。重复曝光计入训练预算，不计为新增独立数据。来源ID映射固定为新池0–5、旧FineWeb6、旧Tatoeba7；前缀裁剪由来源内窗口ID、epoch和seed确定。

配置为`configs/train-v3-b.toml`，输出`artifacts/models/tiny-ja-v3-b-all8-chat05-d320-l6/`：

- V2 TinyGPT，12,537,920参数、d320/l6/context128、RMSNorm/SwiGLU、共享词嵌入；随机初始化，无旧权重或优化器加载。
- 4 epochs，更新数由冻结混合映射逐轮计算；batch512、BF16、compiled backbone、4个worker。
- LR1e-3余弦降至1e-4、warmup2000、AdamW beta .9/.95、weight decay .1、grad clip1。
- Prefix crop概率.30，至少保留8个目标tokens；按每轮平均有效tokens/更新归一化，避免短句batch被等权放大。
- 每5000次更新保存并做固定validation子集评估；每轮完整validation、独立chat/old BPC和IME development评测。
- 连续两轮完整主validation BPC比历史最佳高超过.01则提前停止；不会根据小集偶然波动自动终止。

4轮是首阶段训练预算，是否追加依据完整曲线与任务结果判断。B同时改变初始化和训练数据混合，属于联合重训实验；与A不构成严格单变量或等总算力对照。冻结预算共265,841次更新、4,001,730,586有效tokens，其中聊天200,086,509（5%）；每轮聊天约9.63次有效曝光，四轮累计约38.53次。筛查另排除旧池1,062条、新池16,854条重叠序列及67条跨池重复网页块，详情保存在映射manifest、exclusions与本地准备报告中。

## 评测与运行

训练期间使用新benchmark的728条development，以及原AJIMEE200/dev137；逐轮报告来源分项，保持参考和候选池冻结。训练内入口只允许新格式development，不打开blind。旧500条网页用于后续回归；新725条blind已经在A比较中开启，后续只能注明为固定reference test。不同语料的BPC不直接横比。

正式运行使用`screen vimeml-v3-b`与W&B online，入口`scripts/training/autodl_v3_b.sh`。新服务器为PyTorch 2.7.0/CUDA 12.8，实际识别GPU为RTX 4080 SUPER、报告显存32,760MiB。248个运行文件已部署，运行包2,091,854,528 bytes；只含代码、配置、必要token/index/映射、tokenizer和development评测输入，排除原始大文本、SQLite暂存库、凭据、blind候选和旧权重。

首1,100次正式更新已确认正常，step100到1100约295k–316k有效tokens/s；初始编译约63秒。GPU瞬时利用率73%、显存使用17,484MiB，仅为初段观察，不等同整轮平均。两池窗口长度分布与A不同，不能按GPU型号或A的617k吞吐直接推断本轮耗时。15分钟后台检查已恢复，正常训练保持安静；完成后复用逐轮报告并补齐选定模型的必要同池评测。

构建入口：`scripts/corpus/prepare_v3_b_mixture.py`。四轮全部映射的有效/裁剪前tokens及聊天配额、实际样本裁剪长度和一个spawn worker真实batch已核对通过，未执行本地训练更新。本地产物为`artifacts/training-data/corpus-v3-b-all8-chat05-c128/`，一次必要的真实loader/覆盖核对与迁移清单保存于`outputs/training-v3-b/`。已有来源、token与候选产物直接复用，没有重复token化、全库SHA256、模型评分或额外测试套件。
