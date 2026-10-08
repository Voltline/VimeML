# V2 实验设计

目标是在iOS Keyboard Extension可承受的规模内，改善日语候选重排与句内短语联想。AzooKey提供假名检索和候选池，LM提供上下文语言分数。V2采用12.54M参数、context128，最终INT8包14.33MB；完整结果见[V2总结](reports/v2-summary.md)。

## 版本与状态

| 实验 | 范围 | 状态 |
| --- | --- | --- |
| V2.0 | 新 tokenizer、320×6 架构、prefix crop、4 epochs | 已完成训练与离线评测 |
| V2.1 | 低学习率继续2 epochs、编译优化、扩大IME benchmark | restart1完成：BPC3.08118、AJIMEE137/200；2000条草稿1476正确 |
| V2.1 extend5 | 携带AdamW状态，追加3.27轮后停止 | 保留best step40000：BPC3.08027、AJIMEE144/200、2000条草稿1487；[结果](reports/v2-20261007/v21-extend.md) |

配置分别为[tokenizer-v2.toml](../../configs/tokenizer-v2.toml)、[train-v2.toml](../../configs/train-v2.toml)、[train-v21-restart.toml](../../configs/train-v21-restart.toml)和[train-v21-extend.toml](../../configs/train-v21-extend.toml)。原batch256配置[train-v21.toml](../../configs/train-v21.toml)保留。复现方式见[V2训练](training-v2.md)。

## Tokenizer 与数据

- SentencePiece Unigram，vocab 16384，identity normalization，保留空白，byte fallback，coverage=.9995。
- `add_dummy_prefix=false`；train 抽样 200 万句，seed 42；PAD/UNK/BOS/EOS=0/1/2/3。
- 复用冻结 FineWeb2 Japanese + Tatoeba corpus 与分组 split。
- 序列为 BOS + 完整句子 + EOS；长句切为独立 context-128 窗口，预测目标无重叠、无遗漏。
- Train 首窗口以 30% 概率移除前缀，至少保留 8 个预测对；后续窗口、validation、test 不裁剪。
- 裁剪由 seed、epoch、sample index 确定，epoch 标签随任务发送给 worker，断点以已提交 batch cursor 恢复。

关闭 dummy prefix 不消除接缝重新切分，候选始终联合编码。

## 模型与训练

TinyGPTV2：12,537,920 参数，d_model=320、6 layers、5 heads、head_dim=64、d_ff=832、dropout=0。Pre-Norm RMSNorm（FP32 方差，eps=1e-5），causal SDPA、SwiGLU、无 Linear bias、learned position embedding、共享 LM head。

训练使用 BF16、AdamW β=.9/.95、weight decay=.1（矩阵参数）、clip=1、长度 bucket。V2.0：batch256，4 epochs，LR 1e-3 → 1e-4，warmup 1000；V2.1 restart1：batch512，从 V2.0 best 权重起新 AdamW，2 epochs / 98,426 updates，LR 3e-4 → 3e-5，warmup 500，epoch offset=4。batch选择依据见 [真实数据性能对照](reports/v2-20261007/v21-performance.md)。

每5,000 updates保存checkpoint并做固定子集验证，每轮完整BPC/IME。退化停止规则及其启用范围以各版本配置为准；extend5配置为连续两轮BPC比历史最佳高超过.01时停止。Checkpoint保存模型、优化器、调度器、RNG、epoch/cursor、训练与裁剪计数、配置和来源身份。

代码入口：`model.py` 为 V1；`model_v2.py`、`data_v2.py`、`runtime_v2.py` 为 V2；`model_factory.py` 统一训练与推理的架构选择。正式训练使用 AutoDL、screen、W&B online，版本目录独立。

## 固定评分语义

1. `context + candidate` 联合 SentencePiece 分词，加 BOS。
2. 找候选池公共 token 前缀，累加其后的 full-vocabulary logP。
3. 主分数为 sum；mean 仅作诊断；末尾不加 EOS，并列保留 AzooKey 原顺序。
4. 超长或分词回退以整条样本回退引擎原序，不截断，不删掉召回失败样本。

语言建模比较使用 `BPC = total NLL / (Unicode characters × ln 2)`。NLL 含 EOS，字符分母不含特殊 token。不同 tokenizer 的 token loss/PPL 不直接横向比较。

## IME benchmark

原 AJIMEE 200 条与 development 137 条保留回归口径。扩大集目标为 2,000 development + 1,000 blind：来源句子/上下文家族分组、跨 split 去重，参考标签独立于模型得分，候选由固定版本 AzooKey n-best20 导出。

统一候选池报告 Top-1、Top-5、Recall@20、corrected/damaged、MRR、CER 与 exact McNemar；完整分母保留召回失败，covered subset 单独报告。开发集用于选模型，blind 在最终选择后计分一次。当前 3,000 条候选齐全，正式标签审核和语义类别分层未完成；词典代理诊断不作为正式 gold。

V1/V2.0 checkpoint、参考标签和评分产物冻结，模型间使用同池、同标签和相同评分政策比较。

## 部署选择与后续变量

部署选择extend5 step40000的INT8 block32版本，CPU_ONLY、FP32计算、最低iOS18。严格logits对齐失败、扩大草稿下降0.3个百分点与UI长尾在当前接受范围内；V2系列实验完成。量化任务质量和真机成本独立于训练BPC评估，详见[Core ML](coreml.md)。

后续变量包括更小词表、结构/学习率与有效 batch 对照、subword regularization、领域数据、蒸馏和 ranking loss；每次实验保持候选与评分政策可比较。RoPE/KV cache 由上下文和设备需求决定。
