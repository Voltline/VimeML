# V3 A：新网页与真人聊天续训

2026-10-08。完成两轮训练，选定step25146。聊天域语言建模明显改善，但旧语料与IME主指标下降；保留为领域适应实验产物，当前部署继续使用V2.1 extend5 step40000。

## 配置与运行

从冻结V2.1权重初始化，重新创建AdamW；架构与16K tokenizer不变，12,537,920参数、context128。新网页使用8个JpnMix分片清洗后的抽样池，训练时排除49,205个明显价格/目录块；真人聊天使用RealPersonaChat与MRMP。两份冻结虚拟映射按裁剪后的有效tokens维持网页95%/聊天5%。数据处理范围与限制见[V3实验设计](../../plan_v3.md)。

batch512、BF16、compiled backbone；LR1e-4余弦降至1e-5，warmup500，crop概率0.30。损失按每轮平均有效tokens/更新归一化。共25,146次更新、952,851,047有效tokens，其中聊天47,642,545，实际占比5.0000%；聊天训练池累计约9.08次有效曝光，不计为新增语料。

AutoDL RTX4090D上整轮耗时27.59分钟；排除首100次更新、验证和checkpoint开销的吞吐为614,843有效tokens/s。峰值分配14,831MiB、保留19,232MiB。两轮正常完成，screen已退出；没有启动B。

## 同数据评测

| 指标 | 初始化V2.1 | A第1轮 step12574 | A第2轮 step25146 |
| --- | ---: | ---: | ---: |
| 聊天validation BPC | 4.392485 | 3.035096 | 3.006127 |
| 冻结旧validation BPC | 3.080268 | 3.434576 | 3.425275 |
| 新mixed完整validation BPC | 未单独评测 | 3.090339 | 3.032179 |
| AJIMEE Top-1 / 200 | 144 | 134 | 136 |
| 旧development Top-1 / 137 | 122 | 121 | 123 |
| 扩大development草稿 Top-1 / 2000 | 1487 | 未补测 | 1462 |

BPC使用完整对应validation，训练器BF16评估；不同语料上的BPC不直接横比。IME使用FP32、冻结候选池和原参考，主评分为联合分词公共前缀后的logP sum；不加EOS、不截断、并列保留原候选顺序。原200/137条结果复用第2轮现有逐候选报告，没有重复计分。

选定模型与第2轮、最终checkpoint为同一步。选择规则沿用训练前固定的validation子集最低loss；第2轮完整新mixed BPC也优于第1轮。此选择不表示其IME优于部署基线。

| 相对V2.1的配对比较 | 改善 | 退步 | 净变化 | Exact McNemar双侧p |
| --- | ---: | ---: | ---: | ---: |
| AJIMEE / 200 | 6 | 14 | -8 | 0.1153 |
| 旧development / 137 | 4 | 3 | +1 | 1.0000 |
| 扩大development草稿 / 2000 | 46 | 71 | -25 | 0.0261 |

逐条核对ID、输入、参考与候选顺序一致。2000条仍是未正式审核的草稿标签；其退步是同池诊断信号，p值不将草稿转换为正式gold，也不构成独立blind结论。1000条blind没有进行LM计分，没有改标签或添加后验别名。

12个既有前缀各生成greedy与固定seed采样结果，均保留原前缀，无替换字符。部分日常前缀产生自然续写，部分长续写仍有重复；这些例子仅用于检查推理可用性，不证明聊天质量或设备延迟。

## 结论与后续实验边界

相同聊天validation的BPC下降31.56%，旧validation上升11.20%。结果符合领域适应伴随旧能力遗忘；新网页质量、学习率和数据分布同时变化，当前实验不能把损失单独归因于5%聊天或Edu来源。

暂不替换V2.1部署版本，不继续单纯延长A训练。较低续训学习率与旧语料回放可作为保留口语收益的单变量候选。后续实验已选定[B旧＋新联合重训](b-plan.md)，原先“同新数据、两轮随机初始化”的预备B未执行。新的B不继承V2权重，数据混合与训练预算也不同，不能作为等总算力或单变量A/B对照。

## 本地产物与复用

同日新增[Standard IME独立来源比较](../../benchmarks/standard-ime-results-20261008.md)：新来源blind为V2.1 535/725、V3 A 546/725，旧网页500条为374、368。WRIME在development/blind均改善，JMultiWOZ两组方向不一致。这补充了口语领域收益的证据，前述旧基准下降不能概括为所有场景退化；当前仍保留V2.1部署，尚不足以证明V3 A全面更优。

- 模型与原始两轮报告：`artifacts/models/tiny-ja-v3-a-chat05-d320-l6/`，下载`best.pt`及完整validation/epoch报告；同一步的其他checkpoint副本留在远端。
- 下载归档：`handoff/vimeml-v3-a-results.tar.gz`。
- FP32草稿计分：`outputs/ime-eval/expanded-v3-a-dev-draft/`。
- 配对结果、12条推理示例与完成记录：`outputs/training-v3/{paired-comparison,inference-samples,post-evaluation,stages}.json`。
- 启动时源码与数据映射快照保留在本地handoff，训练期间未改配置；训练结束后推理加载器补充V3 tokenizer manifest兼容，并允许省略checkpoint全文件SHA256。实际推理和2000条计分已覆盖该加载路径。

只核对必要的小型manifest和tokenizer身份，没有重复全语料/checkpoint SHA256、冒烟或回归测试。没有自动关机。
