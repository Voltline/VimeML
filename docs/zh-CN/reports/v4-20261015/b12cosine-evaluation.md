# V4-B 完整 12 轮调度实验 — 2026-10-15

本轮完整 12 轮调度进一步改善 LM：本地 FP32 BPC 为 **3.027777**，比 V4-A 的 3.051781 降低 0.787%，比 V2.1 降低 1.702%。LM 最优 checkpoint 的 Standard 开发集为 498/728；按预先固定开发集规则选择 **best、step 584000、498/728**。各轨道结果见下表；LM 收益与真实候选排序分别判断。

本文 A 指首轮 4→8→12 追加预算实验，B 指本轮从开始就设定 12 轮 cosine 的实验，均属于基础 LM 研究。原始数据、8K tokenizer 与模型维度没有改变。

## 训练事实与实验控制

实际完成 12 轮、590,820 步、6,899,499,874 有效 target，未触发早停。Trainer 记录的本次会话 elapsed 为 15,941.1 秒，计时更新吞吐 439,458.1 有效 tokens/s。A 完成 9 轮、5,174,759,074 有效 target；B 实际多训练 **33.33%** 的有效 token。因此这是调度及其实际训练预算的比较，不是等算量消融，不能把全部提升单独归因于学习率曲线。

固定配置：15,070,848 参数、8K/context 128、d384/layers 7/heads 6/SwiGLU 960；seed 42 随机初始化；batch 512、BF16、compiled backbone、相同 AdamW 和 30% prefix crop。B 从开始使用 12 轮 cosine 6e-4→6e-5、warmup 1000；plateau min_delta 0.01/patience 2。Checkpoint 核对确认模型、tokenizer、token/window manifest、共享 trainer/model/data 文件 hash 与 training 设置一致；A 记录的 continuation helper 仅用于预算延长，B 没有使用该 helper。两轮原始 FineWeb/Tatoeba 数据相同，没有加入或删除数据。

完整 best 为 step 584000（按固定验证子集 loss 选择）；last 和 best-epoch 均为第 12 轮 step 590820，权重逐位相同。已下载 45 个文件、1.373 GiB，逐文件 SHA256 校验；三个完整 checkpoint 保留 optimizer/RNG，全部 12 轮推理权重已归档，对应 best/epoch 12 推理包与完整权重逐位一致。完整 best SHA256：`18c49f1d4de5b2852f98c9bc30bf48ca7c84077cfb09a25e3c9002cedf5d1106`。

## 完整验证 BPC

| Checkpoint | Full FP32 BPC |
| --- | --- |
| V2.1 | 3.080190570 |
| V4-A LM best | 3.051781125 |
| V4-A epoch 8 | 3.057868989 |
| V4-B LM best | 3.027777023 |
| V4-B last / epoch 12 | 3.032232369 |

原始验证集仍为 260,337 句、11,110,489 Unicode 字符、6,627,568 个 V4 预测 target（含 EOS），无截断。表中全部为同一本地 RTX 5060 Laptop、PyTorch 2.10、FP32/TF32 关闭口径。远端 BF16 best/last 为 3.028079685/3.032557366，不能混作本地 FP32 值。V2.1 与 A 沿用上一轮完整 FP32 结果，并核对 checkpoint、tokenizer 与输入 hash。

## 冻结候选重排

| Track | V2.1 | V4-A LM best | V4-A epoch 8 | V4-B LM best |
| --- | --- | --- | --- | --- |
| Standard dev / 728 | 505 | 492 | 503 | 498 |
| Legacy / 500 | 374 | 372 | 367 | 377 |
| AJIMEE / 200 | 144 | 142 | 144 | 144 |
| Historical dev / 137 | 122 | 121 | 120 | 120 |

| Source | V2.1 | V4-A LM best | V4-A epoch 8 | V4-B LM best |
| --- | --- | --- | --- | --- |
| wrime | 298/483 | 288/483 | 292/483 | 297/483 |
| jmultiwoz | 207/245 | 204/245 | 211/245 | 201/245 |

沿用相同冻结 728/500/200/137 条输入和 AzooKey 候选，不读 blind、不改标签、候选池或评分规则。主分数仍为联合分词公共前缀后的 full-vocabulary logP sum，不加 EOS/不截断，并列保留原顺序，整条超窗回退，召回失败保留分母。Standard 标签是 AI 审阅而非母语人工 gold；legacy 500 是历史草稿。原始 Standard 导出 manifest 缺失，仍使用上一轮已逐条核对原参考/source/provenance 的历史候选快照，限制没有改变。V2.1 复用与新输入逐条一致、同精度且已复现的分数，receipt 记录源 hash。

以下配对统计均为探索性未校正 p 值，不作为多重比较后显著性声明。B 的开发集选择只使用 Standard contextual-sum Top-1：best 与 12 轮候选一起比较，并列优先 best、再选更早 epoch；其他轨道不反向改变选择。

| V4-B | Track | Comparator | Improved | Damaged | Net | Exact p |
| --- | --- | --- | --- | --- | --- | --- |
| best | Standard dev / 728 | v21 | 42 | 49 | -7 | 0.529602 |
| best | Standard dev / 728 | v4a-best | 38 | 32 | +6 | 0.550413 |
| best | Standard dev / 728 | v4a-epoch-8 | 35 | 40 | -5 | 0.644464 |
| best | Legacy / 500 | v21 | 18 | 15 | +3 | 0.728332 |
| best | Legacy / 500 | v4a-best | 15 | 10 | +5 | 0.424356 |
| best | Legacy / 500 | v4a-epoch-8 | 21 | 11 | +10 | 0.110184 |
| best | AJIMEE / 200 | v21 | 7 | 7 | +0 | 1.000000 |
| best | AJIMEE / 200 | v4a-best | 10 | 8 | +2 | 0.814529 |
| best | AJIMEE / 200 | v4a-epoch-8 | 7 | 7 | +0 | 1.000000 |
| best | Historical dev / 137 | v21 | 2 | 4 | -2 | 0.687500 |
| best | Historical dev / 137 | v4a-best | 2 | 3 | -1 | 1.000000 |
| best | Historical dev / 137 | v4a-epoch-8 | 4 | 4 | +0 | 1.000000 |

## 训练进程与实际预算

| Epoch | A full BF16 BPC | B full BF16 BPC | A dev Top-1 | B dev Top-1 |
| --- | --- | --- | --- | --- |
| 1 | 3.294657 | 3.310575 | 477 | 473 |
| 2 | 3.206939 | 3.270015 | 483 | 478 |
| 3 | 3.115083 | 3.244630 | 498 | 494 |
| 4 | 3.113201 | 3.218378 | 479 | 498 |
| 5 | 3.100866 | 3.196170 | 475 | 481 |
| 6 | 3.079426 | 3.168985 | 484 | 490 |
| 7 | 3.061917 | 3.142492 | 481 | 482 |
| 8 | 3.058166 | 3.116429 | 503 | 498 |
| 9 | 3.058957 | 3.086446 | 493 | 495 |
| 10 | — | 3.057157 | — | 490 |
| 11 | — | 3.037641 | — | 496 |
| 12 | — | 3.032557 | — | 489 |

相同第 9 轮，B 的 BPC 为 3.086446，A 为 3.058957；B 直到第 11–12 轮才超过 A 的最佳 checkpoint。这支持给完整调度足够预算，也说明较低中途 loss 不能直接判定最终效果。BPC 与开发集 Top-1 仍非单调关系。仅有各一次 seed 42 运行，不能据此宣称 8K 或 15M 架构的普遍因果优势。

## 同音词与联想诊断

| Diagnostic | V2.1 | V4-A LM best | V4-A epoch 8 | V4-B LM best |
| --- | --- | --- | --- | --- |
| Homophones / 32 | 31 | 31 | 32 | 32 |
| Greedy coverage / 14 | 6 | 8 | 5 | 9 |
| Sample Top-5 coverage / 14 | 9 | 11 | 9 | 8 |
| Beam Top-1 coverage / 14 | 8 | 9 | 5 | 8 |
| Beam Top-5 coverage / 14 | 10 | 11 | 11 | 11 |

32 例同音词和 14 例联想均为小规模合成诊断。联想参考不穷尽；greedy/sample/beam 可能方向不同，不能合称真实联想准确率。8K/16K 使用相同 token 预算但字符预算不同。另保留 12 个固定 prompt 的 greedy/sample 原始输出，不用它们反调参数。

## 当前结论与保留产物

完整 12 轮方案有明确 LM 收益，适合作为后续任务训练的候选底座；真实 IME 收益必须按上表各轨道判断。Standard 开发集主 best 相对 V2.1 纠正 42 条、改坏 49 条，净-7（描述性 p=0.529602）；未建立稳定替换收益。对照 AzooKey，B 纠正 98/改坏 53，V2.1 为 100/48。在有上下文 407 条中，B 为 272，V2.1 为 275；B 去掉上下文后降到 254，说明上下文有帮助，但没有带来总体领先。后续更值得验证 reading-conditioned 任务对同一候选池的收益。本轮没有证据支持立即追加更多纯 LM 训练。原始 A、B 的 LM 最优和开发集最优 checkpoint 分别保留，后续训练与替换选择明确记录依据。

[Local summary](../../../../outputs/evaluation-v4-b12cosine-20261015/comparison-summary.json) · [Frozen plan](../../../../outputs/evaluation-v4-b12cosine-20261015/plan.json) · [Download hashes](../../../../outputs/training-v4-b12cosine-20261015/download-manifest.json) · [Control audit](../../../../outputs/evaluation-v4-b12cosine-20261015/experimental-controls.json)
