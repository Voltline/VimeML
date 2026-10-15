# V4 首轮训练与本地评估 — 2026-10-15

V4 的通用日语 LM 有改善，但尚未建立稳定的 V2.1 重排替换收益。主 checkpoint 的完整 FP32 BPC 从 3.080191 降到 3.051781，相对降低 0.922%；其 Standard 开发集为 492/728，V2.1 为 505/728。开发集选择的第 8 轮为 503/728。所有结果均未使用 blind。

## 训练与本地归档

固定 8K、d384、7 层、6 头、SwiGLU 960、context 128、15,070,848 参数。最多 12 轮预算在第 9 个完整数据轮次、step 443,115 正常触发 plateau 早停，累计 5,174,759,074 个有效 target；prefix crop 后相当于原始 token store 的 8.0203 次有效 token 暴露。计时更新吞吐 441,043.7 有效 tokens/s，排除验证、保存及开头 warmup 计时。

已下载 62 个文件、1.078 GiB 并逐文件校验 SHA256。完整 `best.pt`（step 372000）、`best-epoch.pt`（第 8 轮 step 393880）、`last.pt`（第 9 轮 step 443115）均保留 optimizer/RNG；另保留第 1–9 轮推理权重、tokenizer、配置、训练报告和日志。三个对应推理包的全部权重张量已与完整 checkpoint 逐位一致核验。推理包不用于恢复训练。

`best.pt` 由训练期固定验证子集 loss 选择；`best-epoch.pt` 由完整 epoch BPC 选择。这两个选择规则不同。完整 best SHA256：`f12fa61230222485339af9609748f5f074f17a134b35d7d6a4310b6c2db35326`。

## 数据与评估边界

本轮 V4 与部署基线 V2.1 使用同一份原始 train/validation 文本，仅 tokenizer 不同。两者 train TXT SHA256 均为 `9c1b4585adaf0518dc0275e7f8086e967db26f9797fb571e1ff481aeadf2ff8f`，包含 24,938,228 条 FineWeb、247,140 条 Tatoeba。V3 新增八分片 JpnMix 没有进入这轮 V4。这个事实不能排除 FineWeb 域偏差或重复训练的影响，但不支持直接把 V3 与 V4 的现象归因于同一批新增数据。

评估前冻结计划、checkpoint 来源 hash、候选快照和选择规则。完整验证是同样的 260,337 句、11,110,489 个原始 Unicode 字符；包含 EOS 预测、不截断、不拼句。V2 为 5,716,082 个 target，V4 为 6,627,568 个 target，均完整覆盖；只比较 BPC，不跨词表比较 token PPL。V2/V4 的 rows.bin 分别指向 JSONL/TXT，故通过原文 SHA 与 source 顺序确认样本一致。

本地设备为 RTX 5060 Laptop，FP32，关闭 TF32。V2.1 Standard 重跑与既有缓存候选/排序/逐 token logP 完全一致，最大分数漂移为 0。重排复用原有实现：联合分词公共前缀后 full-vocabulary logP sum，不加 EOS、不截断、并列保留 AzooKey 顺序、整条超窗回退，召回失败仍保留分母。所有已评估 checkpoint 在 Standard 开发集均无回退。

Standard 开发 728 条为 AI 专家审阅、非母语人工 gold；legacy 500 为历史草稿。Standard 原始导出 manifest 当前本地缺失，因此从既有完整评估记录恢复原候选快照，并逐条核对保留的原始参考、上下文、source/provenance 与 hash；没有重造候选或改标签。AJIMEE 原导出另经既有 loader 校验。历史 137 条也逐条核对原始参考及源 hash。各轨道分开报告，不合并成独立确认集。未读取、评分或据此推断 blind。

## 完整验证 BPC

| Checkpoint | Local full BPC (FP32) |
| --- | ---: |
| V2.1 | 3.080190570 |
| V4 best / 372000 | 3.051781125 |
| V4 epoch 8 / 393880 | 3.057868989 |
| V4 last / 443115 | 3.058944878 |

本地 FP32 与远端训练 BF16 报告分开记录；远端 best 为 3.051885722、last 为 3.058956864。

## 冻结候选重排 Top-1

| Track | AzooKey | V2.1 | V4 best | V4 epoch 8 | Pool recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard dev / 728 | 453 | 505 | 492 | 503 | 630/728 |
| Legacy regression / 500 | 317 | 374 | 372 | 367 | 443/500 |
| AJIMEE / 200 | 87 | 144 | 142 | 144 | 162/200 |
| Historical dev / 137 | 111 | 122 | 121 | 120 | 136/137 |

| Standard source | V2.1 | V4 best | V4 epoch 8 |
| --- | ---: | ---: | ---: |
| wrime | 298/483 | 288/483 | 292/483 |
| jmultiwoz | 207/245 | 204/245 | 211/245 |

开发集相对 AzooKey：V2.1 纠正 100/改坏 48，主 best 纠正 100/改坏 61。第 8 轮虽然更接近开发集基线，也未在其余轨道形成稳定优势。下表相对 V2.1 配对；p 值仅作探索性描述，未做多重比较校正，不宣称统计显著优劣。

| Track | Checkpoint | Improved vs V2.1 | Damaged vs V2.1 | Net | Exact McNemar p |
| --- | --- | ---: | ---: | ---: | ---: |
| Standard dev | V4 best / 372000 | 33 | 46 | -13 | 0.176611 |
| Standard dev | V4 epoch 8 / 393880 | 39 | 41 | -2 | 0.911072 |
| Legacy regression | V4 best / 372000 | 13 | 15 | -2 | 0.850554 |
| Legacy regression | V4 epoch 8 / 393880 | 11 | 18 | -7 | 0.264931 |
| AJIMEE | V4 best / 372000 | 7 | 9 | -2 | 0.803619 |
| AJIMEE | V4 epoch 8 / 393880 | 9 | 9 | +0 | 1.000000 |
| Historical dev | V4 best / 372000 | 1 | 2 | -1 | 1.000000 |
| Historical dev | V4 epoch 8 / 393880 | 2 | 4 | -2 | 0.687500 |

## Epoch 趋势与选择

| Epoch | Step | Remote full BPC (BF16) | Local Standard dev Top-1 (FP32) |
| ---: | ---: | ---: | ---: |
| 1 | 49235 | 3.294657 | 477/728 |
| 2 | 98470 | 3.206939 | 483/728 |
| 3 | 147705 | 3.115083 | 498/728 |
| 4 | 196940 | 3.113201 | 479/728 |
| 5 | 246175 | 3.100866 | 475/728 |
| 6 | 295410 | 3.079426 | 484/728 |
| 7 | 344645 | 3.061917 | 481/728 |
| 8 | 393880 | 3.058166 | 503/728 |
| 9 | 443115 | 3.058957 | 493/728 |

按预先固定的开发集 Top-1 规则，在 best 及 9 个 epoch 中选择第 8 轮；并列优先 best、再选更早 epoch。本轮没有使用其他轨道反选 checkpoint。它与 LM 最优 step 372000 分别报告。BPC 下降没有对应单调的重排收益。原始 4→8→12 轮续训调度不是从开始就规划的 12 轮调度，早停也不证明模型在其他调度下已充分训练。

## 同音词和联想诊断

| Diagnostic | V2.1 | V4 best | V4 epoch 8 |
| --- | ---: | ---: | ---: |
| Contextual homophones / 32 | 31 | 31 | 32 |
| Greedy reference coverage / 14 | 6 | 8 | 5 |
| Sample Top-5 reference coverage / 14 | 9 | 11 | 9 |
| Beam Top-1 reference coverage / 14 | 8 | 9 | 5 |
| Beam Top-5 reference coverage / 14 | 10 | 11 | 11 |

14 个联想样例参考不穷尽，不能当真实联想准确率。沿用 6-token、beam 8/alpha 0.7、8 次固定 seed 采样；另保存 12 个固定 prompt 的 16-token greedy/sample。两个 tokenizer 的字符预算不同。主 best 和 V2.1 各 24 条定性生成结果均保留 prompt 且无 replacement character；具体文本仅保留在本地结果中。

## Tokenizer 诊断

8K/16K 在同一开发集的平均 candidate 长度为 11.482/10.049 tokens，reading 为 20.444/19.346；candidate 字符/token 为 1.555/1.777。两者 UNK 与 roundtrip mismatch 均为 0，整条 128-token 溢出均为 0。8K reading byte-fallback 占 0.383%，16K 为 3.962%。8K 候选 token 膨胀约 14.26%，这可能影响分数偏好，但当前结果不能将其识别为因果；没有据此调整评分规则。8K 目标仍是扩大主干容量。

## 后续实验

独立 V4-B 实验保留相同语料、8K tokenizer、固定模型、seed 42、batch 512、BF16、AdamW 与 prefix crop。随机初始化即采用完整十二轮、590,820 步 cosine，学习率 6e-4 → 6e-5，warmup 1,000；plateau delta 0.01、patience 2 保持固定。配置为 `configs/train-v4-b12cosine.toml`，独立产物为 `artifacts/models/tiny-ja-v4-b12cosine-e8k-d384-l7/`。该实验检验过早退火假设，未加入新数据或任务目标；已完成结果见 [V4-B 对照报告](b12cosine-evaluation.md)。

本地原始结果：[汇总](../../../../outputs/evaluation-v4-20261015/comparison-summary.json)、[冻结计划](../../../../outputs/evaluation-v4-20261015/plan.json)、[下载 hash 清单](../../../../outputs/training-v4-20261015/download-manifest.json)。这些二进制与逐样本文件不纳入 Git。本报告不提供 Apple 部署结论。
