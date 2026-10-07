# V2.0 最终评测

2026-10-07。V2.0 完成 4 epochs、393,704 updates，用时 4 小时 38 分钟；W&B本地记录 finished。实验起点采用 **best.pt / step 375000**，V1 保留冻结基线。

| 模型 | 完整 validation BPC ↓ | AJIMEE Top-1 | 原 development Top-1 |
| --- | ---: | ---: | ---: |
| V1，1 epoch | 3.4736559 | 124/200 | 122/137 |
| V2.0 best，step 375000 | **3.2598221** | 125/200 | **124/137** |
| V2.0 last，step 393704 | 3.2614271 | **126/200** | 123/137 |

BPC 降低 6.16%，但 tokenizer、结构和训练预算同时改变，不能单独归因于某项。best 按固定子集最低 loss 选择，最终完整 BPC 也低于 last；未穷举全部中间 checkpoint。BPC 为 NLL/(原始 Unicode 字符数×ln 2)，NLL 含 EOS，字符分母不含特殊 token。

## 每轮结果

| Epoch | 完整 BPC | AJIMEE Top-1 | Development Top-1 |
| --- | ---: | ---: | ---: |
| 1 | 3.5058741 | 117/200 | 121/137 |
| 2 | 3.4291191 | 127/200 | 121/137 |
| 3 | 3.3449370 | 126/200 | 122/137 |
| 4 | 3.2614271 | 126/200 | 123/137 |

扫描 100,787,372 个窗口，训练 1,997,721,640 个预测对，crop 移除 228,128,988；窗口覆盖为 4 passes，有效 token passes 为 3.590。纯训练吞吐 121,228 tokens/s，排除评测、保存和初始 warmup。

## IME 结论

统一冻结候选、FP32 IME 推理、joint suffix logP sum，无 EOS/截断，fallback=0。best 的 AJIMEE Top-5 为 158/200，coverage 162/200；development Top-5 为 136/137，coverage 136/137。完整 validation 使用 BF16。

| V1 → V2 best | 改善 | 退化 | 净增 | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: |
| AJIMEE | 14 | 13 | +1 | 1.000 |
| 原 development | 3 | 1 | +2 | .625 |

原开发集四条分歧均涉及表记（頂く/いただく、浸かる/つかる、メガネ/眼鏡、取る/とる）。复核 135 条标签后 V1/V2 都为 **131/135**，见 [标签诊断](label-error-audit.md)。AJIMEE 有实质纠错 `直行座標→直交座標`、`復習→復讐`，也有损坏 `盗難→東南`、`後述→口述`。

新 2,000 条 development 草稿标签下为 72.65% / 73.15%，92 改善、82 退化，p=.495；[扩大集结果](expanded-ime-evaluation.md)仍未证明 V2 稳定优于 V1。原公开文本的训练重叠未知；复核不是独立母语 gold。BPC 第三至第四轮仍下降约 2.50%，但 LM 改善不保证 Top-1 改善。V2 Core ML、真机内存与延迟尚未评测。

## 推理与归档

12 条固定 prompt 的样例包括 `今日はとても良い天気でした。`、`本日はお忙しい中、ご参加いただいた皆様、ありがとうございました。`；自由续写仍有重复。它们是行为观察，不替代候选排序评测。

- 模型与完整结果：`artifacts/models/tiny-ja-v2.0-e16k-d320-l6/`（best/last、summary、full-validation、epoch-evaluation）。
- 最佳模型评分：`outputs/ime-eval/tiny-ja-v2.0-best-{ajimee,development}/`。
- 配对与推理：`outputs/ime-eval/tiny-ja-v2.0-comparison/`、`outputs/model-checks/tiny-ja-v2.0-best/`。
- 迁移归档：`handoff/vimeml-v2-results.tar.gz`；V2.1 参数与性能见 [实验配置](v21-plan.md)。
