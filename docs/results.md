# 冻结基线：tiny-ja-v1

截至 2026-10-06 的本地结果。权重已冻结，不继续训练；Core ML / 4bit / iPhone 性能尚未验证。报告引用的是本地正式版本，不上传语料或模型。

## 模型与训练

vocab16384，context128，d_model256，4heads，4layers，FFN1024，dropout0，pre-LN/GELU/causal SDPA/学习位置 embedding，共享 embedding / LM head，共 7,386,624 参数。

- 原始输入 2,289,346 条来源记录；正式 25,713,003 句。
- train / validation / test 分别 25,185,368 / 260,337 / 267,298 句。
- tokenizer 仅从完整 train 抽 200 万句学词表，三 split 全量 UNK 与 roundtrip 错误均为 0，byte fallback 约 0.30%～0.31%。
- 1 epoch，196,853 updates，25,197,117 训练窗口，573,295,237 个有效预测目标；耗时 3469.8 秒（57分50秒）。
- 固定 validation 子集最优 loss 4.5440；最后全量 5,890,326 目标 loss 4.5415709，PPL 93.8381。best 为最后一步，全量 best/last 一致。
- 本地 RTX5060 Laptop：训练主体约 169,587 有效目标/s；实际 tensor 分配峰值 2427.54MiB，allocator reserved 峰值 10846MiB。reserved 不直接代表物理 GPU 占用，本次未独立监测 Windows 内存回退。

精确训练记录：`artifacts/models/tiny-ja-v1/{summary.json,manifest.json,metrics.jsonl,full-validation.json}`。test 未用于 loss 选择或调参，暂无完整 test LM loss。

## 固定真实候选评测

AJIMEE JWTD_v2/v1，200 条，100 有左文 / 100 无左文，N-best20；Mac 转换器及字典固定见 [AJIMEE 指南](ajimee-benchmark.md)。首轮无空候选或长度回退，候选池答案覆盖 162/200（81%）。

| 策略 | Top-1 | Top-5 | 纠正 / 改坏（相对原排序） |
| --- | ---: | ---: | ---: |
| AzooKey 原顺序 | 87/200 (43.5%) | 143/200 (71.5%) | — |
| LM contextual logP sum | 124/200 (62.0%) | 151/200 (75.5%) | 48 / 11 |
| LM 无左文 logP sum | 116/200 (58.0%) | 149/200 (74.5%) | 见完整报告 |
| 原分数 + 2 × contextual logP | 118/200 (59.0%) | 153/200 (76.5%) | 38 / 7 |

独立合成开发集 137 条，AI 复核后冻结，未按 Tiny LM 得分筛标签。原排序111/137，纯LM122/137，组合123/137。预设 λ grid 中 2 和 3 并列，λ=2 改坏更少而选定；没有在 AJIMEE 上扫描 λ。组合 Top-1 低于纯 LM，但改坏减少；λ=2 仅为原型策略。

正式依据：

- `outputs/ime-eval/tiny-ja-v1-ajimee/`（含 11 条纯 LM 改坏复核）
- `outputs/ime-eval/tiny-ja-v1-dev-v2-scores/` / `tiny-ja-v1-dev-v2-hybrid/`
- `outputs/ime-eval/tiny-ja-v1-ajimee-hybrid-dev-v2/`
- `artifacts/ranking-policies/tiny-ja-v1-hybrid-dev-v2/policy.json`

手写同音诊断为 30/32 contextual Top-1、12/32 无左文，保存在 `outputs/ime-eval/tiny-ja-v1/`。它不是 AzooKey 候选，不能作为实际输入法准确率。

## 开放式联想

20 个固定开发者前缀，各显示最多 5 个短建议。Codex 逐条复核：10 个有明确自然建议、3 个较弱、7 个明显不理想；不是唯一答案命中率，也不是母语者裁定。

致谢、请求和简单动作较好；因果与自由话题容易复述、跑偏或截断。采样改善部分场景，但未稳定解决语义问题。本机 CPU4线程FP32 热推理中位 beam50ms、sample116ms，不能外推 iPhone 性能。

依据在 `outputs/phrase-demo/tiny-ja-v1/{samples.json,review.json,review.md}` 和 `tiny-ja-v1-sample/`；[演示说明](phrase-demo.md)。当前联想保留为实验能力，候选排序为主要部署方向。

## 身份和解释范围

```text
best.pt SHA256:
93b6139aa05031df02038efcaf26788bc9916712919f3d2a4c4f398615cef75b
tokenizer.model SHA256:
d9f1ba1e456ce72dd9c06a62b12e804cf14090c30179e6078cd51d03063b2982
```

公开 AJIMEE 已被用于分析，不是完全盲测；训练语料与评测的重叠、语义近重复尚未全面审计。合成开发标签可能有共同 AI 错误，不能外推所有真实输入。这些都是开发机 FP32 的冻结基线，压缩模型须重新核对效果和设备性能。
