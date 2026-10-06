# AzooKey + Tiny LM 组合排序

现有原型策略已冻结：`AzooKey score + 2 × contextual LM logP sum`。λ=2 由独立审核开发集选择，不在 AJIMEE 上搜索。策略文件为 `artifacts/ranking-policies/tiny-ja-v1-hybrid-dev-v2/policy.json`。

## 评分约定

工具 `scripts/benchmarks/hybrid.py` 负责 prepare-dev / score / tune / evaluate，不调用 API、不训练模型。生成入口见 [开发数据](development-generation.md)，真实候选出口见 [AJIMEE](ajimee-benchmark.md)。

两种分数原样组合，无长度归一化、额外 EOS、候选删除或词语硬编码。λ 必须有限且非负，λ=0 严格恢复原导出顺序；并列保持原顺序。无法完整 LM 计分时整条回退。byte fallback、严格 token 前缀竞争、首选字种变化只记录诊断。

context+candidate 联合分词、从公共 token 前缀后累计 logP；保留边界重分词信息。该 token 后缀似然代理并不精确等同于字符串条件概率。

## 已完成的数据

- 成功导出 180 条生成草稿；18/20 批次，缺失两批未继续补齐。
- Codex 逐条复核保留 137 条（63 有左文、74 无左文），43 隔离；9 条去掉读音不同的替代表记，不改写 input/context。
- 审核未使用 Tiny LM 得分或候选结果筛标签，是 AI 复核，未经过外部母语者裁定。
- `artifacts/benchmarks/ime-dev-reviewed-v2/` 保存决定，`ime-dev-v2/` 保存 Mac 输入与导出。
- 137 条全对齐；136 条 20 候选，1 条 18 候选，无空池，答案覆盖 136/137。

与 AJIMEE 按 NFKC / 去空白检查 query/context、context+答案的明显重叠，拒绝写出重叠集；不保证没有语义近似或训练重叠。

## 复现流程

以下为新实验目录示例，现有结果无需重跑。先复核 JSON 的读音和全部可接受表记；`--labels-reviewed` 是已审查声明，不自动审核，也不等于人类母语者认证。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/benchmarks/hybrid.py prepare-dev --input artifacts/benchmarks/ime-dev-reviewed-v2/evaluation_items.json --labels-reviewed --output artifacts/benchmarks/ime-dev-recheck
```

输出原始 JSON、`ajimee-input.json`、case-map、manifest；开发数据单独标记 role，不继承 AJIMEE 许可。把生成输入交给 Mac 同版本转换器，复制整个 `ajimee-results/` 回新开发目录。

已保存的真实开发候选可直接进行新的只读评分：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/benchmarks/hybrid.py score --benchmark artifacts/benchmarks/ime-dev-v2 --output outputs/ime-eval/dev-recheck-scores
.\.venv\Scripts\python.exe -X utf8 scripts/benchmarks/hybrid.py tune --scores outputs/ime-eval/dev-recheck-scores --output artifacts/ranking-policies/dev-recheck
```

默认固定 grid：`0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3`。选择开发 Top-1 最高；并列时改坏更少；再并列选较小 λ。更改 grid 属于新开发实验，不能在 AJIMEE 上自动 tune。

当前 grid 中 λ=2 与 λ=3 均为 123/137，改坏分别 3 / 4，因此按事先规则选 2。

复用原 AJIMEE 分数缓存，以已冻结策略产生新报告：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/benchmarks/hybrid.py evaluate --policy artifacts/ranking-policies/tiny-ja-v1-hybrid-dev-v2/policy.json --output outputs/ime-eval/ajimee-hybrid-recheck
```

evaluate 核对 checkpoint、tokenizer、FP32 精度、转换器/字典 commit、N-best 和开发/测试身份及明显重叠。切换模型、量化或字典后须核对独立开发集，必要时新版本重新校准；不把现有 λ 直接当已验证部署策略。

手动 `--lambda` 会标记为 user_specified_not_calibrated；用于一次明确诊断，不冒充开发集选定参数。零系数一致性验证已完成，原 200 条排序逐条一致，结果保留在 `outputs/ime-eval/tiny-ja-v1-ajimee-hybrid-zero/`。

## 固定结果和限制

| 数据 | 原排序 Top-1 | 纯 LM | λ=2 | 组合纠正 / 改坏 |
| --- | ---: | ---: | ---: | ---: |
| 开发 137 条 | 111 (81.0%) | 122 (89.1%) | 123 (89.8%) | 15 / 3 |
| AJIMEE 200 条 | 87 (43.5%) | 124 (62.0%) | 118 (59.0%) | 38 / 7 |

组合在 AJIMEE 的 Top-5 为 153/200，纯 LM 为 151/200，原排序为 143/200。组合 Top-1 比纯 LM 低，改坏从 11 减至 7；检索未覆盖的 38 条无法通过排序修复。合成开发集较简单，当前 λ 仅为原型策略，不能断言适用于所有真实输入。

缓存：`outputs/ime-eval/tiny-ja-v1-dev-v2-scores/`；报告：`tiny-ja-v1-dev-v2-hybrid/` 与 `tiny-ja-v1-ajimee-hybrid-dev-v2/`。metrics 保存指纹和分组，scores.jsonl 保存逐候选分数，changed-cases 保存全部首选变化。AJIMEE 已被用于分析，应保持固定对照并另留新测试，不能把反复查看后的改善称为完全盲测。
