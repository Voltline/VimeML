# 评测

所有评测都只读冻结模型，不训练。FP32 结果来自 Windows CPU；Core ML INT8 的复测见 [Core ML 部署](coreml.md#质量)。

## 评分定义

候选重排使用 **contextual logP sum**：

1. 把 `context + candidate` 作为一个字符串整体分词。不能分开 encode 再拼接，因为 SentencePiece 在接缝处可能重新切分。
2. 在 BOS + context 与所有候选序列的公共 token 前缀之后，逐 token 累加完整词表 log-softmax 的值。
3. 不在候选末尾加 EOS。分数相同时保持原顺序。
4. 所需 forward 的输入超过 128 token、分词无法 roundtrip，或候选池为空时，整条退回原顺序（仍计入分母）。

这是 token 后缀似然的近似，并不等于精确的字符串条件概率。`mean`（按 token 数平均）只作为次要诊断。实现：`src/vimeml/training/evaluate_ime.py` 的 `score_candidates`。

## AJIMEE（公开集）

[azooKey/AJIMEE-Bench](https://github.com/azooKey/AJIMEE-Bench) `JWTD_v2/v1`，数据 commit `401666cd`：200 条（100 条有左文 / 100 条无左文），83 条有多个可接受答案。数据 CC-BY-SA 3.0，保存在 `artifacts/benchmarks/ajimee-jwtd-v2-v1/`，没有主动加入训练语料；尚未全面审计网页语料与该公开集的文本重叠。

**候选导出（Mac）**：AzooKeyKanaKanjiConverter `d59a28e4`，主词典 `4d418525`，emoji 词典 `67b82260`，N-best 20，关闭预测、学习、typo 和 Zenzai，不加 `--stable`（该版本会把 score 取整）。

```bash
git clone https://github.com/azooKey/AzooKeyKanaKanjiConverter.git AzooKeyKanaKanjiConverter-ajimee
cd AzooKeyKanaKanjiConverter-ajimee
git checkout --detach d59a28e4c7ca049aef04f29a91eae9677a7753f2
git submodule update --init --recursive --jobs 8
swift build -c release --product CliTool -Xcxx -xobjective-c++
mkdir -p ajimee-results && cp ~/Downloads/ajimee-input.json ajimee-results/
.build/release/CliTool evaluate ajimee-results/ajimee-input.json \
  --config_n_best 20 --config_typo_mode off --output ajimee-results/azookey-candidates.json
```

`ajimee-input.json` 由 `scripts/benchmarks/prepare_ajimee.py` 从官方 `evaluation_items.json` 生成（`input` → query，`context_text` → left_context，右文为空）。导出后把 `ajimee-results/` 整个目录复制回 benchmark 目录。

**评分（Windows）**：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/benchmarks/evaluate_ajimee.py --output outputs/ime-eval/ajimee-recheck
```

指标：Top-1 / Top-5（与任一可接受答案精确匹配，不做宽度归一化）、Recall@20（候选池覆盖率，即重排能达到的上限）、MinCER（沿用官方定义），以及纠正 / 改坏 / 回退数。

| 策略 | Top-1 | Top-5 | MinCER | 纠正 / 改坏 |
| --- | ---: | ---: | ---: | ---: |
| AzooKey 原序 | 87 (43.5%) | 143 | 0.0822 | — |
| LM contextual sum | **124 (62.0%)** | 151 | 0.0566 | 48 / 11 |
| LM 无左文 sum | 116 (58.0%) | 149 | 0.0716 | 44 / 15 |
| LM contextual mean | 76 (38.0%) | 144 | 0.1197 | 31 / 42 |
| 原分数 + 2 × LM sum | 118 (59.0%) | 153 | — | 38 / 7 |

候选池覆盖 162/200，没有空池或长度回退。分组看：有左文 40 → 59，无左文 47 → 65。结果在 `outputs/ime-eval/tiny-ja-v1-ajimee/`（`changed-cases.json` 列出所有首选变化，`regressions-review.md` 复核了 11 条改坏）。

AJIMEE 已经用于错误分析，不再算盲测。官方左文可能跨句，而 iOS 侧只取当前句，两者的上下文定义不同。

## 开发集（合成，137 条）

用于选组合系数，与 AJIMEE 分开。

1. **生成**：`scripts/benchmarks/generate_development.py` 调用 DeepSeek，按 10 个主题生成「左文 + 片假名读音 + 可接受表记」；另起一个盲核验请求，只给左文和读音。key 通过 `SJTU_API_KEY` 提供，`--dry-run` 不联网。成功请求会被缓存，可续跑。
2. **导出**：v2 共 20 批，成功 18 批（180 条），用 `export_cached_development.py` 离线导出，不补跑失败批次。
3. **复核**：Codex 逐条复核，保留 137 条（63 条有左文 / 74 条无左文），隔离 43 条；没有用 LM 得分筛选。这是 AI 复核，不是母语者裁定。
4. **候选**：`hybrid.py prepare-dev` 生成输入，由 Mac 上同版本 AzooKey 导出 N-best 20。136/137 条的答案在候选池内。

目录：`artifacts/benchmarks/ime-dev-generation-v2/`（生成）→ `ime-dev-review-v2/`（快照）→ `ime-dev-reviewed-v2/`（复核决定）→ `ime-dev-v2/`（候选）。

`scripts/benchmarks/generate.py` / `evaluate.py` 是更早的合成诊断工具（短语二选一、受控同音词），不属于当前评测链路。

## 组合排序

`scripts/benchmarks/hybrid.py`：`score`（逐候选算分并缓存）→ `tune`（在开发集上选 λ）→ `evaluate`（用固定策略跑 AJIMEE）。

组合分数 = AzooKey score + λ × LM sum。λ 从固定网格 `0, .05, .1, .2, .3, .5, .75, 1, 1.5, 2, 3` 中选，依次按开发集 Top-1 最高、改坏最少、λ 最小来定。λ=2 与 λ=3 都是 123/137，λ=2 改坏更少，所以选 2。λ=0 能逐条复现原顺序（已验证）。

| 数据 | 原序 | 纯 LM | λ=2 |
| --- | ---: | ---: | ---: |
| 开发 137 | 111 | 122 | 123 |
| AJIMEE 200 | 87 | 124 | 118 |

iOS 部署用的是纯 LM 排序。λ=2 只对 FP32 模型和这一版词典有效，换模型或量化后需要在开发集上重新选。

## 短语联想

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/tools/phrase_demo.py                       # 网页 http://127.0.0.1:8765/
.\.venv\Scripts\python.exe -X utf8 scripts/tools/phrase_demo.py --suite --output outputs/phrase-demo/new-beam-run
.\.venv\Scripts\python.exe -X utf8 scripts/tools/phrase_demo.py --suite --suite-mode sample --output outputs/phrase-demo/new-sample-run
```

- **beam（默认）**：width 8，长度惩罚 alpha 0.7，最多 8 个新 token，最多 5 条。禁止 PAD/UNK/BOS；后缀为空时禁止 EOS；遇到 EOS 或句末标点 `。！？!?` 结束；按可见文本去重。
- **sample**：greedy + 8 条采样，temperature 0.8，top-k 50，top-p 0.9。

iOS 键盘不用句子续写，而是用更短的「下一个词」预测（Vime `VimeLanguageModel.nextWords`）；beam 在 iOS 中保留为与 Mac 对照的参考实现。

固定 20 个前缀在 `configs/phrase-demo-prompts.json`。FP32 beam 的 Codex 复核：10 个前缀有明确自然的建议，3 个较弱，7 个不理想。正式用语、请求和简单动作效果较好；因果句和自由话题容易复述前文、跑题或截断。原始输出与复核在 `outputs/phrase-demo/tiny-ja-v1/`。

## 其他诊断

手写同音词 32 组（`configs/ime-homophones.json`，`evaluate_ime.py`）：有左文 30/32，无左文 12/32。它不经过 AzooKey 候选池，只用来诊断。

## 已知范围

- 训练语料和评测文本之间的语义重叠没有审计。
- 合成开发集的标签和复核都来自 AI，可能存在相同的系统性错误。
- 以上都是离线基准，不代表 Vime 实际使用时的准确率：应用里的候选池（预测、纠错、学习）与基准不同。


## INT8 复测

部署采用纯 LM contextual sum，AzooKey 仍提供假名检索与候选池。INT8 的开发集122/137、AJIMEE124/200与 FP32 的 Top-1 命中数相同，但并非所有分数和顺序相同。没有在量化后重新确认 λ=2；如果恢复融合策略，应在开发集重新选 λ，再对固定 AJIMEE 报告结果。见 [Core ML 质量与限制](coreml.md#质量)。
