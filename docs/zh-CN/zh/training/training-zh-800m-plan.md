# zh-800m 中文桌面模型：训练计划（预注册）

2026-11-15 起草。这是[桌面模型路线](../benchmarks/predict-zh-plan.md)的 800M 训练计划。路线图的门槛已经满足（2026-11-14 判定），第 0 步摸底也没有推迟 800M。本文登记形状、词表、数据、配方和评估。提交即算注册，之后的修改记入文末的偏离记录。

## 要回答的问题

1. **规模收益**：从 200M 加到 800M，联想指标还涨多少？这样就有 20M、200M、800M 三个点。
2. **蒸馏老师**：800M 和 20M 用同一个 12K 分词器，能不能当 vimeml-tiny-zh-v1 系列做 token 级蒸馏的老师？

只训一臂，不做词表或上下文对比，因为 200M 上这两组对比都不有效。

## 形状

| 臂 | 词表 | 上下文 | d_model | 头 | SwiGLU d_ff | 层数 | 参数 | seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `12k-c512` | 12,288 | 512 | 1536 | 24 × 64 | 4096 | 27 | 784,157,184 | 1816189249 |

- 结构仍属 tiny_gpt 家族：RMSNorm，无 bias，embedding 与输出层共享，learned position。每层 28,314,624 个参数，层数取最接近 800M 的整数。头维度保持 64，与 200M 相同。模型在 `src/vimeml/training/model_zh800m.py`，带 FP32 KV 缓存路径；KV 缓存是所有新模型的要求。
- **词表用 12K（vimeml-tiny-zh-v1 的分词器），不用 200M 判定的默认值 24K。** 200M 上 16K/24K/32K 之间没有有效差异。按 2026-11-13 的决定，所有拼音生成模型统一用 12K，可以对 20M 做 token 级蒸馏，也能直接复用 12K 拼音数据、`zh_search` 和 Vime 的链路。代价是同样的文本 token 数比 24K 多约 10%。
- **上下文 512。** 窗口按段落切分，平均只有约 34–42 个 token；200M 上 c128 对 c512 的差异也不有效。增大到 1024 几乎没有样本能用上。
- seed 在写本文时用 `secrets.randbelow(2**31)` 一次抽出。

## 数据

一遍，不重复。网页由三个 12K token store 拼接：

| store | 内容 | 来源 |
| --- | --- | --- |
| `zh13-web-12k` | export-r1 的 P1（212.8 亿字），含分层与记忆探针 | vimeml-tiny-zh-v1 现成，不重编码 |
| `zh800m-web-r1rest-12k` | export-r1 中 P1 以外的 train 文档（167.7 亿字），同一份 pools | 新编码，选择 `rest` |
| `zh800m-web-r2-12k` | export-r2 全部 train 文档（约 884 亿字） | 新编码，选择 `all` |

- 网页合计约 1,265 亿字（r1 全部 380.5 亿 + r2 884 亿）。12K 下每个内容 token 约 1.40 字，约 900 亿个内容 token，约每参数 115 个。
- **聊天：LCCC（`zh13-chat-12k`）占监督 token 的 2%，不是 20M/200M 的 5%。** 按 5% 算，LCCC 要重复约 24 遍，对 800M 的记忆风险太大；2% 约 10 遍，绝对量仍是 200M 的两倍以上。取法与 v1 相同：整个训练是一串遍历，第 k 遍是一次新的随机排列，用第 k 套裁剪种子，共备 12 套。
- 每个网页 store 用各自的裁剪种子（`seed + k·常数`）。混合映射存为 uint32，一个 epoch 一个文件，SHA256 钉入配置。
- 验证集用 `zh13-web-12k` 的 validation（export-r1）和 LCCC validation，与 20M、200M 直接可比；记忆探针仍是 P1 中 tier 0 的 1 万篇文档。
- 编码格式不变（`vimeml_sentence_tokens_v1`），`encode_zh13.py` 新增 `rest`/`all` 两种选择。这两种 store 只写网页、不写 `train.tiers.bin`，manifest 中带 `selection` 字段。

## 配方

- **每次更新 2048 个窗口**（约 8 万个监督 token，是 200M 的 4 倍）。如果仍按 512 个窗口，800 多亿 token 要更新约 480 万次，批次对 800M 太小，数据并行也分不开。预计约 100 万次更新。
- cosine 3e-4 → 3e-5，warmup 2000；wd 0.1，β 0.9/0.95，clip 1.0，prefix crop 0.30，BF16 + compile。学习率因模型变大、批次变大而从 6e-4 降到 3e-4。
- `lm_microbatch` 暂定 64。GPU 冒烟后可以调整，只改微批不改批次，记入偏离。
- 数据并行：`VIMEML_NPROC` 大于 1 时用 torchrun 启动，全局批和数据顺序与单卡相同（见 [200M 计划](training-zh-200m-plan.md)的数据并行一节）。续训时可以换进程数。
- 20 个测点做 FP32 全量网页、聊天验证和记忆探针，保留 1/20 和 1/5 两个检查点，另外每 2000 步保存 `last.pt`。最终评估只用最终 checkpoint。
- 算力估算：约 6·N·T ≈ 4.4×10²⁰ FLOPs，约是一个 200M 臂的 30 倍。

## 评估（规则在此冻结）

全部在 development 上，blind 不读。

- **联想**（路线图口径）：`ime-predict-zh-v1-r1` 轨道 A Top-3 为主指标，与 v1 和 200M 默认臂 `24k-c512-s1` 做配对簇检验；轨道 B/C/D 只作描述。
- **重排守门**：`ime-standard-zh-v1-r1` development 全拼 Top-1。
- **BPC 与记忆**：网页、聊天验证集 BPC，记忆探针，以及联想 Top-1 逐字出现在训练数据里的比例。
- **拼音生成不在本计划内。** 本次训练只做 LM 预训练。拼音能力用后训练补上（沿用 r2-cont 的做法），或者改成联合拼音预训练；二者由 12K joint-pinyin 200M 的评估结果决定。改成联合预训练要在开训前作为偏离记录，配置另加 `[pinyin]` 一节。

## 运行方式

```bash
# token 数据（集群节点；--root 可写到节点本地盘再拷回）
python -I -X utf8 scripts/zh/tokenizer/encode_zh800m.py encode --part r1rest
python -I -X utf8 scripts/zh/tokenizer/encode_zh800m.py encode --part r2
python -I -X utf8 scripts/zh/tokenizer/encode_zh800m.py index          # zh13-web-12k / zh13-chat-12k 的 c512 索引
# 混合映射；把输出的 dataset_sha256 钉进 configs/zh/train-zh800m-12k-c512.toml
python -X utf8 scripts/zh/corpus/prepare_zh800m.py --arm 12k-c512
python -X utf8 scripts/zh/training/train_zh200m.py --config configs/zh/train-zh800m-12k-c512.toml --dry-run
# 训练（存在 last.pt 时自动续训）
VIMEML_NPROC=<GPU 数> scripts/zh/training/cluster_zh800m.sh
# 联合拼音预训练（见偏离 1）：同一混合数据，配置 train-zh800m-12k-c512-pinyin.toml
python -X utf8 scripts/zh/training/train_zh200m.py --config configs/zh/train-zh800m-12k-c512-pinyin.toml --dry-run
VIMEML_ZH800M_VARIANT=pinyin VIMEML_NPROC=<GPU 数> scripts/zh/training/cluster_zh800m.sh
```

训练入口与 200M 共用：`train_zh200m.py` 和 `joint_zh200m.py` 按配置里的 `stage = "zh-800m"` 选择模型、混合数据和批大小。测试在 `tests/test_zh800m.py`，用假 export 加真实 12K 分词器，端到端核对：P1 与 `rest` 恰好划分 export-r1，`all` 覆盖全部文档，映射中每个网页窗口只出现一次，聊天比例为 2%，映射可以逐字节重建，`build_plan` 接受该混合数据。

## 开训前要补齐

- [ ] r1rest、r2 两个 store 编码完成，manifest 写明 `selection`；c512 索引建好。
- [ ] 混合映射建好，`dataset_sha256` 钉入配置，`--dry-run` 核对参数量 784,157,184 和步数。
- [ ] GPU 冒烟：`smoke_updates` 20 次更新跑通 20 个测点，实测微批显存和单步耗时，确认数据并行的进程数。冒烟用拼音变体的配置，测点里要有四类拼音验证。
- [ ] 拼音 store `zh-pinyin-pt-12k` 在集群上，manifest SHA 与配置一致；`--dry-run` 打印各类拼音的遍数。
- [ ] 训练数据经私有渠道分发，按 SHA 清单核对（细节不进仓库）。

## 局限

- export-r2 没有人工审核，审核队列里的文档全部删除，网页质量分布与 r1 的 P1 略有不同。
- 只有一个 seed，和 200M、v1 的差异要和 200M 的 seed 极差（轨道 A Top-3 0.0165）对照着看。
- 学习率和批次没有做小规模扫描（算力充足，按经验值直接定）；如果前几个测点明显异常，再按偏离处理。

## 偏离记录

1. **2026-11-17，开训前：改为联合拼音预训练。** 12K joint-pinyin 200M 的评估（[报告](../reports/zh-200m-pinyin-eval-20261116/report.md)）显示，联合预训练的 200M 通过了 v2 的全部三道 development 门槛。以它自身为先验做重打分，公开拼音集 Top-1 为 58.8%，比 v1 单独高 6.2（p = 5e-5）；LM 能力保住了，所以桌面端只需加载一个模型。走后训练的 r2-cont 则削弱了 LM 能力，自身先验不如 v1 先验。因此 800M 不再先训纯 LM 再后训练，而是直接联合训练。纯 LM 配置 `train-zh800m-12k-c512.toml` 仍然保留，但不启动。
   - 配置 `configs/zh/train-zh800m-12k-c512-pinyin.toml` 与纯 LM 配置只差输出目录、日志目录、tracking 分组和 `[pinyin]` 一节。数据混合、`dataset_sha256`、形状和配方都不变。
   - 拼音数据就是 200M 用的 `zh-pinyin-pt-12k` 四类 store（full/mixed/initials/error），不含 plain。每次更新在 2048 个 LM 窗口之外再加 96/128/112/48 条拼音样例（共 384 条，按样例数计占 15.8%）。200M 的比例是 48/64/56/24。这里翻倍而不是按 LM 批次的 4 倍放大，是为了让 1,198,964 次更新（dry-run 实测）只把拼音数据过 1.35 到 1.75 遍（full 1.42、mixed 1.35、initials 1.38、error 1.75），不反复刷小数据集。
   - 损失沿用 200M 的 per_kind 归一化：每个任务各自取平均损失，再乘以它在批中的样例占比。拼音任务同样按 `lm_microbatch` 切微批，显存上限仍由 LM 窗口决定。实现在 `joint_zh200m.update_joint`。测试在 `tests/test_zh800m_pinyin.py`：任意微批切法的梯度都与 200M 的联合更新一致。
   - 评估新增**拼音生成**：公开拼音集 `ime-pinyin-zh-v1-r1` development，用分桶搜索生成，再用自身作先验做重打分（λ1.6），报告 Top-1。与 v1 LM 模式、joint-pinyin 200M 做句簇配对符号翻转检验。联想主指标和重排守门不变。

## 结果（2026-11-20）

2026-11-20 训完：1,198,964 次更新，用时 58.07 小时。评估见 [800M 评估报告](../reports/zh-800m-eval-20261120/report.md)。800M 在每个注册指标上都领先两个对照，所有配对差异都显著：

- **联想（主指标）**：轨道 A Top-3 为 0.3394。
  - 对 200M 默认臂（0.3164）：+0.0230，超过 200M 的 seed 极差 0.0165；Holm p 1e-4。
  - 对 v1（0.2760）：+0.0634。
- **拼音生成**：重打分 Top-1 为 65.3%。
  - 对用同一流程重跑的 joint-pinyin 200M（57.3%）：+8.0。
  - 对 v1 LM 模式（52.6%）：+12.7。
- **重排守门**：通过，对 v1 净 +26。
- **记忆**：没有长片段照抄。带 8 字上文时，Top-1 在 r1 训练文本中原样出现的比例是 2.4%，人工续写为 2.3%。

问题 1 的三个点齐了：20M、200M、800M 分别为 0.276、0.316、0.339。问题 2（蒸馏老师）由蒸馏第 4 轮回答。
