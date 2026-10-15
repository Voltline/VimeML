# V4 基础语言模型训练

V4 使用 8K 词表，将 embedding 参数预算转移给 Transformer 主干。固定 15,070,848 参数架构及 tokenizer 准备见[模型验收](reports/v4-20261014/model.md)和[分词报告](reports/v4-20261014/tokenizer.md)。两次基础 LM 实验于 2026-10-15 在 AutoDL RTX 4090 完成。[英文文档](../training-v4.md)。

## 数据与共同设置

两次实验均使用与 V2.1 相同的原始 FineWeb/Tatoeba train/validation 文本，采用新 8K tokenizer。Token store 包含 25,208,300 个训练窗口；batch 512 时每轮 49,235 次更新，包含最后不足整批的数据。原验证集为 260,337 句、11,110,489 个 Unicode 字符；完整 BPC 覆盖全部 6,627,568 个 V4 预测目标，含 EOS。不同 tokenizer 优先比较 BPC。

| 项目 | 设置 |
| --- | --- |
| 环境 | AutoDL Linux、RTX 4090 24 GiB；固定 CUDA 12.8 / PyTorch 2.7 环境 |
| 初始化 | 随机权重，seed 42 |
| 词表 / context | 8,192 / 128 |
| 架构 | d384、7 层、6 头、SwiGLU 960；15,070,848 参数 |
| 精度 | BF16；参数与 optimizer state 为 FP32 |
| Batch / accumulation | 512 / 1；每次更新最多 65,536 个 padding 后位置 |
| AdamW | CUDA fused；betas (0.9, 0.95)、weight decay 0.1、gradient clipping 1.0 |
| Prefix crop | 合格首窗口概率 0.30，至少保留八个目标 |
| 数据加载 | 四个 workers、四个 CPU threads、长度分桶、pinned memory、persistent workers、prefetch 2 |
| 编译 | 动态形状主干；有效 token 词表头与 loss 保持 eager |
| 验证 | 每 2,000 次更新使用固定 4,096 个抽样窗口并补齐母句；每轮及结束时完整验证 |
| 保存 | 每 5,000 次更新及每轮、完成、中断时 |
| 日志 | TensorBoard，可选 W&B 同步 |

有效目标 token 数随句长与 prefix crop 变化，padding 位置数不等于有效 batch tokens。基础 LM 配置未加入训练期 IME benchmark；固定候选池评估在训练完成后单独进行。

## 已完成调度

| 实验 | 调度 | 实际结束 | 配置 |
| --- | --- | --- | --- |
| V4-A | 初始四轮 cosine，依次延长至八轮、十二轮 | 第 9 轮 plateau 早停，step 443,115 | [初始](../../configs/train-v4-b.toml)、[八轮续训](../../configs/train-v4-b-extended.toml)、[十二轮续训](../../configs/train-v4-b-extended12.toml) |
| V4-B | 初始化时设定完整十二轮 cosine | 第 12 轮，step 590,820；未触发早停 | [十二轮对照](../../configs/train-v4-b12cosine.toml) |

两种调度起始学习率均为 6e-4，warmup 1,000 次更新，最低 6e-5。A 的两次延长从 step 144,039 和 368,826 已提交的 optimizer 学习率开始，在剩余预算内平滑衰减；保留权重、AdamW、RNG、数据游标和已有早停计数。原 checkpoint 与配置保存在 A 模型目录的 `extension-from-four-epochs/` 和 `extension-from-eight-epochs/`。绑定 checkpoint 的 receipt 记录允许的配置/代码变化及源 SHA256；普通 resume 仍严格检查身份。

Plateau 策略要求相对停止参考值累计改善超过 0.01 完整 BPC；连续两轮未达阈值则结束，较小改善的实际最优模型仍保留。A 的续训曲线与 B 从开始设定的十二轮曲线分开记录。B 实际多训练 33.33% 的有效目标，最终结果包含调度与实际预算两项影响。

## 结果与选择

| 实验 | 有效训练目标 | LM 最优完整本地 FP32 BPC | LM 最优 Standard dev | 开发集选择 |
| --- | ---: | ---: | ---: | --- |
| V4-A | 5,174,759,074 | 3.051781125，step 372,000 | 492/728 | 第 8 轮，step 393,880；503/728 |
| V4-B | 6,899,499,874 | 3.027777023，step 584,000 | 498/728 | LM best，step 584,000；498/728 |
| V2.1 基线 | — | 3.080190570 | 505/728 | 部署版 extend5 step 40,000 |

训练 `best.pt` 按固定子集 loss 选择，`best-epoch.pt` 按完整轮次 BPC 选择；后续开发集选择另行比较已记录候选。远端 BF16 与本地 FP32 指标分开记录。两轮均改善 LM BPC，尚未建立稳定优于 V2.1 的重排替换收益。完整方法、来源分组、配对统计与限制见 [A 评估](reports/v4-20261015/evaluation.md)及 [B 对照](reports/v4-20261015/b12cosine-evaluation.md)。

## 运行与产物

运行路径复用 V2.1/V3 的 BF16 SDPA、fused AdamW、pinned memory 非阻塞传输及长度分桶。词表投影和 cross entropy 只处理有效目标。`torch.compile(fullgraph=True, dynamic=True)` 保留参数名称和 checkpoint 格式；prefix crop 与轮次顺序支持确定性恢复。

A 在 443,095 个计时更新、11,732.5 秒更新时段内记录 441,043.7 有效 tokens/s；B 为 439,458.1 有效 tokens/s。计时排除验证、保存及开头 warmup。[V2.1 RTX 4090 D 结果](reports/v2-20261007/v21-performance.md)的模型与 tokenizer 不同，不能据此作受控吞吐对比。Linux 固定依赖见 [requirements-autodl.txt](../../requirements-autodl.txt)，启动封装保留持久化 TorchInductor/Triton cache。

| 路径 | 用途 |
| --- | --- |
| `artifacts/tokenizers/v4-a/ja-unigram-8192/` | 冻结 tokenizer |
| `artifacts/token-data/corpus-v4-8k/` | 完整 train/validation token store |
| `artifacts/training-data/corpus-v4-c128/` | Context-128 窗口索引 |
| `artifacts/models/tiny-ja-v4-b-e8k-d384-l7/` | A checkpoint、进度与验证记录 |
| `artifacts/models/tiny-ja-v4-b12cosine-e8k-d384-l7/` | 独立 B checkpoint 与记录 |
| `runs/training/` / `artifacts/tracking/` | TensorBoard / W&B 记录 |

二进制产物不纳入 Git。恢复时核对源码、tokenizer、token/窗口 manifest、精度与配置身份。训练只消费 train/validation，blind 保持隔离。

## 转换后训练

[V4-C1](posttraining-v4.md)以 B 的 step-584,000 权重为起点，加入读音条件转换与普通 LM 重放。[首批数据](data-v4-c1.md)、专用代码与 [AutoDL 配置](../../configs/posttrain-v4-c1.toml)已准备，正式后训练尚未开始。转换集的轮次预算与上述基础 LM 调度分别记录。
