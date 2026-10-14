# V4 基础语言模型训练设计

目标环境为 AutoDL Linux、RTX 4090 D（24 GiB）。Tokenizer、完整 train/validation token 数据、context128 索引与固定 15,070,848 参数模型已准备好。已有本地检查记录见 [模型验收](reports/v4-20261014/model.md)。

V4 固定使用 8192 词表，将 embedding 参数预算转移给 Transformer 主干容量。本次实验为从随机权重开始的完整基础 LM 训练，配置见 [train-v4-b.toml](../../configs/train-v4-b.toml)。

## 训练设置

| 项目 | 设置 |
| --- | --- |
| 目标环境 | AutoDL Linux、RTX 4090 D 24 GiB；沿用已有 CUDA12.8/PyTorch2.7 固定环境 |
| 初始化 | 全新随机权重 |
| 数据 | 完整冻结 train，独立 validation |
| 词表 / context | 8192 / 128 |
| 模型 | d384、7 layers、6 heads、SwiGLU960 |
| 精度 | CUDA BF16；参数与 optimizer state 保持 FP32 |
| Batch / accumulation | 512 / 1；每次更新最多65,536个 padding 后的输入位置 |
| 预算 | 4 epochs，每 epoch 49,235 updates，合计196,940 |
| AdamW | CUDA fused；betas(0.9,0.95)、weight decay0.1 |
| 学习率 | 6e-4 → 6e-5，cosine |
| LR warmup | 正式训练的前1000个 updates |
| Gradient clipping | 1.0 |
| Prefix crop | 基线概率0.30，最少保留8 tokens |
| DataLoader | 4 workers、4 CPU threads、length buckets、pinned memory、persistent workers、prefetch2 |
| Validation | 每2000 updates 使用固定4096个抽样窗口并补齐对应句子的所有窗口；每 epoch 和结束时 full validation |
| Checkpoint | 每5000 updates，以及 epoch/完成/中断时保存 |
| Compile | 动态形状主干编译；有效 token 的输出头与 loss 保持 eager |
| 日志 | TensorBoard 与已有 W&B 同步 |

预算根据25,208,300个训练窗口计算，包含最后一个不足512的 batch。有效 target token 数取决于句长与 prefix crop；65,536只是 padding 上限，不是有效 token batch size。Validation BPC 包含 EOS，分母使用原始 Unicode 字符数；不同 tokenizer 优先比较 BPC。Benchmark 表为空。训练超参数是本次实验的初始设置，不宣称已优化到最优。沿用每轮 BPC 退化保护：patience2、最小变化0.01，持续退化时可能在4轮预算用完前结束，并保留最佳 checkpoint。

## 运行效率

实现复用 V2.1/V3 的训练路径：BF16 SDPA attention、fused AdamW、pinned memory 配合 nonblocking transfer，以及减少 padding 的长度分桶。词表投影与 cross entropy 只处理有效 target 位置。主干使用 `torch.compile(fullgraph=True, dynamic=True)`，有效 token 数量变化的输出头留在编译图外。编译不改变参数名称与 checkpoint format；prefix crop、epoch 顺序继续支持确定性恢复。

已有 [V2.1 的 RTX4090 D 实测](../../reports/v2-20261007/v21-performance.md)中，batch512 eager 吞吐为215,837有效 tokens/s，compiled 为278,818，compiled peak allocated 为10.26 GiB。这些属于 V2.1 的结果。V4 主干更大、词表更小、序列长度分布不同，目前没有 V4 在 AutoDL 上的实际吞吐与显存结果。Batch512 和4 workers 沿用已有配置，不宣称已经最大化 GPU 利用率。

AutoDL 入口沿用旧 launcher 的约定，包括持久化 TorchInductor/Triton cache 与已有 tracking 环境。[requirements-autodl.txt](../../requirements-autodl.txt)记录之前的 Linux 依赖。编译在正常训练内部发生，配置没有单独的前置短程训练阶段。

## 产物与身份

| 路径 | 内容 |
| --- | --- |
| `artifacts/tokenizers/v4-a/ja-unigram-8192/` | 冻结 tokenizer |
| `artifacts/token-data/corpus-v4-8k/` | 完整 train/validation token 数据 |
| `artifacts/training-data/corpus-v4-c128/` | 窗口索引 |
| `artifacts/models/tiny-ja-v4-b-e8k-d384-l7/` | 权重、optimizer/RNG checkpoint、进度与 validation 结果 |
| `runs/training/tiny-ja-v4-b-e8k-d384-l7/` | TensorBoard events |
| `artifacts/tracking/` | W&B 同步记录 |

已准备的二进制产物保存在本地，不纳入 Git。训练配置同时依赖 tokenizer、token store 与索引，训练过程不包含在 AutoDL 上重新生成这些数据。Checkpoint 记录模型/数据/代码指纹与配置；恢复要求身份相同，并还原 optimizer/RNG state 与已提交的数据游标。V2 token IDs/checkpoint 及合成 fixture 不作为本次初始化。训练仅消费 train 与 validation，不访问 blind 数据。准备阶段尚未启动正式 V4 语料训练。
