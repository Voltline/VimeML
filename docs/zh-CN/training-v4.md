# V4 基础语言模型正式训练

Tokenizer、完整 train/validation token 数据、context128 索引与固定 15,070,848 参数模型已准备好。已有基本检查记录见 [模型验收](reports/v4-20261014/model.md)。

V4 固定使用 8192 词表，将 embedding 预算转移给 Transformer 主干容量。当前只有一条训练路线：直接启动完整基础 LM 训练，不安排前置短程训练，也不以 Apple 测试作为本地训练前置条件。

## 启动与恢复

2026-10-15 早上在仓库根目录直接执行：

```powershell
.venv/Scripts/python.exe -X utf8 -u scripts/training/train_v4.py
```

默认配置为 [train-v4-b.toml](../../configs/train-v4-b.toml)，显式写法等价：

```powershell
.venv/Scripts/python.exe -X utf8 -u scripts/training/train_v4.py --config configs/train-v4-b.toml
```

中断后使用相同配置恢复：

```powershell
.venv/Scripts/python.exe -X utf8 -u scripts/training/train_v4.py --resume
```

准备阶段未启动正式语料训练。以上为手动启动命令，没有创建定时任务。

## 训练设置

| 项目 | 设置 |
| --- | --- |
| 初始化 | 全新随机权重 |
| 数据 | 完整冻结 train，独立 validation |
| 词表 / context | 8192 / 128 |
| 模型 | d384、7 layers、6 heads、SwiGLU960 |
| 精度 | CUDA BF16 |
| Batch / accumulation | 128 / 1 |
| 预算 | 2 epochs，393,880 updates |
| AdamW | betas(0.9,0.95)、weight decay0.1 |
| 学习率 | 6e-4 → 6e-5，cosine |
| LR warmup | 训练开始后的前1000个 updates |
| Gradient clipping | 1.0 |
| Prefix crop | 基线概率0.30，最少保留8 tokens |
| DataLoader | 4 workers、length buckets |
| Validation | 每2000 updates 的 subset；每 epoch 和结束时 full validation |
| Checkpoint | 每5000 updates，以及 epoch/完成/中断时保存 |
| Compile | 关闭 |
| 日志 | 本地 TensorBoard |

有效 target token 数取决于句长与 prefix crop。Validation BPC 包含 EOS，分母使用原始 Unicode 字符数；不同 tokenizer 优先比较 BPC。Benchmark 表为空，不消费 blind/reference sets，也不自动进行 IME 模型选型。训练超参数是本次实验起始设置，不宣称已优化到最优。

## 文件位置

| 路径 | 内容 |
| --- | --- |
| `artifacts/tokenizers/v4-a/ja-unigram-8192/` | 冻结 tokenizer |
| `artifacts/token-data/corpus-v4-8k/` | 完整 train/validation token 数据 |
| `artifacts/training-data/corpus-v4-c128/` | 窗口索引 |
| `artifacts/models/tiny-ja-v4-b-e8k-d384-l7/` | 权重、optimizer/RNG checkpoint、进度与 validation 结果 |
| `runs/training/tiny-ja-v4-b-e8k-d384-l7/` | TensorBoard events |

Checkpoint 记录模型/数据指纹和配置。V2 token IDs/checkpoint 及合成测试 fixture 不作为本次初始化。Apple 导出、量化与设备部署在后续阶段单独处理。
