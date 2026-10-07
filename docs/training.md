# 训练

tiny-ja-v1 已完成并冻结。本文为基线复现入口；当前实验见 [V2 训练](training-v2.md)。

## 环境

Python ≥ 3.11（Windows 当前 3.13），用 uv 管理。

```powershell
python -m venv .venv
# CUDA 机器：先从 PyTorch 官方源装 torch，再从默认 PyPI 装其余依赖
uv pip install --python .venv/Scripts/python.exe torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

不要把整份 requirements 的下载源设成 PyTorch 专用源。Mac 上只做 CPU 推理时直接 `pip install -r requirements.txt`；Core ML 用另一个环境，见 [Core ML 部署](coreml.md#环境)。训练器支持 CPU / CUDA，不支持 MPS。

主要版本：pyarrow 25.0.1、sentencepiece 0.2.1、tensorboard 2.20.0、torch 2.10.0+cu128。

## Tokenizer

配置 `configs/tokenizer.toml`：unigram，词表 16,384，identity normalization，保留空白，byte fallback，特殊 ID PAD/UNK/BOS/EOS = 0/1/2/3。从 train 中随机抽 200 万句学习词表。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/tokenizer/train.py --dry-run
.\.venv\Scripts\python.exe -X utf8 -u scripts/tokenizer/train.py --config configs/tokenizer.toml --output artifacts/tokenizers/NEW-VERSION
```

v1 结果：三个 split 全量 UNK 和 roundtrip 错误都为 0，byte fallback 约 0.30%。多线程训练在不同平台上不能保证词表逐位一致，要复用请直接复制 `tokenizer.model` 并核对哈希。

## 编码

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/tokenizer/encode.py --dry-run
.\.venv\Scripts\python.exe -X utf8 -u scripts/tokenizer/encode.py --workers 8
```

默认使用 v1 tokenizer，输出到 `artifacts/token-data/corpus-v1-16k/`。按完整行分块并行编码，恢复原顺序后合并，逐句核对 TXT/JSONL、SHA 与 roundtrip；`--resume` 复用已校验的分块。

格式：每条序列是 BOS + 完整句子 + EOS，不截断，不拼接不同句子。`tokens.bin` 为 uint16，`offsets.bin` / `rows.bin` 为 uint64，`sources.bin` 为 uint8，均为小端；`rows` 指向 JSONL 中的字节位置。三个 split 合计约 1.74 GB。

## 数据加载

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/training/check_data.py               # 含 CUDA 检查
.\.venv\Scripts\python.exe -X utf8 scripts/training/check_data.py --prepare-only   # 只建索引
```

配置 `configs/loader.toml`：context 128，batch 128，4 workers。

- 输入 `x = s[:-1]`，目标 `y = s[1:]`；PAD 的 label 为 -100，EOS 参与 loss。
- 每个窗口只装一个句子，超过 129 token 的长句切成多个窗口，每个预测目标恰好覆盖一次，不在后续窗口里虚构 BOS。
- 训练按固定 seed 的块顺序与块内 shuffle，在有限缓冲内按长度组批以减少 padding，再打乱批次顺序。
- 索引写在 `artifacts/training-data/corpus-v1-c128/`，只保存窗口位置，不复制 token。

## 训练

模型：pre-LN decoder-only GPT，GELU，causal SDPA，学习式位置 embedding，共享 embedding / LM head。

`configs/train-v1.toml`：BF16，AdamW（β 0.9 / 0.95，weight decay 0.1），batch 128，梯度裁剪 1.0，warmup 2000 步，cosine 学习率 3e-4 → 3e-5，1 epoch = 196,853 步。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/training/train.py --config configs/train-v1.toml --dry-run
```

- 新实验复制 TOML 并设置独立 `output_dir` / `log_dir`，正式目录保持冻结。
- 中断：按一次 Ctrl+C，等当前 update 保存后退出；再用同一命令加 `--resume` 续跑（要求配置、数据和核心代码都相同）。
- 每 5000 步在固定的 16,384 个 validation 窗口上评估并选 best，结束时对 best / last 做全量 validation。test 不参与选模型。
- `configs/train-smoke.toml` 保留为 300 步诊断配置。

tiny-ja-v1 实际结果：25,197,117 个窗口，573,295,237 个预测目标，3469.8 秒，约 169,587 有效 token/s；全量 validation loss 4.5416（PPL 93.84），best 就是最后一步。记录在 `artifacts/models/tiny-ja-v1/{summary.json,manifest.json,metrics.jsonl,full-validation.json}`。

```text
best.pt         93b6139aa05031df02038efcaf26788bc9916712919f3d2a4c4f398615cef75b
tokenizer.model d9f1ba1e456ce72dd9c06a62b12e804cf14090c30179e6078cd51d03063b2982
```

## 监控

```powershell
.\.venv\Scripts\tensorboard.exe --logdir runs --port 6006
```

训练指标包括 train / validation loss、PPL、学习率、有效 token/s、padding 比例、数据等待时间、显存。tokenizer 训练显示的是 Unigram EM 目标，不是 Transformer loss。

W&B（通过 TensorBoard 同步，不上传模型、语料或代码）：

```powershell
.\.venv\Scripts\wandb.exe login
.\.venv\Scripts\python.exe -X utf8 -u scripts/training/train_wandb.py --config configs/train-v1.toml --project vimeml
```

`--offline` 只写本地，`--dry-run` 不联网也不训练。续跑时加 `--resume`，会新建一个 run 并放进同一 group。W&B 状态保存在 `artifacts/tracking/`。

## 推理

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/training/infer.py            # greedy / sample 续写
.\.venv\Scripts\python.exe -X utf8 -u scripts/training/evaluate_ime.py   # 手写同音词诊断
```

默认 CPU 4 线程 FP32，可加 `--device cuda`。候选评分和联想见[评测](evaluation.md)。
