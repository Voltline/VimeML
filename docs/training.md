# Tokenizer 与 Tiny GPT 训练

正式训练已完成，当前权重冻结。以下是复现指南；每一步都由使用者手动启动，不需要现在重新训练。已有产物和精确结果见 [基线结果](results.md)。

## 环境

Python >=3.11，当前 Windows 使用 3.13。在仓库根目录：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt -r requirements-tokenizer.txt
# Windows / Linux CUDA 的第一轮固定环境
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt --index-url https://download.pytorch.org/whl/cu128
```

数据依赖 pyarrow 25.0.1；SentencePiece 0.2.1；TensorBoard 2.20.0；PyTorch 2.10.0+cu128。TensorBoard 所需 setuptools 固定在 requirements 中。Mac 上的 CPU 推理可用 `.venv/bin/python -m pip install -r requirements-data.txt -r requirements-tokenizer.txt -r requirements-training.txt`，不使用 CUDA index。现有训练器支持 CPU / CUDA，尚未实现 MPS 训练；Core ML 使用另一个 Mac 环境。

## SentencePiece 与并行编码

`configs/tokenizer.toml`：unigram，vocab 16384，identity normalization，保留空白，byte fallback，PAD/UNK/BOS/EOS=0/1/2/3。从完整 train 随机抽 200 万句学习词表，12 个原生线程；三个 split 的全量统计用 8 个独立进程。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/tokenizer/train.py --dry-run
# 新实验必须使用新目录，避免覆盖现有词表
.\.venv\Scripts\python.exe -X utf8 -u scripts/tokenizer/train.py --output artifacts/tokenizers/ja-unigram-16k-v2
.\.venv\Scripts\python.exe -X utf8 scripts/tokenizer/encode.py --dry-run
.\.venv\Scripts\python.exe -X utf8 -u scripts/tokenizer/encode.py --workers 8
```

最后一条默认使用正式 v1 tokenizer、写 `artifacts/token-data/corpus-v1-16k/`。现有输出已完成；换 tokenizer 时用 `--help` 指定相应路径与新输出。编码进程各使用 1 个原生线程，按完整行分块、恢复原顺序合并，并逐句核对 TXT/JSONL、SHA 与 roundtrip。`--resume` 复用校验通过的分块；完成的 v1 不会被再次编码。

每条序列为 BOS + 完整句子 + EOS，不截断、不连接不同句子。tokens.bin 是小端 uint16，offsets.bin / rows.bin 是小端 uint64，sources.bin 是 uint8；rows 指向原 JSONL 字节位置。真实二进制合计约 1.74GB（1.62GiB），无需为训练复制 provenance。

多线程原生训练跨平台未保证词表逐位一致。精确复用必须复制已有 `tokenizer.model` 并校验 hash；固定 seed 并不替代冻结产物。

## 数据加载

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/training/check_data.py
# 仅 NumPy / 索引检查；不执行 CUDA 检查
.\.venv\Scripts\python.exe -X utf8 scripts/training/check_data.py --prepare-only
```

配置在 `configs/loader.toml`：context128、batch128、4 workers。索引只保存长句分窗信息，不复制 token。`x=s[:-1]`、`y=s[1:]`，labels 已右移；PAD label=-100 不参与 loss，EOS 参与。长句分窗覆盖每个预测目标一次，不虚构窗口 BOS。

训练按固定 seed 的块顺序与块内 shuffle，并在有限缓冲内按长度组批后 shuffle 批次，减少 padding；包含最后不足 batch 的一批。validation 顺序读取，test 仅结构检查。worker 独立打开只读 memmap，兼容 Windows spawn。报告为 `artifacts/training-data/corpus-v1-c128/loader-check.json`。

## 手动训练和续跑

模型为标准 decoder-only Tiny GPT，pre-LN、GELU、causal SDPA、学习位置 embedding、共享 embedding/LM head。第一轮配置在 `configs/train-v1.toml`，738 万参数、context128、BF16、AdamW、batch128、4 workers、梯度裁剪、warmup2000、cosine 3e-4 → 3e-5。

300 步 smoke 配置保留在 `configs/train-smoke.toml`，用于新环境检查；旧 smoke 权重已清理。查看计划不会训练：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/training/train.py --config configs/train-v1.toml --dry-run
```

历史正式启动命令：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u scripts/training/train.py --config configs/train-v1.toml
```

现有正式目录已完成，不要更改配置强行覆盖。新训练复制 TOML，修改 output/log 目录后手动启动。`--resume` 只在相同配置、数据与核心代码下继续保存的 checkpoint；已完成一轮不会因此多训练一轮。中断时按一次 Ctrl+C，等待当前 update 后保存退出，再以同一命令加 `--resume`。

每 5000 步在固定 16384 个 validation 窗口计算 token 加权 loss，选择 best；训练末尾分别对 last/best 全量 validation 评估，不改变先前选择规则。训练共 196853 updates，25197117 窗口，573295237 目标；test 不参与选模型。CPU 合成测试覆盖续跑数据位置、梯度累积与随机状态，不保证不同 CUDA 环境逐位一致。

## TensorBoard / W&B

```powershell
.\.venv\Scripts\tensorboard.exe --logdir runs --port 6006
```

Tokenizer 显示 Unigram EM objective、候选 pieces、阶段和统计进度；它不是 Transformer loss。训练显示 train/validation loss、PPL、学习率、有效 tokens/s、padding、数据等待、显存和完整 validation。Tokenizer EM 不展示无法确定的完成百分比。

可选远程面板：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-tracking.txt
.\.venv\Scripts\wandb.exe login
.\.venv\Scripts\python.exe -X utf8 -u scripts/training/train_wandb.py --config configs/train-v1.toml --project vimeml
```

W&B 通过 TensorBoard 同步指标、配置与 SDK 元数据，默认不上传模型、corpus 或 Git 代码。key 在本地登录，不写仓库。`--offline` 无法远程实时看；`--dry-run` 不联网、不训练。续跑加 `--resume`，建立新 run 并归入相同 group，避免回退 step 覆盖历史；不是恢复原云端 run ID。本地事件在 `runs/`，W&B 状态在 `artifacts/tracking/`。

## 只读推理

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/training/infer.py
.\.venv\Scripts\python.exe -X utf8 -u scripts/training/evaluate_ime.py
```

默认 CPU4线程FP32，可加 `--device cuda`。greedy 每步取最大概率 token；sample 按 temperature/top-k/top-p 抽样。联合分词评分和真实候选见 [组合排序](hybrid-ranking.md)，网页见 [联想演示](phrase-demo.md)。不自动开始训练或 API 请求。
