# V4 8K token 数据与 context-128 窗口

2026-10-14。V4 固定选择 8192 词表，目的是减少 embedding 的参数分配、增加 Transformer 主干的可学习容量。12K 不作为后续 V4 模型候选；此前 12K/16K tokenizer 实验保留为历史统计对照。速度是需要测量的约束，不是选择 8K 的主要理由。

[配置](../../../../configs/data-v4.toml)、[入口](../../../../scripts/tokenizer/encode_v4.py)、[实现](../../../../src/vimeml/tokenizer/encode_v4.py)、[tokenizer 记录](tokenizer.md)。

## 数据范围与指纹

使用完整冻结 `outputs/corpus-fast-v1/train.txt` 的 25,185,368 句，以及 `validation.txt` 的 260,337 句，保留原顺序、空白和 split。使用已经验收的 `artifacts/tokenizers/v4-a/ja-unigram-8192/tokenizer.model`，不重训 tokenizer，也不把 tokenizer 的 200 万句抽样当作完整 LM 语料。不读取 test/blind 文本、benchmark 或参考答案，不进行模型评分或 Transformer 训练。

完整 train JSONL 已在此前本地清理中移除。本次直接从 TXT 编码，仅复用冻结 V2 `sources.bin` 的逐句主要来源 ID。复用前必须核对 corpus manifest、实际 TXT SHA256、句数/字符数/来源计数和来源文件指纹。不复用 V2 token 序列或 offsets。

每句必须完整 roundtrip，token ID 小于 8192，正文不得出现保留 ID 0–6。序列为 BOS + 完整正文 + EOS；不丢弃、不截断、不拼接句子。分片位于完整 UTF-8 行边界，并记录起始原句行号，保证各 worker 来源对齐。

V4 的 `rows.bin` 保存对应 **TXT** 的 uint64 字节偏移，与旧 JSONL 偏移不同；manifest 明确写入 `row_offset_source=corpus_txt`。主要来源 ID 不能替代完整文档级 provenance，历史 corpus 构建记录仍单独保留。

## 产物与复现

```powershell
.venv/Scripts/python.exe -X utf8 scripts/tokenizer/encode_v4.py --dry-run
.venv/Scripts/python.exe -X utf8 -u scripts/tokenizer/encode_v4.py
# 中断后复用已验证分片，或完成窗口/loader 验收。
.venv/Scripts/python.exe -X utf8 -u scripts/tokenizer/encode_v4.py --resume
```

配置为 8 workers、每个 native encoder 单线程、batch4096、128 MiB 分片。非空目录默认拒绝覆盖，`--resume` 核对 signature 与分片指纹。编码前后计算输入 SHA256，并保留分片和最终二进制指纹。临时分片在最终 token 验收后仅从本实验输出目录中删除；V1/V2 代码和产物不变。

| 路径 | 内容 |
| --- | --- |
| `artifacts/token-data/corpus-v4-8k/` | train/validation 的 uint16 tokens、uint64 offsets/TXT 行位置、uint8 来源、manifest/stats/part audit |
| `artifacts/training-data/corpus-v4-c128/` | 窗口 NPZ 索引、manifest 与 loader-check |
| `outputs/tokenizer/v4-data-20261014.log` | 日志 |
| `runs/tokenizer/corpus-v4-8k-encode/` | TensorBoard 进度 |

context128 窗口保留所有 next-token prediction pairs，包括 EOS。窗口接缝共享一个 input token，预测目标无重叠、无遗漏；后续窗口不补 BOS。PAD=0，padding labels=-100，并有 attention mask。使用现有 `TokenStore`、`SentenceWindowDataset` 和 CPU PyTorch DataLoader 契约，不修改 V1/V2 核心代码。

V4 index 只包含 train/validation。共享 `prepare_indexes` 已支持显式 split，V4 训练入口选择这两个 split，旧 CLI 默认行为不变。正式配置与运行设计见 [V4 训练设计](../../training-v4.md)。

## 验证与限制

11 项 tokenizer 测试通过，覆盖 TXT 分片边界/来源对齐、实际 native token 序列、TXT 位置恢复、长句预测目标覆盖、padding loss mask 和完成后的 resume。完整真实数据验收包括每句 roundtrip/ID 检查、validation 与 tokenizer 阶段统计一致性、合并 offsets/计数/大小/来源核对、文件指纹、选定长句窗口，以及两个 spawn workers 的 CPU PyTorch batches。这是数据路径验收，不是 LM 正式评测或 GPU 训练吞吐测量。

原 corpus 的 `ready_for_lm_training=false` 及清洗/near-duplicate 限制仍明确保留。这次重编码不等于完成新的日语质量审核或发布级语料验收。不读取 blind，不修改 benchmark 标签。

## 实测产物

后续模型准备已给共享 index preparation 增加显式 split 选项，并提供 V4 训练入口；旧命令仍默认三个 split。见 [模型、本地验收与启动配置](model.md)。

编码、合并与 token-store 指纹计算用时 158.44 秒，之后另行完成窗口与 loader 验收。8 个二进制文件共 1,787,137,101 bytes（1.664 GiB），窗口 NPZ 共 523,408 bytes。

| Metric | Train | Validation |
| --- | ---: | ---: |
| Sentences | 25,185,368 | 260,337 |
| Content tokens | 620,021,409 | 6,367,231 |
| Stored tokens (BOS/EOS) | 670,392,145 | 6,887,905 |
| Prediction pairs | 645,206,777 | 6,627,568 |
| Context-128 windows | 25,208,300 | 260,591 |
| Additional long-sentence windows | 22,932 | 254 |
| Longest sequence (BOS/EOS) | 709 | 506 |
| Characters/token | 1.7434 | 1.7449 |
| Byte-token fraction | 0.2615% | 0.2709% |
| UNK / roundtrip errors | 0 / 0 | 0 / 0 |

完整保留 train 的 645,206,777 个、validation 的 6,627,568 个预测目标。核心结构检查覆盖 train 60 个窗口 / 3022 个目标和 validation 59 个窗口 / 2618 个目标，包含长句接缝。两个 spawn workers 的 CPU PyTorch 验收每个 split 运行 4 batches / 256 windows，核对 train 6429、validation 6522 个目标。未使用 length bucket 的诊断 padding 比例不能作为正式训练效率结论。

最终 token-store manifest SHA256：`c988f8eabfb903de25bc6d526f9ea42e686771814f7e33da56e0e6dc17931d5b`。完成后重新核对二进制/来源指纹、窗口指纹、tokenizer 身份、split 清单、预测目标总数，并确认没有 test 输出或临时分片残留。

Token store、窗口索引、模型与本地检查均已完成，下一步直接启动完整基础 LM 正式训练。原 corpus quality metadata 保持不变，重编码不等于重新审核语料质量。
