# V4-a tokenizer 初始化与训练

本次范围是本地 tokenizer 初始化、训练与分词统计。固定 V4-a 模型规格仍为 8192 / d384 / 7 层 / 6 heads / FFN960 / context128，15,070,848 参数；本次没有初始化或训练 Transformer，没有导出 Core ML，没有执行模型评分。

## 配置与隔离

2026-10-14 决策：V4 固定使用 8K，目的是把 embedding 的参数分配转移给主干容量；12K 不作为后续模型备选。下文 12K/16K 保留为已完成的历史 tokenizer 统计。完整语料准备见 [8K token 数据与窗口](token-data.md)。

入口：[train_v4.py](../../../../scripts/tokenizer/train_v4.py)。配置：[tokenizer-v4.toml](../../../../configs/tokenizer-v4.toml)。实现：[v4.py](../../../../src/vimeml/tokenizer/v4.py)。V1/V2 的 tokenizer、模型代码和产物保持原状。

```powershell
.venv/Scripts/python.exe -X utf8 -u scripts/tokenizer/train_v4.py --dry-run
.venv/Scripts/python.exe -X utf8 -u scripts/tokenizer/train_v4.py
```

正式入口拒绝覆盖已有实验目录。新的实验需要独立配置与 output 路径。

三个 tokenizer 为 8192、12288、16384，全部使用 Unigram、identity、byte fallback、`add_dummy_prefix=false`、保留空白、hard vocab limit。PAD/UNK/BOS/EOS = 0/1/2/3；`<CTX>/<READ>/<OUT>` 为 USER_DEFINED token，ID = 4/5/6。它们仅预留，本次没有生成 Conversion 输入。将来构造任务输入需要明确原始文本中标记字符串的处理规则。

训练复用 `outputs/corpus-fast-v1/train.txt`。完整扫描 train 并核对 SHA256、句数、字符数、最大句子 UTF-8 长度和保留字符串；使用 Python `random.Random(42)` reservoir sampling 抽取 2,000,000 句，持久化为单个 `train-sample.txt`，三个训练读取相同文件的全部内容，不再次抽样。这样固定对照中的训练文本变量。SentencePiece 0.2.1 内部 reservoir 使用独立固定 seed，不能将其采样等同于 Python seed42；本次 sample 不宣称与 V2 的 sample 相同。

validation 的实际 SHA256 与冻结 corpus integrity report 一致；只用于统计，不参与拟合。评估范围是完整训练 sample、完整 validation、显式 standard development 目录。入口拒绝 test/blind 路径，未打开 test 或 blind 文本，也没有模型 logP、reranking 或参考答案准确率计算。benchmark 输入与参考未修改。

冻结 corpus manifest 的 `ready_for_lm_training=false` 是已有状态；tokenizer 开发不改变这一状态，也不能据此声明 V4-b 已具备正式模型训练资格。

## 单字符假名约束

固定清单：[kana-v4.txt](../../../../configs/kana-v4.txt)，共 576 个 Unicode 15.1 已分配字符，一行一个字符。范围：

| 范围 | 内容 |
| --- | --- |
| U+3041–U+309F | 平假名、组合/间隔浊音符、迭代符 |
| U+30A0–U+30FF | 片假名、长音、间隔符及迭代符 |
| U+31F0–U+31FF | Katakana Phonetic Extensions |
| U+FF66–U+FF9F | 半角片假名与浊音符 |
| U+1AFF0–U+1AFFF | Kana Extended-B 的已分配字符 |
| U+1B000–U+1B0FF | Kana Supplement |
| U+1B100–U+1B12F | Kana Extended-A 的已分配字符 |
| U+1B130–U+1B16F | Small Kana Extension 的已分配字符 |

清单包括历史假名，是本次显式采用的保守覆盖范围；不包括未分配码位或所有注音/日语符号。清单已冻结为文件，运行时不会随着 Python Unicode 数据库升级自动扩大。

`required_chars` 保证单字符词表项保留，但允许常用假名串形成多字符词片。这里“独立 token”的含义是词表包含对应的 NORMAL 单字符项，而且单字符输入编码为一个 ID；不强制所有 reading 逐字拆分。若以后要求任何位置都逐字拆分，那是另一个实验变量。

训练后验收逐项检查 576 个字符的 token 类型、精确 piece、单字符编码、解码和 UNK。另检查组合浊音、半角浊音、emoji、罕见汉字、空白、tab 与保留标记的 roundtrip。

特殊 token 和 byte token 均占总词表额度：4 + 3 + 256 = 263。8K 的普通词片额度为 7929；576 个保留假名也在这部分之内，其他必要单字符进一步占用词片额度。不是“8192 个普通词片外加这些标记”。

训练可能因 required alphabet + meta pieces 超过 vocab 而失败，也可能因语料过少无法达到 hard vocab limit。入口保留完整 native 日志且不静默缩小词表；不能只凭 character coverage 判断约束成立。

实现依据为 [SentencePiece 0.2.1 TrainerInterface](https://github.com/google/sentencepiece/blob/v0.2.1/src/trainer_interface.cc)、[Unigram FinalizeSentencePieces](https://github.com/google/sentencepiece/blob/v0.2.1/src/unigram_model_trainer.cc)、[TrainerSpec](https://github.com/google/sentencepiece/blob/v0.2.1/src/sentencepiece_model.proto)。required characters 在最终 Unigram 词表中保留，meta pieces 从词表预算扣除。

## 统计定义与产物

所有长度均不含特殊 token。报告 tokens/sentence、tokens/candidate、characters/token、byte-token 占全部 content tokens 的比例、UNK、roundtrip、长度分位数与 `content+BOS+EOS≤128` 的句子覆盖。

reading 同时统计 benchmark 原始片假名与诊断性平假名版本。平假名版本仅执行 U+30A1–U+30F6 的对应映射，不修改输入文件，不宣称是后续 Conversion 的最终规范化策略。

语料的 `128 × chars/token` / `126 × chars/token` 是字符容量估计，不是实测连续窗口覆盖。development 另用 tokenizer offsets 统计前 126 个 content tokens 实际覆盖的字符数，包含短文本；它也不能替代真实 packed LM context 的覆盖测试。

| 路径 | 内容 |
| --- | --- |
| `artifacts/tokenizers/v4-a/train-sample.txt` | 三个词表共用的训练抽样 |
| `artifacts/tokenizers/v4-a/sampling.json` | train/sample 指纹与抽样策略 |
| `artifacts/tokenizers/v4-a/ja-unigram-{8192,12288,16384}/` | model、vocab、trainer request/log、manifest、stats、逐字 validation |
| `artifacts/tokenizers/v4-a/comparison.json` | 三个实验的统计对照 |
| `artifacts/tokenizers/v4-a/manifest.json` | 整个实验的最终完成状态、数据/代码指纹 |
| `outputs/tokenizer/v4-a-20261014.log` | 本次运行日志 |
| `outputs/tokenizer/v4-a-v2-reference.json` | 既有 V2 tokenizer 的 development 统计与历史 validation 统计 |
| `runs/tokenizer/v4-a-*` | TensorBoard 日志 |

sample、tokenizer 与 benchmark 统计均留在本地忽略目录。primary tokenizer 是 `artifacts/tokenizers/v4-a/ja-unigram-8192/tokenizer.model`。完整 train/validation 已使用这份冻结模型与指纹重编码，见 [token 数据](token-data.md)；不复用 V2 token 数据或 checkpoint embedding。

本实验明确区分 train sample 和完整 train。V4 独立 train/validation 导出入口已完成，完整语料统计另见 token 数据报告。

## 验证

```powershell
.venv/Scripts/python.exe -X utf8 -m unittest discover -s tests -p 'test_tokenizer*.py'
```

本次 tokenizer 测试 9 项通过，包含实际 SentencePiece 训练对未出现扩展假名的覆盖检查、身份/空白 normalization、固定特殊 ID、保留标记、数据路径隔离以及确定性 reservoir sample。

## 实测结果

本次完整运行用时 799.51 秒（13.33 分钟）。三个词表均精确达到目标容量，576 个假名单字符检查全部通过。每个模型的 200 万句训练抽样、260,337 句完整 validation 均为 UNK=0、roundtrip mismatch=0。Standard development 覆盖 728 条输入、8790 个真实候选；candidate 和 context+candidate 全部满足 content+BOS+EOS≤128，UNK 与 roundtrip 错误均为 0。

| Metric | V4 8K | V4 12K | V4 16K | Existing V2 16K |
| --- | ---: | ---: | ---: | ---: |
| Validation tokens/sentence | 24.4576 | 22.2564 | 21.0468 | 20.9565 |
| Validation chars/token | 1.7449 | 1.9175 | 2.0277 | 2.0365 |
| Validation content tokens | 6,367,231 | 5,794,172 | 5,479,254 | 5,455,745 |
| Validation byte tokens | 17,252 | 17,252 | 17,252 | 17,113 |
| Validation byte-token fraction | 0.2709% | 0.2977% | 0.3149% | 0.3137% |
| Validation p50 / p95 / p99 | 21 / 54 / 78 | 19 / 49 / 71 | 18 / 46 / 68 | 18 / 46 / 67 |
| Validation fits 128 + BOS/EOS | 99.9070% | 99.9416% | 99.9547% | 99.9551% |
| Estimated chars / 126 content tokens | 219.86 | 241.61 | 255.49 | 256.60 |
| Development tokens/candidate | 11.4820 | 10.4761 | 10.0470 | 10.0494 |
| Development candidate chars/token | 1.5552 | 1.7046 | 1.7774 | 1.7769 |
| Development candidate byte fraction | 1.3723% | 1.5040% | 1.5683% | 1.5781% |
| Development tokens/context+candidate | 18.8307 | 17.1592 | 16.4230 | 16.3642 |
| Reading tokens (original katakana) | 20.4437 | 19.5481 | 18.9066 | 19.3462 |
| Reading tokens (hiragana diagnostic) | 16.8146 | 15.8489 | 15.0096 | 14.9409 |

8K 相对同抽样、同假名约束的 V4 16K，validation content tokens 增加 16.21%；相对已有 V2 16K 增加 16.71%，development candidate tokens 增加 14.26%。V2 对照还包含抽样和假名清单差异，不能作为单独词表容量变量的消融。三档受控实验的 validation byte tokens 均为 17,252，比例差异来自总 token 分母。Reading 中的 fallback 仅来自 19 个全角括号（57 个 byte tokens），不是假名缺失。这些实测仅证明分词开销，不证明 LM 或重排性能提高。

主模型 SHA256：`296a2bdf4f4862be8bda848ed1cd261f0b0fd43064181fad5a513c6ce73081b5`。共享 sample SHA256：`0a7aa1eeaa8abf160d34d8338deb5d4b8386ed1a0bf1a7bc8c4c3e388c538e3d`。完成后已重新核对 model/vocab、sample、code/config 和 development 输入指纹。

固定 15,070,848 参数模型与本地检查现已完成，见 [模型准备](model.md)。下一步按照 [训练指南](../../training-v4.md)直接启动完整基础 LM 正式训练。
