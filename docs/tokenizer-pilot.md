# SentencePiece pilot

在仓库根目录手动执行：

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-tokenizer.txt
.\.venv\Scripts\python.exe -X utf8 src\vimeml\data\train_tokenizer.py --config configs\tokenizer-pilot.toml
```

配置默认读取 `outputs/corpus-pilot/`，只使用 `train.txt` 拟合词表。validation/test 只做编码统计，不参与训练。语料是目前的规则清洗基线：遗漏片段恢复、近似去重和代表性质量抽样尚未完成，生成的 tokenizer 用于跑通流程。

## 参数

- SentencePiece 固定版本 0.2.1；Unigram，词表总大小 16,384。
- 总词表包含 pad=0、unk=1、bos=2、eos=3，以及 256 个 byte fallback token。
- character_coverage=0.9995；未进入字符词表的稀有字符使用 UTF-8 字节回退。
- identity normalization：corpus 已做 NFC，tokenizer 不再把全角、半角等形式统一。保留多余空白，并通过编码/解码检查确认行为。
- 固定 seed=42，默认单线程；全部 train 行参与训练，关闭随机抽样。记录版本、输入和模型哈希。跨平台或库版本改变时，保留原来的 `.model` 才能保证沿用原来的 token IDs。
- max_sentence_length 按 UTF-8 字节计，先预检再训练，超限就报错，避免 SentencePiece 默默跳过长句。

## 输出

`artifacts/tokenizers/ja-unigram-16k-pilot/` 下生成：

- `tokenizer.model`：之后编码训练数据、执行推理时使用的模型。
- `tokenizer.vocab`：可阅读的词表。
- `stats.json`：三个 split 的实际 token 数、字符/token 比率、句长分位数、64/128 tokens 容纳比例、byte fallback 使用量、未知 token 和还原失败数量。
- `examples.json`：固定测试文本与 train/validation 的分词示例。test 不输出人工检查样本。
- `config.json`：配置与实际 SentencePiece 参数。
- `manifest.json`：训练输入、代码和模型哈希；所有检查通过后才生成。

实际 token 统计不含 BOS/EOS；另列的 `tokens_with_bos_eos_per_sentence` 假设每条句子添加两个特殊 token。这一步不做序列拼接或截断，也不生成 PyTorch 数据包。

脚本会检查所有 split 的编码后解码是否等于原文，以及是否出现 `<unk>`。这些检查通过并不说明语料质量已经达到最终训练要求。

输出目录必须不存在或为空；重新试验时指定新目录：

```powershell
.\.venv\Scripts\python.exe -X utf8 src\vimeml\data\train_tokenizer.py --config configs\tokenizer-pilot.toml --output artifacts/tokenizers/ja-unigram-16k-pilot-02
```

后续比较词表大小时，可以使用 `--vocab-size 8192` 或 `--vocab-size 32768`，并为每次实验选择独立输出目录。当前先完成一次 16K 试训。

## 参考

- [SentencePiece v0.2.1 normalization](https://github.com/google/sentencepiece/blob/v0.2.1/doc/normalization.md)
- [SentencePiece v0.2.1 model schema](https://github.com/google/sentencepiece/blob/v0.2.1/src/sentencepiece_model.proto)

已在独立依赖目录用 1,000 条 train 文本训练 4K Unigram，检查了 train-only 输入、特殊 token IDs、256 个 byte tokens、未见 Unicode 字符回退、固定示例及三个 split 的还原、统计和拒绝覆盖。完整 16K pilot 训练留给手动执行。
