# Pilot token data

在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 src\vimeml\data\tokenize_corpus.py
```

默认读取 `outputs/corpus-pilot/` 和 `artifacts/tokenizers/ja-unigram-16k-pilot/`，保存到 `artifacts/token-data/corpus-pilot-16k/`。复用已安装的 SentencePiece，无需增加依赖。

## 存储格式

每条 corpus 记录保存为 `[BOS] + 正文 token IDs + [EOS]`。三个 split 各自导出：

- `<split>.tokens.bin`：小端 uint16，顺序保存完整序列。
- `<split>.offsets.bin`：小端 uint64，共 N+1 个数。第 i 条记录是 tokens[offsets[i]:offsets[i+1]]；offset 单位是 token，不是字节。
- `<split>.provenance.jsonl`：来源、文档 ID、文档与文本哈希。每条记录的 index / corpus_row 对应原 corpus JSONL 的零基行号，完整清洗位置等 metadata 保留在原 JSONL。
- `stats.json`：句子、正文 tokens、包含 BOS/EOS 的存储 tokens、next-token 预测位置数量、文件大小等统计。
- `manifest.json`：格式、特殊 IDs、输入/输出哈希和模型版本。所有导出检查通过后才生成。

存储文件在磁盘上连续排列，但每条样本的边界由 offsets 明确指定。后续 DataLoader 应按 offsets 读取单条序列。长句完整保留，窗口长度、padding 和 labels mask 留到 DataLoader 阶段处理。

BOS=2，EOS=3。对完整序列 `s`，next-token 训练输入为 `s[:-1]`，目标为 `s[1:]`；因此每条序列提供 len(s)-1 个预测位置。BOS 用于预测正文第一个 token，正文最后一个 token 用于预测 EOS。

## 读取示例

在仓库根目录运行下面 Python 代码：

```python
import sys
sys.path.insert(0, "src")

from vimeml.data.token_store import TokenStore

with TokenStore("artifacts/token-data/corpus-pilot-16k", "train") as store:
    s = store[0]
    x, y = s[:-1], s[1:]
    print(len(store), s, x, y)
```

`TokenStore` 使用只读内存映射，仅将当前读取的一条序列转换为 Python 整数列表。请使用 with 或 close() 关闭文件。将来使用 Windows 多进程 DataLoader 时，各 worker 应独立创建自己的 TokenStore。

## 当前结果与验证

使用当前 pilot corpus 和模型的完整临时导出已通过：所有 offsets 递增、TXT/JSONL 与 provenance 对齐、模型和输入哈希一致、导出 token 数与 tokenizer 统计一致，以及每个 split 首条、末条、最长条的解码检查。包含 BOS/EOS 的 421-token 最长训练序列被完整保留。

预期 train：228,284 条记录，正文 4,889,425 tokens，存储 5,345,993 tokens，5,117,709 个预测位置。tokens.bin 为 10,691,986 bytes，offsets.bin 为 1,826,280 bytes。来源索引文件另计。

这些是 pilot 数据，语料恢复和近似去重仍待完成。文件格式和编码正确性不等于语言质量验证。

脚本拒绝覆盖非空目录。需要重新导出时使用 `--output artifacts/token-data/corpus-pilot-16k-02`；不同 corpus / tokenizer 可用 `--corpus`、`--tokenizer` 指定。
