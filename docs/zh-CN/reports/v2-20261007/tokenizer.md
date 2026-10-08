# V2.0 tokenizer

2026-10-07，正式训练与全量分词验收完成，用时 313.97 秒。配置见 [tokenizer-v2.toml](../../../../configs/tokenizer-v2.toml)：16K Unigram、train 抽样 200 万句、seed 42、identity normalization、保留空白、byte fallback、`add_dummy_prefix=false`，特殊 ID 0/1/2/3。

三个 split 共 25,713,003 句，UNK=0、roundtrip mismatch=0；7 个句首 probe 无人工空白标记。

| 指标 | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| 句数 | 25,185,368 | 260,337 | 267,298 |
| Content tokens | 531,277,289 | 5,455,745 | 5,544,921 |
| Token 数相对 V1 | -3.071% | -3.095% | -3.122% |
| Chars/token | 2.0346 | 2.0365 | 2.0373 |
| 平均 tokens/句 | 21.0947 | 20.9565 | 20.7443 |
| p50 / p90 / p95 / p99 | 18 / 38 / 46 / 67 | 18 / 38 / 46 / 67 | 18 / 37 / 46 / 67 |
| Context 64 覆盖 | 98.6133% | 98.5864% | 98.6446% |
| Context 128 覆盖 | 99.9560% | 99.9551% | 99.9547% |
| Byte fallback 比例 | 0.3030% | 0.3137% | 0.3108% |

分位数不含特殊 token；覆盖采用 content+BOS+EOS≤context。Token 减少是整套新词表的结果，不能全部归因于 dummy prefix。6 个 IME probes 中 2 个仍有接缝重切分，例如 `昨日友達と` + `話した` 联合编码为 `昨日 / 友達 / と話した`；评分继续联合编码。

校验采用 metadata，复用冻结 corpus 指纹；新 tokenizer 指纹计算一次。产物位于 `artifacts/tokenizers/ja-unigram-16k-v2/`，含模型、词表、config/manifest、全量 validation、stats、examples 与 V1 比较。模型原指纹：`cdb0c60529300619fb0d7b01370c87b3f1028f2687ff67d97c778b15bfa2d177`。

日志：`outputs/history/v2-20261007-session/tokenizer-v2-20261007.log`；TensorBoard：`runs/tokenizer/ja-unigram-16k-v2/`。
