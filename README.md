# VimeML

Vime 的本地日语语言模型实验。当前阶段先构建可追踪的语料，供 SentencePiece 和后续 causal LM 使用。

- [当前进度与下一步](docs/next-steps.md)
- [本机多进程清洗，一条命令运行](docs/corpus-parallel.md)
- [复用现有 worker，独立并行汇总](docs/corpus-fast-merge.md)
- [全量 review 与抽检工具](docs/corpus-review.md)
- [Windows + Mac 分工清洗与汇总](docs/corpus-two-machines.md)
- [单机 corpus 与清洗标注契约](docs/corpus-formal.md)
- [Tokenizer pilot](docs/tokenizer-pilot.md)
- [Token 文件与读取接口](docs/token-data.md)

Python 3.13 推荐，最低 3.11。清洗阶段依赖见 `requirements-data.txt`；tokenizer 阶段另见 `requirements-tokenizer.txt`。

代码、配置、少量审核标注和测试随 Git 保存；原始数据、模型、生成结果、虚拟环境和密钥留在本地。
