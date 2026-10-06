# VimeML

Vime 的本地日语语言模型数据与训练实验。当前全量 corpus 已构建并通过导出完整性检查，下一步是 review 和抽检。Transformer 尚未开始实现或训练。

## 目录

```text
scripts/                 手动运行的入口
  corpus/                并行构建、独立分区、并行合并
  review/                准备材料、API 审核、明确批准
  tokenizer/             SentencePiece 训练、token 文件生成
  tools/                 原始数据检查、现有 corpus 检查
src/vimeml/              脚本背后的实现
  data/                  带版本指纹的清洗、分句、分组和导出核心
  corpus/                构建进程编排
  review/                质量审核与缓存/限流
  tokenizer/             tokenizer 与 TokenStore
  tools/                 检查工具
configs/                 当前 corpus 与 tokenizer 配置
annotations/             批准标注和校准文档 ID
tests/                  回归测试
legacy/                  早期脚本和配置，追踪用
datasets/               本地原始数据，Git 忽略
outputs/                 本地 corpus 与审核结果，Git 忽略
  history/               早期检查与审批证据
artifacts/               tokenizer、token 文件和未来模型，Git 忽略
```

## 当前状态

- 全量语料：`outputs/corpus-fast-v1/`，2,289,346 条来源记录、25,713,003 条唯一句子。
- 精确重复与导出完整性已检查；638 条规则隔离块待 review，语义抽检和近重复检查尚未完成。
- 旧 worker 与合并中间目录已清理。`integrity-check.json`、`cleanup-report.json`、`build-records/` 保存校验和构建依据。
- `outputs/corpus-pilot/`、`outputs/corpus-integration-v1/` 和 `artifacts/` 中的 pilot 模型保留；正式 tokenizer 配置是待确认模板，不自动执行。

Python 3.13 推荐，最低 3.11。数据环境依赖见 `requirements-data.txt`，tokenizer 另见 `requirements-tokenizer.txt`。所有命令都在仓库根目录执行，不需要安装项目包。

```powershell
# 快速确认现有 corpus；不会重读全部训练文件
.\.venv\Scripts\python.exe -X utf8 scripts/tools/check_corpus.py

# 下一步：只准备审核材料，不请求 API
.\.venv\Scripts\python.exe -X utf8 -u scripts/review/prepare.py --workers 8
```

逐步命令、API dry-run 和后续顺序见 [操作说明](docs/workflow.md)。每个入口支持 `--help`。默认使用当前语料，输出目录不能覆盖已有非空目录。

## 开发与复现

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
```

`src/vimeml/data/` 的核心和当前 `merge_parts_fast.py` 保留原路径与内容，使已有语料的指纹继续有效；日常操作用 `scripts/` 中的入口。新构建流程自动使用并行合并，分区 worker 和合并的临时目录分开保存。

Git 只保存代码、配置、测试、文档和少量审批台账。原始语料、生成结果、模型、虚拟环境、API key 留在本地；clone 后需另行复制数据。单账号使用环境变量 `SJTU_API_KEY`；独立多账号使用 `scripts/review/run_multi_key.py`，默认另读 `SJTU_API_KEY_2`，共享既有审核缓存。
