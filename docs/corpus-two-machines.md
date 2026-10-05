# Windows + Mac 分开清洗

两个入口：`preprocess_part.py` 处理本机分配到的原文；`merge_parts.py` 汇总并统一分组、精确去重、划分 split、导出语料。现有 `build.py` 保留原样，可继续用于单机流程。

所有命令在仓库根目录执行。两台机器使用相同 Git commit、配置和审批台账；运行期间保持这些文件不变。

## 1. 把代码保存到 GitHub

当前远程仓库为 `https://github.com/Voltline/VimeML.git`。在 Windows 查看要提交的文件，再执行：

```powershell
git add .gitignore .gitattributes README.md src configs docs annotations tests requirements-data.txt requirements-tokenizer.txt
git diff --cached --stat
git commit -m "Add partitioned corpus preprocessing and global merge"
git push -u origin main
git rev-parse HEAD
```

本指南只提供命令，未替你 commit/push。记录最后的 commit ID，Mac 克隆后核对一致。

可提交：源码、配置、文档、依赖清单、测试及 `annotations/` 中的小台账。`annotations/calibration-documents-v1.json` 只包含曾用于调规则的文档 ID，审批台账只含四条公开语料片段及依据；它们取代 Mac 对旧 `outputs/` 审核文件的依赖。

原始 parquet/TSV、`outputs/`、`artifacts/`、`.venv/`、`.env` 文件已经忽略；它们不通过 GitHub 分发。这次清洗无需 API key。

## 2. 分配文件

两台使用同一个 `configs/corpus-sharded.toml`。其中写明全部输入的相对路径；`--part` 决定本机负责的文件，不需要分别修改配置。

| 机器 | 参数 | shard 编号 |
| --- | --- | --- |
| Windows | `--part 0/2` | 00010、00050、00090、00130、00170 |
| Mac | `--part 1/2` | 00030、00070、00110、00150，另加全部 Tatoeba |

这些编号对应完整文件名 `train-00030-of-00283.parquet` 等。两台机器分别只需要自己负责的文件；不分割单个 parquet，不拆文档。

Mac 中把四个 parquet 放入仓库内 `datasets/fineweb-2-edu-japanese/`，把 `jpn_sentences.tsv` 放入 `datasets/Tatoeba/`。用移动硬盘、局域网或 AirDrop 复制即可；保持相对路径和大小写一致。Windows 现有数据目录无需移动。

## 3. Windows 运行

当前 `.venv` 已有 pyarrow 25.0.1，可直接执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 src/vimeml/data/preprocess_part.py --part 0/2 --list-only
.\.venv\Scripts\python.exe -X utf8 src/vimeml/data/preprocess_part.py --part 0/2 --output outputs/corpus-parts/windows
```

## 4. Mac 从头运行

安装 uv（已有则跳过）；命令来自 [uv 安装文档](https://docs.astral.sh/uv/getting-started/installation/)：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

克隆代码并准备独立环境。uv 可以下载所需 Python，见 [Python 管理文档](https://docs.astral.sh/uv/guides/install-python/)：

```bash
git clone https://github.com/Voltline/VimeML.git
cd VimeML
git rev-parse HEAD
uv python install 3.13
uv venv --python 3.13
uv pip install --python .venv/bin/python -r requirements-data.txt
mkdir -p datasets/fineweb-2-edu-japanese datasets/Tatoeba
```

复制分配给 Mac 的原始文件后运行：

```bash
.venv/bin/python -X utf8 src/vimeml/data/preprocess_part.py --part 1/2 --list-only
.venv/bin/python -X utf8 src/vimeml/data/preprocess_part.py --part 1/2 --output outputs/corpus-parts/mac
```

若已克隆过，使用 `git pull` 获取已提交版本，再核对 commit；无需复制 Windows 的 `.venv`。如果 GitHub 仓库是私有的，克隆时使用你已配置的 GitHub 登录方式。

## 5. 拷回结果并汇总

等待两台输出成功完成信息及 `manifest.json`，再复制。把 Mac 的整个 `outputs/corpus-parts/mac/` 文件夹拷回 Windows 同一路径。文件夹含 `index.sqlite`、`part-stats.json`、三个审核 JSONL 和 `manifest.json`，须完整复制；不要只拷某个 TXT。

Windows 此时的结构：

```text
outputs/corpus-parts/
  windows/
  mac/
```

运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 src/vimeml/data/merge_parts.py --parts outputs/corpus-parts/windows outputs/corpus-parts/mac --output outputs/corpus-full-staging-v2
```

顺序不影响最终语料。汇总机只需代码、配置、审批台账和两个完成的结果目录，汇总时无需访问原始 parquet/TSV。

汇总会检查分片是否齐全、分工是否重叠、代码与标注版本是否一致，以及复制文件的 SHA-256。代码与审批文件按统一 UTF-8/LF 文本计算指纹，兼容 Windows/Mac 的 Git 换行差异；数据与结果按原始字节校验。随后统一恢复跨机器的相同文本、相同来源 ID 和相同 URL 文档分组；同一原文只计一次清洗指标，保留所有来源位置。审核过的文档组和其中相同句子固定在 train。

最终仍为 staging：近重复检查和代表性质量抽查尚待完成。验收文件是 `stats.json` 和成功完成的 `manifest.json`；完整输入的原文来源数应为 2,289,346，初始四条标注应全部匹配。

## 运行约束

- 每 5,000 条输出读取进度；汇总时另输出分组进度。哈希检查与最终导出也需要时间。
- 两个工作目录与最终目录必须为空或不存在；中断结果不支持续跑，重跑使用新的 `--output`。
- 全量运行期间不要修改源码、配置或审批台账；版本不一致会拒绝汇总。
- 此方案加速预处理，汇总仍在一台机器执行；中间文件体积与复制耗时也影响总耗时。
- 旧单机运行保持有效，但旧单机目录不是 `preprocessed_part`，不能当作新分片输入。

本地测试：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

分片回归包含：两份结果与单机的 train/validation/test、文档来源、句子 provenance 和统计一致；跨机器重复文档、来源 ID、URL 分组；仅有本机原始文件时预处理；汇总时无原始文件；缺失/重复分片、版本不一致与复制损坏的拒绝处理。测试在 Windows 执行，实际 Mac 环境仍需按上面的安装步骤验证。
