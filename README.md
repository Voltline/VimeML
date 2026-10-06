# VimeML

Vime 日语输入法的本地 Tiny Japanese LM：AzooKey 提供假名转换检索，LM 按句内上下文重排候选，并提供短语／下一词联想。部署使用纯 LM 评分，原 λ=2 融合策略保留为历史对照。

v1 权重已冻结。Windows 的数据、训练和 FP32 基线，以及 Mac 的 Core ML 转换、量化、客户端接入与设备记录已完成归档合并。转换适配独立放在 `scripts/deployment/`；数据、模型、评分及推理核心保留原始指纹。

| 项目 | 当前记录 |
| --- | --- |
| 模型 | decoder-only GPT，7,386,624参数；vocab16384、context128、256维、4层、4头、FFN1024，共享 embedding/LM head，无KV cache |
| 训练 | 25,713,003条唯一句子；1 epoch；完整validation loss4.5416、PPL93.84；test未用于调参 |
| FP32 / INT8纯LM | 开发集Top-1均122/137；AJIMEE均124/200，候选池覆盖162/200 |
| 当前客户端资源 | INT8 block32权重、FP32计算、CPU_ONLY、最低iOS18；`.mlpackage` 8,077,801字节 |
| 验证范围 | INT8严格logits门槛失败，排序命中数保持；完整顺序有变化。真实键盘有约2.5分钟手动记录，不能代替长期稳定性验收 |

模型和报告是本地产物，不在Git。新机器按 [产物与迁移](docs/artifacts.md) 复制；核心文档入口见 [文档索引](docs/index.md)。

## 文档

| 入口 | 内容 |
| --- | --- |
| [数据](docs/data.md) | 来源、清洗、分句、去重与审核 |
| [训练与推理](docs/training.md) | 环境、SentencePiece、冻结模型与复现参考 |
| [评测](docs/evaluation.md) | 评分规则、开发集、AJIMEE、融合策略与联想限制 |
| [Core ML](docs/coreml.md) | 手动转换、压缩、验证、iOS打包及实测结论 |
| [本地产物](docs/artifacts.md) | 文件保存、迁移、原始快照与校验 |

## 本地使用

Python≥3.11。Windows已有CUDA环境安装 `requirements.txt`；新环境按 [训练环境](docs/training.md#环境) 先选择PyTorch版本。Mac的Core ML依赖也由同一文件管理。

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 scripts/training/infer.py
.\.venv\Scripts\python.exe -X utf8 -u scripts/tools/phrase_demo.py
.\.venv\Scripts\python.exe -X utf8 scripts/tools/check_corpus.py
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
```

联想网页为 `http://127.0.0.1:8765/`。所有真实训练、转换、压缩和设备操作均由使用者手动启动，工具合并不自动执行它们。

## 目录

```text
src/vimeml/       原实现与冻结推理包支持（保持指纹）
scripts/          按 corpus/review/tokenizer/training/benchmarks/tools/deployment 分类的入口
examples/ios/     Core ML探针、SentencePiece运行时与首次Swift集成参考
configs/          版本配置和固定联想前缀
tests/            回归测试
annotations/      清洗批准依据
docs/             当前指南；reference/详细基线；reports/实测；history/历史计划
datasets/         本地原始数据
artifacts/        本地模型、tokenizer、候选和部署包
outputs/          本地语料、评测、trace、历史与维护记录
runs/             本地训练事件
handoff/          本地Mac快照与另一个Vime仓库的交接ZIP
```

Git保存代码、配置、测试、文档与批准记录。模型、语料、trace、交接ZIP、虚拟环境和密钥不提交。
