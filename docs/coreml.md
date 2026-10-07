# Core ML 部署

V1已有INT8接入与有限真机记录；V2.1 extend5 best（step40000）已完成训练和FP32评测，**尚未转换或量化**。2026-10-08的V2任务、交接包与Git方式见[Mac准备](mac-v21-preparation.md)。

## 接口与环境

目标接口：INT32 `input_ids [1,T]`，1≤T≤128；FLOAT32 `logits [1,T,16384]`，完整词表、无softmax。batch1、绝对位置、causal mask、右PAD、无KV cache；SentencePiece、联合分词评分和搜索在客户端。

Mac沿用独立Python3.11环境与`requirements.txt`的Darwin配置：torch2.7.1、coremltools9.0、numpy2.3.5。已有环境可复用，Windows不执行Apple运行时操作。

```bash
uv venv --python 3.11 venv/coreml
uv pip install --python venv/coreml/bin/python -r requirements.txt
source venv/coreml/bin/activate
```

## 工具与顺序

| 入口 | 范围 |
| --- | --- |
| `scripts/deployment/prepare_mac.py` | 当前Git与V2 FP32输入交接ZIP，不转换模型 |
| `scripts/deployment/coreml.py` | 推理bundle、reference、转换、对齐、候选/联想/计时；当前bundle仅支持V1 |
| `scripts/deployment/coreml_conservative.py` | V1混合权重存储、FP32计算和INT8 block32；V2精度/参数筛选需适配 |
| `scripts/deployment/package_ios.py` | 附带iOS参考的资源包，不是客户端构建或设备认证 |

V2先增加架构分派和独立bundle格式，完成未压缩转换与数值对齐，再量化和同池评测，最后接入Vime及真机测量。V2 tokenizer不同于V1，资源需整体更新。具体输入路径与参考结果都在[Mac准备](mac-v21-preparation.md)，不直接套用V1导出命令。

每个模型和报告使用新版本目录；旧checkpoint、tokenizer、manifest、失败实验与V1部署资源保持冻结。模型身份在首次导出或资源变更时绑定，之后复用记录；不重复全目录哈希或完整回归。新图与量化只做有关的长度、PAD/causal、分数及实际设备验证。

## 质量

V1参考路径为`artifacts/deployment/tiny-ja-v1-conservative-int8-b32-v1/`，INT8 block32权重、FP32计算、CPU_ONLY、最低iOS18，包8.08MB。

| V1检查 | 原结果 |
| --- | --- |
| 未压缩conservative对齐 | 通过 |
| INT8严格FP32 logits | 失败，最大绝对差约3.66 |
| 同池IME | dev122/137、AJIMEE124/200，与FP32命中数相同；完整顺序与部分分数不同 |
| 真实键盘扩展 | 约152.7秒记录；私有驻留采样峰值51.922MiB、footprint峰值21.25MiB，不能混用 |
| GPU／ANE | 当前路线CPU_ONLY；其他计算单元存在崩溃或性能问题 |

量化误差、排序质量、宿主性能与键盘扩展体验分别记录，旧结果不能替代V2验收。原环境、完整质量表、计时和局限见[V1详细实测](reference/coreml-v1.md)，设备证据见[真实扩展](reports/mac-20261006/iphone-keyboard-memory.md)。
