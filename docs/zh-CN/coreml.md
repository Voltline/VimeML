# Core ML 部署

当前部署版本为V4.2第20轮的INT8 block32模型（见下文V4.2一段）。上一部署版本为V2.1 extend5 best（step40000）的INT8 block32模型，已完成Core ML转换、同池评测和iPhone真实扩展验证；V1保留作对照与回退。结果见[V2量化](reports/mac-20261008/v21-coreml.md)与[真实扩展](reports/mac-20261008/v21-iphone.md)，源码与产物迁移见[跨平台管理](reference/artifact-exchange.md)。

V2未压缩对齐通过，INT8严格logits对齐失败；AJIMEE144/200、原dev122/137保持命中数，扩大草稿1487→1481/2000。INT8包14.33MB，真实扩展内核footprint峰值34.72MiB、私有驻留采样峰值73.78MiB；UI发布最大约1.93秒，原因未定位。代码合并和本次候选验证不等于正式发布或长期设备验收。

V4.2第20轮已完成FP32/INT8的stateless、单token KV与分段KV转换，并在Vime中通过模拟器与iPhone 16 Pro Max功能测试，现已替换V2.1成为部署版本。V4.2 INT8包17.16 MB，严格对齐失败（T=128最大0.477、平均0.0495，约为V2.1 INT8的一半），四组Top-1为522/371/139/123（V2.1 INT8同环境506/375/144/122）。iPhone测试宿主中predict每次慢0.16–0.35 ms，重排p95 120.9对136.8 ms，峰值footprint 44.35对49.95 MB；键盘扩展未测。详见[V4.2 Core ML报告（英文）](../reports/mac-20261020/v42-coreml.md)。

V2系列实验完成，当前INT8质量损失和UI长尾可接受；数值对齐失败仍按原阈值记录。量化机制、同池分差与配对统计见[量化分析](reports/mac-20261008/v21-quantization-review.md)。

## 接口与环境

目标接口：INT32 `input_ids [1,T]`，1≤T≤128；FLOAT32 `logits [1,T,16384]`（V2.1）或`[1,T,8192]`（V4.2），完整词表、无softmax。batch1、绝对位置、causal mask、右PAD；客户端使用stateless模型，KV cache图（V2.1分段、V4.2单token与分段）为已验证的研究导出，调用方持有FP32快照。SentencePiece、联合分词评分和搜索在客户端；V4词表的ID 4–6为保留标记，输入须拒绝、生成须排除。

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
| `scripts/deployment/coreml_v2.py` / `package_v2_ios.py` | 独立V2推理bundle、FP32转换、门控INT8及客户端资源 |
| `scripts/deployment/prepare_mac_v42.py` | V4.2 Git bundle、推理权重、PyTorch fixtures与冻结评分交接包 |
| `scripts/deployment/coreml_v4.py` / `package_v4_ios.py` | V4 bundle、stateless／单token KV／分段KV转换、门控INT8、四组评分及客户端资源 |
| `scripts/deployment/measure_v42.py` / `summarize_v42.py` | V4.2与V2.1同一Mac测试环境的大小、加载、延迟与内存测量及汇总 |
| `scripts/deployment/coreml_conservative.py` | V1混合权重存储、FP32计算和INT8 block32 |
| `scripts/deployment/package_ios.py` | 附带iOS参考的资源包，不是客户端构建或设备认证 |

V2使用独立bundle与导出入口。V2 tokenizer不同于V1，客户端模型与tokenizer成套更新；实际产物和调用记录见[V2量化报告](reports/mac-20261008/v21-coreml.md)。

每个模型和报告使用版本目录，保留原checkpoint、tokenizer、manifest与失败结果。转换验证覆盖长度、PAD/causal和数值复现；量化另评估同池任务质量，客户端另测分词、评分与实际设备行为。

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
