# TinySwallow-1.5B 本地候选重排对照

2026-10-16 在 RTX 5060 Laptop 8GB 上完成四组冻结开发／回归集，共 1,565 条。TinySwallow 在 Standard development 得到 527/728，相比 V2.1 增加 22 条、相比 V4-B 增加 29 条；旧 500 条与 AJIMEE 和 V4-B 打平，历史开发集增加 7 条。各组不作为相互独立的测试集累加。

## 模型与协议

使用官方 [SakanaAI/TinySwallow-1.5B](https://huggingface.co/SakanaAI/TinySwallow-1.5B)，固定 revision `535d1322615c9f8171c3d60b10ef6955cc896fe9`。模型说明记录其以 Qwen2.5-32B-Instruct 为教师、Qwen2.5-1.5B-Instruct 为学生，采用 TAID 并继续进行日语预训练。本轮选用普通 TinySwallow 版本；未使用其另行发布的 Instruct 版本。

实际加载 1,543,714,304 参数、151,936 词表。BF16 权重／forward、FP32 full-vocabulary cross entropy，无量化、无训练更新、无远程推理；使用 safetensors 与 Transformers 内置 Qwen2 实现，`trust_remote_code=False`。

主分数保持既有普通 LM 规则：原始 `context + candidate` 联合分词，在所有候选及原 context 的公共 token 前缀之后累计完整词表 log probability。不附加候选末尾 EOS，不截断，不加入任务提示、chat template、读音条件或答案。起始采用模型原生 BOS 151643。Mean 与空 context 只作为预先声明的辅助诊断，未根据结果改分数或调混合系数。

评分窗口仍为 128 个预测位置；没有使用模型原生 32K 上下文扩充左文。各 tokenizer 在同一窗口内覆盖的字符数可能不同，但输入文本、候选、全部可接受答案、稳定并列规则与整条回退规则沿用冻结协议。本轮所有四组均无窗口或 roundtrip 回退，完整分母保留。

候选最多八条一批。只提取需要评分的位置，每 128 个位置分块投影到完整词表，避免产生大量无用 logits；没有裁剪或重新归一化词表。CPU 检查覆盖分块与完整 logits 等价、不同候选长度、padding、边界重分词和分块大小。完成后逐条检查候选集合、输入及基线 hash、目标数、有限 token 分数与 logP 加总，共 1,565 条通过。

## 主结果

| 冻结数据组 | 样本 | AzooKey | V2.1 | V4-B | TinySwallow-1.5B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard development | 728 | 453 | 505 | 498 | **527** |
| Legacy regression | 500 | 317 | 374 | 377 | **377** |
| AJIMEE | 200 | 87 | 144 | 144 | **144** |
| Historical development | 137 | 111 | 122 | 120 | **127** |

表中均为原冻结参考答案下的 Top-1 正确条数。V2.1 为正式 extend5 基线；V4-B 为原 step-584,000 LM best。复用已经完成的 FP32 基线逐条评分，并验证其文件身份，本轮没有重新选择基线 checkpoint。

Standard development 的 TinySwallow 正确率为 72.39%，V2.1 为 69.37%，V4-B 为 68.41%；分别增加 3.02 和 3.98 个百分点。相对原 AzooKey 纠正 112 条、改坏 38 条，净增加 74 条。候选池覆盖正确答案的样本为 630/728，本轮没有生成或补入新候选。

| Standard 来源 | 样本 | V2.1 | V4-B | TinySwallow |
| --- | ---: | ---: | ---: | ---: |
| WRIME | 483 | 298 | 297 | **319** |
| JMultiWOZ | 245 | 207 | 201 | **208** |

相对 V2.1 的主要净改善来自 WRIME，JMultiWOZ 仅增加一条。相对 V4-B，WRIME 增加 22 条、JMultiWOZ 增加七条。这支持其在本批自然表达上的排序信号更强，但不等于所有输入域均有同等收益。

## 配对比较与辅助分数

以下为相同样本上“基线错误而 TinySwallow 正确／基线正确而 TinySwallow 错误”。p 为未经多重比较校正的双侧 exact McNemar 描述值，不作跨四组的确证性显著声明。

| 数据组 | 相对 V2.1 改善／退化 | p | 相对 V4-B 改善／退化 | p |
| --- | ---: | ---: | ---: | ---: |
| Standard development | 61 / 39 | 0.03520 | 61 / 32 | 0.00346 |
| Legacy regression | 18 / 15 | 0.72833 | 15 / 15 | 1 |
| AJIMEE | 14 / 14 | 1 | 13 / 13 | 1 |
| Historical development | 7 / 2 | 0.17969 | 9 / 2 | 0.06543 |

总分打平并不表示逐条结果相同。尤其 AJIMEE 相对 V2.1 同时改善和损失各 14 条，不能把教师输出直接视为正确标签。

| 数据组 | 有 context 的 sum，主分数 | 空 context 的 sum | 有 context 的 mean，辅助 |
| --- | ---: | ---: | ---: |
| Standard development | 527 | 504 | 345 |
| Legacy regression | 377 | 361 | 323 |
| AJIMEE | 144 | 139 | 127 |
| Historical development | 127 | 121 | 100 |

保留左文在这四组上均提高总正确数。Mean 在本轮明显较弱，结果不用于重新挑选主评分规则。部分逐条差异涉及「つらい／辛い」「なぜ／何故」等表记偏好，不能把每个答案命中变化都解释为同音语义理解变化；既有可接受答案保持不变。

## 本地资源与产物

| 项目 | 本轮实际记录 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8GB |
| PyTorch / CUDA | 2.10.0+cu128 |
| Transformers | 4.57.6，独立依赖目录 |
| 权重分配 | 2,945.28 MiB，约 2.88 GiB |
| PyTorch 峰值 allocated | 3,144.79 MiB，约 3.07 GiB |
| PyTorch 峰值 reserved | 3,486 MiB，约 3.40 GiB |
| 四组评测含模型加载 | 195.83 秒，不含下载、依赖准备及后续结果核对 |

这些是本地 PyTorch 分配器记录，不包含桌面显示和其他进程占用，也不是物理 footprint 或 iOS 延迟。原训练虚拟环境未安装或升级依赖；可选包在 `artifacts/external-eval/site-packages/`。远端 V4.1 未执行任何操作。

入口为 `scripts/benchmarks/evaluate_external_lm.py`，可选版本见 `requirements-external-eval.txt`，数学一致性检查为 `tests/test_external_lm_scoring.py`。官方模型及 tokenizer 文件保留在 `artifacts/models/external/TinySwallow-1.5B/`，本次记录在 `outputs/evaluation-tinyswallow-20261016/`：

- `model-source.json`、`plan.json`：固定 revision、模型文件及评测输入 SHA256、评分和精度协议。
- `environment.json`、`dependencies.json`：硬件和运行依赖。
- 各数据组 `scores.jsonl`、`metrics.json`、`paired-v21.json`、`paired-v4b.json`：候选及 token 分数、汇总、配对变化。
- `summary.json`、`acceptance.json`：完整结束与逐条核对结果。

## 解释边界

本轮没有读取 blind 或 test，没有修改参考答案，没有跑完整 LM validation BPC，也没有评估读音条件转换或指令式转换。外部模型与这些公开评测文本的预训练／蒸馏数据重叠未知；现有开发／回归标签及原候选快照的来源限制继续适用。

TinySwallow 的参数约为 V4 的 102 倍；数据、词表、训练流程和精度也不同，因此本轮不能用于估计单独增大参数量的因果收益。结果说明它有作为后续教师候选的研究价值，尤其是当前较弱的 WRIME；能否将其排序收益转移到 15M 模型仍需独立训练实验验证。
