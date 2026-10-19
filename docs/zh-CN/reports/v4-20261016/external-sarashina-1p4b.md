# Sarashina2.2-1B 本地候选重排对照

2026-10-16 在 RTX 5060 Laptop 8GB 上完成四组冻结开发／回归集，共 1,565 条。Sarashina 的 Standard development 为 518/728，比 V2.1 增加 13 条，比 TinySwallow 少九条；其余三组分别比 TinySwallow 多四条、六条、一条。结果支持保留两个教师候选，尚不足以认定某一个全面更强。

## 模型和协议

使用官方 [sbintuitions/sarashina2.2-1b](https://huggingface.co/sbintuitions/sarashina2.2-1b) 基座，固定 revision `3bb836ad7475ba192926be66651e4730825df7da`。官方“1B”统计不含 embedding 和输出头；实际加载总参数 **1,407,538,944**，102,400 词表、24 层、hidden 1792、原生 BOS 1。模型不是另行发布的 Instruct 版本。

权重 SHA256 为 `d23ae09c0c8e1f161bea04640359f27f706514016b65a22ad726e8b0b1624420`，与官方发布指针一致。仅加载 safetensors 和 Transformers 内置 Llama 实现，`trust_remote_code=False`。Tokenizer 的日文、空格、换行和全角字符 roundtrip 兼容性已检查。

沿用 [TinySwallow](external-1p5b.md) 与 [Rakuten](external-rakuten-1p5b.md) 的原始文本概率协议：`context + candidate` 联合分词，公共 token 前缀之后 full-vocabulary logP sum；原生 BOS 开始，不添加候选末尾 EOS、chat template、任务指令或 READ 条件。窗口仍为 128，未扩充到模型原生 8,192；不裁剪、稳定并列、整条回退和完整分母均保留。

BF16 权重及 forward、FP32 CE，无量化和训练更新。候选八条一批，只对评分位置以 128 个位置为单位投影完整词表；评分公式未改变。Mean 和空左文仅为预先声明的辅助诊断，没有根据分数调参。

## 完整结果

表中为 Top-1 正确条数。V2.1 为正式 extend5 基线，V4-B 为 step-584,000 LM best；全部复用身份已核对的历史逐条结果。

| 数据组 | 样本 | AzooKey | V2.1 | V4-B | TinySwallow | Sarashina |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard development | 728 | 453 | 505 | 498 | **527** | 518 |
| Legacy regression | 500 | 317 | 374 | 377 | 377 | **381** |
| AJIMEE | 200 | 87 | 144 | 144 | 144 | **150** |
| Historical development | 137 | 111 | 122 | 120 | 127 | **128** |

四组不作为相互独立的测试样本累加。Standard 正确率为 71.15%，比 V2.1 增加 1.79 个百分点、比 TinySwallow 低 1.24 个百分点。相对原 AzooKey 纠正 108 条、改坏 43 条，净增 65 条。候选池覆盖正确答案的样本仍为 630/728，本轮未生成或补入候选。

| Standard 来源 | 样本 | V2.1 | TinySwallow | Sarashina |
| --- | ---: | ---: | ---: | ---: |
| WRIME | 483 | 298 | 319 | 313 |
| JMultiWOZ | 245 | 207 | 208 | 205 |

相对 V2.1 的净改善来自 WRIME 的增加 15 条，JMultiWOZ 少两条。外部日语模型在本批自然表达上的排序信号具有研究价值，但这些差异不能单独归因于参数量、训练数据或架构。

## 配对比较

以下为“原模型错误而 Sarashina 正确／原模型正确而 Sarashina 错误”，p 是未经多重比较校正的双侧 exact McNemar 描述值。

| 数据组 | 相对 V2.1 改善／退化 | p | 相对 TinySwallow 改善／退化 | p |
| --- | ---: | ---: | ---: | ---: |
| Standard development | 58 / 45 | 0.23690 | 32 / 41 | 0.34918 |
| Legacy regression | 23 / 16 | 0.33678 | 22 / 18 | 0.63583 |
| AJIMEE | 13 / 7 | 0.26318 | 10 / 4 | 0.17957 |
| Historical development | 7 / 1 | 0.07031 | 5 / 4 | 1 |

不能仅凭九条总分差认定 TinySwallow 稳定更强，也不能凭其余三组的点估计认定 Sarashina 稳定更强。Standard 上 Sarashina 能纠正 TinySwallow 的 32 条错误，同时又损失其 41 条正确样本，说明逐条排序信号有所不同；本轮没有利用答案调融合权重或训练选择器。

| 数据组 | 原左文 sum，主分数 | 空左文 sum | 原左文 mean，辅助 |
| --- | ---: | ---: | ---: |
| Standard development | 518 | 492 | 297 |
| Legacy regression | 381 | 349 | 298 |
| AJIMEE | 150 | 139 | 105 |
| Historical development | 128 | 125 | 40 |

## 资源、产物与边界

完整评测含模型加载用时 **158.97 秒**，不含下载、依赖准备和结果核对。PyTorch 模型加载后的 allocated 为 2,725.17 MiB，峰值 allocated 为 2,861.55 MiB，峰值 reserved 为 3,174 MiB，约 3.10 GiB。它们是分配器记录，不包含桌面及其他进程，不是物理 footprint 或 iOS 延迟。运行环境为 PyTorch 2.10.0+cu128、Transformers 4.57.6，复用隔离的外部模型依赖。

入口 `scripts/benchmarks/evaluate_external_lm.py` 增加了显式固定的 Sarashina 注册项，评分实现未改。源码快照、来源和模型文件 hash、协议、环境、依赖、逐 token 分数、逐条配对结果及检查记录保存在 `outputs/evaluation-sarashina-20261016/`。`acceptance.json` 确认全部 1,565 条候选集合、输入及三套基线 hash、监督位置数量、有限分数和 logP 加总检查通过；四组均无窗口或 roundtrip 回退。

没有读取 test 或 blind、修改参考答案、运行训练更新或操作远端 V4.1。未评估完整 LM validation BPC、读音条件或指令式转换。外部预训练文本与公开 benchmark 的重叠未知，既有开发／回归标签及快照来源限制继续适用。这是教师候选筛选结果，向 15M 模型转移收益仍需独立蒸馏实验验证。
