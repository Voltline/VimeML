# vimeml-tiny-zh-v1 Mac 交接结果核对与 iOS 发布（2026-11-11）

Mac 端完成了 Core ML 转换、INT8 量化、量化方案调查和 Mac 实测，过程和全部数字见 [Mac 报告](../../../reports/mac-20261111/zh-v1-coreml.md)（英文）。本文做三件事：在 Windows 上独立核对交接包，分析结果，记录发布到 Hugging Face 的内容。这一版将部署到 iOS。

交接包 `vimeml-zh-v1-mac-results-20261111.zip`，SHA256 `0a97da02650bbbd9a4e32f4ec8f389047c4026b29666470649f9cc7af191b3bf`，171 个文件。只用到 development；blind 没有再读。

## 结论

- **iOS 用 INT8 block32**（只量化权重，FP32 计算，CPU_ONLY，iOS 18 起），有无状态和分块 KV 两个包。和 FP32 相比，development 全拼 Top-1 +2/−1，简拼 +0/−2，Top-3 两条轨道都没有变化，差别在噪声范围内。
- **交接包核对通过**：文件大小、bundle 哈希、权重与训练 checkpoint 逐张量一致，Mac 的逐题结果和 Windows 的逐题排名一致（见下一节）。
- **iOS 上线前还缺真机数据**：iPhone 和键盘扩展都还没测。Mac 上 zh-v1 的体积、延迟和模型自身的内存增量都约为 V4.2 的 1.3–1.4 倍。V4.2 在 iPhone 测试宿主上的 `predict` T=128 p50 是 4.90 ms，按同样比例推算 zh-v1 约 6.5 ms。这只是估计，要以实测为准。
- **训练数据许可有一处需要决定**：LCCC 的仓库和 Hugging Face 数据卡标的是 MIT，但仓库 README 的免责声明写明数据集和预训练对话模型“仅限科研用途”。详见“许可”一节。

## Windows 端独立核对

| 检查 | 结果 |
| --- | --- |
| 交接包 171 个文件的字节数与 `handoff.json` | 全部一致 |
| 推理 bundle 三个文件的 SHA256 与 bundle 清单 | 全部一致；清单 SHA256 `9d9dad19…`，与两个 Core ML 包、iOS 资源清单里记录的相同 |
| bundle 权重与训练 checkpoint（`final.pt`，`44c423a1…`） | 66 个张量逐个相等；`lm_head` 与词嵌入共享，不单独存；参数 20,036,736 |
| 从逐题结果重算 Top-1 / Top-3 | 8 个包、两条轨道都与 Mac 报告相同 |
| Mac 逐题排名与 Windows CUDA FP32（判定时的打分） | FP32 两条轨道每题 rank 完全相同（1,456 + 990 题）；INT8 只有 3 + 3 题 rank 不同，变动题目的 id 与 Mac 报告逐个一致 |
| 交接包里的环境信息 | 只有 Mac 本机信息，无训练基础设施信息；3 个记录文件含 Mac 用户目录路径，这些文件不进发布包 |

## 结果分析

### 量化对排序的影响

参照为 Core ML FP32（与 Mac PyTorch、Windows CUDA 的 Top-1 都是 1,363 / 337）。

| 包 | 体积 | 全拼 Top-1（1,456） | 简拼 Top-1（990） | 对 FP32 多对 / 多错 | 结论 |
| --- | ---: | ---: | ---: | --- | --- |
| FP32 | 80.4 MB | 1,363 | 337 | — | 基准 |
| **INT8 b32（部署）** | **22.8 MB** | 1,364 | 335 | 全拼 +2/−1，简拼 +0/−2 | 无显著差异，Top-3 不变 |
| INT8 b32 + FP16 词嵌入 | 26.9 MB | 1,365 | 337 | +2/−0，+0/−0 | 无显著差异，大 18% |
| 调色板 6 bit | 15.7 MB | 1,364 | 339 | +7/−6，+4/−2 | Top-1 持平，但排序变动多 |
| INT4 b32 | 12.8 MB | 1,349 | 332 | +10/−24，+8/−13 | 全拼显著变差（p = 0.024） |
| 调色板 4 bit | 10.4 MB | 1,320 | 316 | +16/−59，+17/−38 | 显著变差（p = 6e-7） |

- INT8 的严格 logits 门槛照例不通过（最大误差 0.392，V4.2 是 0.477），这是有损量化的预期漂移，不是数值等价。分块 KV INT8 和无状态 INT8 在 21 项检查里最大差 4.2e-5，排序完全相同，所以两个 INT8 包可以互换。
- 6 bit 调色板的 Top-1 和 FP32 持平，但改变了全拼 1,424 题、简拼 983 题的候选顺序，第一候选变化 15 + 62 题，比 INT8 的 4 + 17 题多得多。只看 Top-1 会低估它的误差。用在 iOS 上还需要单独评估。
- 这次调查按计划只记录数字，不换方案。如果以后在意 18% 的体积，可以考虑 FP16 词嵌入；如果在意更小的体积，6 bit 调色板值得先在联想基准上看看。

### Mac 性能（与 V4.2 同机、交错测量）

| 指标 | zh-v1 INT8 | V4.2 INT8 | 比值 |
| --- | ---: | ---: | ---: |
| 包体积 | 22.8 MB | 17.2 MB | 1.33× |
| 无状态 `predict` p50，T=1 / T=128 | 1.65 / 8.49 ms | 1.29 / 6.43 ms | 1.28× / 1.32× |
| 分块 KV 预填 119 token / 单步解码 p50 | 9.33 / 2.71 ms | 7.18 / 2.05 ms | 1.30× / 1.32× |
| 模型自身的峰值内存增量 | 18.4 MiB | 13.4 MiB | 1.37× |

比值和词表（12,288 对 8,192）、层数（9 对 7）的增加相符。INT8 只缩小体积，不降低 CPU 延迟，因为计算仍是 FP32。

### 上线前要补的测量

- iPhone 16 Pro Max 测试宿主：无状态 INT8 的 `predict` 和候选重排延迟，与 V4.2 交替四轮。Vime 端的 `.zhV1` 变体由 Vime 维护者接入后，用现有的 `benchmark_lm_variants.py --variants zhV1,v42` 即可。
- 键盘扩展：常驻内存和真实打字时的延迟。扩展的内存上限很紧，zh-v1 的模型内存约比 V4.2 多 5 MiB，需要实测确认余量。
- 模拟器上的功能检查已经通过（4/4：资源身份、logits 形状、原生 SentencePiece 分词、联合打分与 Mac INT8 相差不到 0.002），但这不能代替真机测量。

## 发布内容

Hugging Face 仓库 `Voltline/vimeml-tiny-zh-v1`，格式与日文 V4.2 的发布相同：

| 路径 | 内容 |
| --- | --- |
| `inference/` | FP32 权重（不含优化器）、配置、tokenizer、bundle 清单 |
| `coreml/ios18-int8-block32/` | 无状态 INT8 Core ML 包和清单 |
| `coreml/ios18-int8-block32-kvchunk/` | 分块 KV INT8 Core ML 包和清单 |
| `source/` | 发布时 commit 的仓库源码快照（只含已跟踪文件） |
| `infer.py` | FP32 候选打分和联想示例 |
| `evaluation/summary.json` | 只有汇总数字，不含题目文本、候选或题目 id |
| `RELEASE.json`、`SHA256SUMS.txt`、`verify_release.py` | 清单与校验 |
| `LICENSE`、`LICENSE-EXCEPTION`、`NOTICE.md` | GPL-3.0-or-later、App Store 附加许可、训练数据署名 |

模型卡和发布包都不写训练基础设施的任何信息，只写“单卡训练”。

## 许可

- 权重和代码沿用项目当前的 GPL-3.0-or-later 加 App Store 附加许可，与日文 V4.2 现在的发布一致。
- FineWeb2 `cmn_Hani`：ODC-By 1.0，同时受 Common Crawl 使用条款约束。
- **LCCC-base**（thu-coai，CDial-GPT 仓库）：仓库 LICENSE 和 Hugging Face 数据卡都是 MIT；但 README 的免责声明写着“本项目所提供的 LCCC 数据集和预训练对话模型仅限科研用途”。两处说法不一致，README 那句针对的是用途。LCCC 只占训练监督 token 的 5%，但模型确实在它上面训练过。是否在 iOS 上线前向作者确认，由项目负责人决定；模型卡和 NOTICE 会如实写出这两处条款。
- 评测来源（CrossWOZ、RiSAWOZ、KdConv、CPED）不进权重，发布包只含汇总数字。
