# V1部署设计（实测前）

2026-10-06实测前的设计记录。模型为7,386,624参数的V1，原方案先验证未压缩Core ML，再比较压缩与设备成本；实际采用的INT8路线见[V1实测](../reference/coreml-v1.md)，V2结论见[实验总结](../reports/v2-summary.md)。

## 模型与应用边界

AzooKey检索假名候选，LM按contextual logP sum重排；SentencePiece、评分、文本解码和搜索位于应用侧，Core ML只执行Transformer图。取消词典检索需要独立的读音约束生成任务，原GPT训练目标不覆盖该任务。

context≤128，无KV cache。生成使用末位置logits，重排使用逐位置logits；context与candidate联合编码，公共token前缀后累计完整词表logP，不加入候选EOS。整池回退和稳定并列规则与FP32评测一致。

## 原实验路径

1. 导出推理权重、配置、tokenizer及来源manifest，剥离optimizer/RNG。
2. 转换未压缩FP16包，验证logits、causal、位置、PAD与特殊token。
3. 比较4bit palettization、线性量化及8bit对照，分别记录数值、排序、联想、包体与内存。
4. 在iPhone测试加载、重排、联想和真实扩展footprint，选择执行配置。

纯4bit参数的理论体积约3.69MB，不包含LUT/scale、未压缩常量和可能重复的共享权重。存储位数不等于执行算术精度，包体也不等于运行内存。分组压缩和最低系统版本共同确定。

## 评估约定

原137条开发集、AJIMEE200与20个联想前缀采用固定输入和评分方式。纯LM为部署主线，历史λ=2融合策略单独保留；新模型的融合系数需要独立开发集校准。

Mac预测计时、测试宿主与真实键盘扩展分别统计。数值复现、任务质量和设备成本属于不同指标，压缩收益以实际实验为准。

方法依据：[优化流程](https://apple.github.io/coremltools/docs-guides/source/opt-workflow.html)、[Palettization](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html)、[Quantization](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html)。原命令入口见[V1方法参考](coreml-guide.md)。
