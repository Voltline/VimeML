# Core ML 转换与量化计划

当前模型冻结，尚未实现转换或生成 `.mlpackage`。在 Mac 建立独立转换环境，记录实际 Python / PyTorch / coremltools 版本，不修改现有 Windows 训练环境。

## 实施顺序

1. 导出仅推理权重、模型配置、SentencePiece 与身份 hash，剥离 optimizer / RNG / 训练状态；当前 FP32 评分与文本输出为参照。
2. 转换 FP16 Core ML `.mlpackage`，先不压缩。核对 PyTorch 与 Core ML 的 logits、causal mask、位置 embedding、PAD 与特殊 token。
3. 对转换后的权重尝试 4bit 压缩，比较数值偏差、组合排序、联想、实际包体积和运行内存；FP16 / 8bit 作为质量或兼容性选择。无需先继续训练或做 QAT。
4. 在 iPhone 键盘扩展实测冷启动、单次候选重排、生成 3～5 项联想及内存峰值，再决定缓存与执行配置。

先把可验证的 FP16 模型转对，再压缩；避免同时引入图转换与压缩偏差。参考 [Apple Optimization Workflow](https://apple.github.io/coremltools/docs-guides/source/opt-workflow.html)。

## 模型与应用边界

首版 context≤128，当前 PyTorch 实现没有 KV cache。先验证整段 forward：生成需要末位置 logits，reranking 需要逐位置 logits。之后可优化输出/缓存，但必须保持评分语义。SentencePiece、文本解码、生成搜索、候选联合分词和组合评分在应用侧，Core ML 只执行 Transformer 图。

SentencePiece 接缝可能重新分词；不能把 context 与 candidate 分别 encode 再拼接。评分不额外加入 EOS，不改变候选池、错误回退或稳定并列规则。批量候选与 beam 路径的边界也需核对。

## 压缩与系统版本

可比较 grouped-channel palettization 和 blockwise 线性权重量化。palettization 是查表表示，不等于所有执行内核都使用 INT4 算术。配置与最低 iOS 版本共同选择：[Palettization 概览](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html)、[Quantization 概览](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html)。

基础 palettized ML Program 支持从 iOS16/macOS13 起；grouped-channel 配置要求 iOS18/macOS15 起。最低部署版本尚待确定，转换工具应明确记录 target，不能预先承诺所有 iPhone 版本可用。

7,386,624 参数纯 4bit 数据约 3.69MB，包内还有未压缩参数、LUT/scale 和结构；共享 embedding / LM head 可能在导出后重复存储，需要检查。磁盘大小不等于运行内存，不能直接保证最终包或键盘内存是 3.7MB。

## 验证与迁移

- Mac 对齐 logits / 每条候选分数与排序，覆盖 BOS、EOS、短句、长句、padding、byte fallback 和联合分词边界。
- 压缩后先跑独立 137 条开发集，必要时针对新模型重新校准 λ，固定后再测 200 条 AJIMEE。FP32 λ=2 和缓存不能直接当成压缩模型已验证策略。
- 联想对固定 20 个前缀保留原始结果并复核自然度，不只比较 loss 或文件大小。
- iPhone 的加载、延迟和内存单独测量；Mac predict 不替代目标设备验证。参考 [Core ML Getting Started](https://apple.github.io/coremltools/docs-guides/source/introductory-quickstart.html)。

新机器所需文件见 [本地产物与迁移](artifacts.md)。转换阶段无需复制整套原始语料或训练 token 二进制，保留 Windows 正式数据用于复现和追踪。
