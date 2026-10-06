# iOS参考与来源

`CoreMLProbe.swift` 是手动加载、预测与测量探针。`VimeSentencePiece/` 是官方SentencePiece0.2.1源码加C桥接，保留Apache-2.0及第三方许可，最低iOS18。`integration/` 是首次INT8候选重排接入的Swift代码与fixture快照。

`integration/` 早于Mac客户端后续的句内上下文、下一词联想、设置、取消与内存审计改动，不能把它当作当前Vime完整实现。最新客户端保存在本地 `handoff/mac-20261006-v1/Vime-client.zip`，含Git基点与patch；该客户端属于另一个仓库。早期 `artifacts/deployment/tiny-ja-v1-ios18-int8-release-v1/` 的Swift副本与README也是历史快照。

当前INT8包使用FP32计算、CPU_ONLY、iOS18。GPU/ALL路径曾触发量化gather断言；重新试验必须使用单独版本并手动验证。SentencePiece、联合分词评分、稳定排序与搜索均在应用侧实现，模型仅输出原始logits。

接入和实测限制见 [Core ML](../../docs/coreml.md)；操作与上传均不自动启动。
