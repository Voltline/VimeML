# Hugging Face 发布草案

v1 可以作为自定义PyTorch模型与Core ML资源发布。Hub允许任意模型库；无需为了上传先改写成Transformers，也不能标注已支持 `AutoModel.from_pretrained`。[官方上传说明](https://huggingface.co/docs/hub/models-uploading)

本页仅准备发布方案；未创建远程仓库、登录或上传。建议先建私有model仓库，核对发布文件、模型卡和许可后再公开。仓库名可用 `<账号或组织>/vimeml-tiny-ja-v1`，具体命名与公开时间由作者决定。

## 拟发布内容

```text
README.md                     独立模型卡，区别于项目README
inference/                    完整tiny-ja-v1-inference-v1（权重、配置、tokenizer、manifest）
coreml/ios18-int8-block32/      当前.mlpackage和原manifest
evaluation/summary.json       小型开发集/AJIMEE结果与来源哈希，不打包所有trace
SHA256SUMS.txt                 所有发布文件的哈希
LICENSE                       明确选择的模型权重许可
```

推理包的manifest和所有被它校验的文件应原样保留；下载后使用固定Git提交的VimeML代码加载。完整 `best.pt` 含训练状态，不作为默认下载；训练语料、token二进制、API回复、设备trace、密钥、Mac全量handoff和另一个Vime客户端均不上传。`.mlmodelc` 可作为额外平台资源，但应注明编译环境，优先提供可重新编译的 `.mlpackage`。

第一版可发布现有 `weights.pt`，同时注明安全加载方式 `weights_only=True`。如以后提供safetensors或Transformers适配，应在独立导出／适配层实现，重新验证共享权重与logits，不改冻结核心来迁就发布格式。

## 模型卡

至少写清参数量、结构、context128、SentencePiece词表与特殊ID、共享head、无KV cache、版本哈希、用途与加载步骤。[官方模型卡说明](https://huggingface.co/docs/hub/model-cards)

用途是日语候选重排和句内联想；假名检索和搜索由应用侧完成。它不是聊天／指令模型。列出训练来源、清洗与拆分、完整validation、开发集与AJIMEE指标，说明已做错误分析、公开文本重叠未全面审计、开发集为AI标签与复核、候选池覆盖上限，以及生成的重复、截断和事实可靠性限制。

INT8必须单独注明权重INT8、计算FP32、CPU_ONLY、iOS18、8,077,801逻辑字节；严格logits门槛失败但离线Top-1命中数保持。将宿主App性能、真实扩展内存采样和未完成的长期验证分开，不能把宿主峰值或2.5分钟记录写成键盘的完整性能认证。

## 许可与署名

项目代码当前是GPL-2.0文本；发布权重使用什么许可需作者明确决定，不能由代码LICENSE自动推定。SentencePiece运行时及第三方许可随相关源码保留。[Hub许可元数据说明](https://huggingface.co/docs/hub/repositories-licenses)

训练数据来源和发布时需要保留的署名依据应核对实际下载版本：FineWeb2-Edu Japanese数据卡标注ODC-BY；Tatoeba文字默认CC-BY2.0FR，并说明按句子作者署名要求。数据许可不自动等于模型权重许可，本页不替作者选择。[FineWeb2-Edu Japanese数据卡](https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese/blob/main/README.md)、[Tatoeba使用条款](https://tatoeba.org/en/terms_of_use)

如果只发布自己的汇总指标，无需打包AJIMEE原文；如随包再分发公开评测文本，保留其CC-BY-SA3.0来源说明。引用固定版本和数据来源，避免把整个网站语料作为模型附件上传。

## 手动发布顺序

1. 确定账号／组织、仓库名、私有或公开、模型权重许可；定稿独立模型卡。
2. 从冻结包复制到新的release目录，生成清单并检查文件范围与哈希，保持本地产物不动。
3. 在Hub创建model仓库，通过网页上传，或在单独工具环境使用 `huggingface_hub` 的 `upload_folder`；不向当前训练环境随意添加发布依赖。[官方文件夹上传指南](https://huggingface.co/docs/huggingface_hub/guides/upload)
4. 从干净目录下载固定revision，使用对应VimeML提交验证bundle和模型包，手动复跑小型推理fixture；公开前完成这次下载验证。
5. 给release打版本并在模型卡记录VimeML提交、bundle与Core ML manifest哈希。

现阶段不需要上传所有未压缩和失败实验。公开模型卡保留关键对照结论即可，完整实验记录继续在本地归档。
