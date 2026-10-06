# Hugging Face 手动发布

目标仓库为 [Voltline/vimeml-tiny-ja-v1](https://huggingface.co/Voltline/vimeml-tiny-ja-v1)。作者已明确选择：**权重也采用GPL-2.0**。发布包的模型卡写入 `license: gpl-2.0`，根LICENSE与源码LICENSE保留完整条款。仓库网页当前未能通过浏览工具读取，因此不声称已验证远程可见性或内容；手动登录后由Hub工具检查现有仓库。

本地准备工具与上传工具分开。它们不训练模型、不运行Core ML转换／压缩、不操作设备，也不新建Git分支。原始checkpoint、tokenizer、manifest及评分实现保持不变。

## 发布目录

默认本地目录：`artifacts/releases/vimeml-tiny-ja-v1-hf-v1/`，不进入项目Git。内容为：

```text
README.md                       英文模型卡，含用途、数据、结构、结果与限制
LICENSE                         模型权重GPL-2.0
inference/                      完整原始FP32推理包（无optimizer）
coreml/ios18-int8-block32/        原始INT8 .mlpackage和manifest
source/{src,scripts,configs,...} 对应提交的模型、评分、转换与训练源码及统一requirements
infer.py                        使用原BundleLM和原评分函数的手动推理示例
evaluation/summary.json          重算核对的摘要与原始报告哈希
RELEASE.json                    固定源码提交、包身份与文件清单
SHA256SUMS.txt                   全部发布文件的哈希（清单自身除外）
verify_release.py               只使用Python标准库的校验器
```

源码快照只从已提交的 `src/ scripts/ configs/ templates/huggingface/ requirements.txt LICENSE` 取得。语料、annotations、训练checkpoint、optimizer、API回复、trace、虚拟环境、Mac完整handoff与另一个Vime客户端不上传。Core ML优先提供 `.mlpackage`，由Mac使用者手动编译；不默认发布 `.mlmodelc`。

模型卡模板在 `templates/huggingface/README.md`。它明确：自定义PyTorch模型不支持Transformers AutoModel；INT8权重、FP32计算、CPU_ONLY、iOS18；严格logits门槛失败但冻结候选池Top-1命中数保持；开发集对照基线与宿主／键盘的设备范围分别注明。

## 1. 准备并校验本地包

已经准备过v1目录时，直接运行verify；prepare拒绝覆盖任何已有目录。复现打包前先把审阅的源码正常提交到当前分支，使源码身份可固定；重新打包请改用新的输出版本。

```powershell
# 首次准备；若此目录已有完整发布包，不必再执行。
.\.venv\Scripts\python.exe -X utf8 scripts/publishing/prepare_hf.py prepare --output artifacts/releases/vimeml-tiny-ja-v1-hf-v1

.\.venv\Scripts\python.exe -X utf8 scripts/publishing/prepare_hf.py verify --release artifacts/releases/vimeml-tiny-ja-v1-hf-v1

# 只读FP32示例，调用发布包内的原始代码，不需要best.pt。
.\.venv\Scripts\python.exe -X utf8 artifacts/releases/vimeml-tiny-ja-v1-hf-v1/infer.py --prompt "今日は雨が降っているので、" --max-new-tokens 8
```

prepare检查原推理包全部文件与核心指纹、Core ML包身份，以及保存的开发集／AJIMEE评分哈希和metrics；重新核对完整排序差异，不运行新的Core ML推理。原始manifest按字节复制。复制完成后核对发布清单。若中途失败，保留现场，选新的输出版本处理。

`SHA256SUMS.txt`能检测字节损坏或清单外修改，不是数字签名。原manifest中绝对路径是来源记录，不改写为下载者的路径。源代码在 `source/` 下保留原目录关系，BundleLM核对这些源码的LF指纹。

## 2. 你手动安装发布依赖与登录

依赖仍统一在 `requirements.txt`，新增的 `huggingface_hub==2.1.1` 只在执行Hub步骤时使用。这里没有自动安装或登录。

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\hf.exe auth whoami
```

按登录工具提示使用浏览器，或在本地交互界面粘贴具备该仓库写权限的token；无需把token发到聊天中。[官方CLI登录说明](https://huggingface.co/docs/huggingface_hub/guides/cli)

## 3. 你手动上传

先阅读本地模型卡与清单，再执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/publishing/hub.py upload --release artifacts/releases/vimeml-tiny-ja-v1-hf-v1
```

工具先完整校验目录，再检查目标必须与 `RELEASE.json` 的repo_id相同，向现有model仓库的main提交**清单内**文件。不会调用create_repo、创建分支或删除远程文件。相同路径的README、LICENSE和资源会更新；它不把整个Windows项目目录递归上传。用当前远程commit作为parent，若上传期间远程发生变更则失败，避免悄悄覆盖并发更新。[官方提交API](https://huggingface.co/docs/huggingface_hub/guides/upload#create_commit)

完成后记录打印的 **hub_commit**（40位），这是Hugging Face提交，与VimeML源码提交不同。网页也可上传该目录内容，但上述工具能保证只上传已核对的文件。

## 4. 你手动下载验证

把下面的 `<hub_commit>` 换成上传工具打印的40位提交：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/publishing/hub.py download --revision <hub_commit> --output outputs/releases/vimeml-tiny-ja-v1-hf-download-v1
.\.venv\Scripts\python.exe -X utf8 outputs/releases/vimeml-tiny-ja-v1-hf-download-v1/infer.py --prompt "今日は雨が降っているので、" --max-new-tokens 8
```

download仅接受固定提交与新目录，并在下载后自动检查完整文件清单和SHA。Hub生成的 `.cache/`、可选根 `.gitattributes` 与本地Python字节码不当作模型payload。发布推理包的原始hash验证仍由BundleLM进行。如果想重跑Mac或iPhone验证，按 [Core ML指南](coreml.md) 单独手动启动，不把下载验证当作新设备验收。

## 来源与许可

权重的GPL-2.0是作者本次明确选择，项目源码继续采用原GPL-2.0；依赖和训练数据保留各自条款。[Hub支持的许可证标识](https://huggingface.co/docs/hub/repositories-licenses)

FineWeb2-Edu Japanese数据卡标注ODC-BY，Tatoeba文字默认CC-BY2.0FR并说明作者署名要求。模型卡标明来源，本次不再分发训练文本。AJIMEE只发布汇总指标和来源哈希，未附评测文本。数据许可不改写成权重GPL，来源记录继续保留。[FineWeb2-Edu Japanese数据卡](https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese/blob/main/README.md)、[Tatoeba条款](https://tatoeba.org/en/terms_of_use)

当前是自定义格式发布，不提供safetensors／Transformers适配；如以后增加，应在独立导出层验证共享权重和logits。完整失败实验与设备trace继续保存在本地，模型卡保留关键对照结论即可。
