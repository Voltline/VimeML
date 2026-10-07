# V2.1 Mac 准备与双仓库协作

计划日期：2026-10-08。模型为 **V2.1 extend5 best / step40000**，12,537,920参数。今天只准备推理输入、参考结果和交接包；Core ML 适配、转换、量化及设备验证在Mac执行。

## ZIP与启动

交接包：`handoff/vimeml-v21-mac-20261008.zip`。解压后在该目录执行：

```bash
python3 setup_mac.py
# 指定其他新目录：python3 setup_mac.py ~/Documents/Sources/VimeML-v21-coreml
```

入口从包内`repository.bundle`建立完整Git仓库，绑定GitHub origin，在`codex/mac-v21-coreml`分支工作，再复制忽略目录中的模型与评测资料。默认目标为`~/Documents/Sources/VimeML-v21-coreml`；已有目录不覆盖。旧Mac工作区可继续保留，不需整目录手工合并。源码基点以包内`handoff-manifest.json`的`source_commit`为准。

| 包内材料 | 用途 |
| --- | --- |
| `repository.bundle` | 本次已提交代码、文档、iOS参考及Git历史 |
| `payload/artifacts/models/tiny-ja-v2.1-e16k-d320-l6-extend5/deployment.pt` | 原best的相同FP32模型tensor；不含optimizer/RNG，不能恢复训练 |
| V2 tokenizer与原token manifest | 保留精确token ID；不需要token store、训练语料或索引 |
| AJIMEE200、原development137、扩大development2000 | 冻结候选、标签与同池量化比较 |
| FP32逐候选评分、12条续写、5组logits fixture | 复用Windows参考，降低重复计分 |
| V1 INT8包与V1 tokenizer | 原部署对照；与V2词表分开 |

未包含1000条blind、训练数据、优化器、SSH/W&B/API凭据、虚拟环境、Mac旧客户端快照或失败实验全量备份。包内只有准备材料，没有已转换的V2 Core ML模型。

## 模型与适配范围

| 项目 | 约定 |
| --- | --- |
| 架构 | 320×6、5heads、SwiGLU832、RMSNorm eps1e-5、无linear bias、共享embedding/LM head |
| 输入／输出目标 | `input_ids` INT32 `[1,T]`，1≤T≤128；`logits` FLOAT32 `[1,T,16384]`，完整词表、无softmax |
| 状态与评分 | batch1、无KV cache、右PAD；联合编码context+candidate，公共token前缀后logP sum，不加EOS、不截断 |
| 冻结结果 | BPC3.0802686；AJIMEE144/200；原dev122/137；扩大草稿1487/2000 |

**V2 tokenizer不同于V1**，即使词表尺寸同为16384也不能沿用V1 token ID。客户端必须同步模型、tokenizer、manifest和fixture；不能只换权重。

现有`src/vimeml/deployment/bundle.py`仅接受V1，`BundleLM`也固定使用`GPTConfig/TinyGPT`；`coreml_conservative.py`含V1精度边界假设。Mac首先增加V2架构分派与新的推理bundle格式，再检查显式attention适配器、RMSNorm、SwiGLU、权重共享和压缩参数筛选。原模型可用`model_factory.model_from_checkpoint`加载`deployment.pt`。新增V2路径，不通过改旧manifest或伪装为V1绕过检查。

Mac环境沿用`requirements.txt`的Darwin配置（torch2.7.1、coremltools9.0、numpy2.3.5），优先复用昨天已验证的环境。V1接口、失败记录与发布资源保持冻结。

## 明天的顺序与验收

1. 适配V2推理bundle和FP32转换；以`outputs/deployment/tiny-ja-v2.1-extend5-fp32-input/{reference.json,logits.npz}`检查长度1/16/128、右PAD和未来token不影响有效前缀。JSON记录的是Windows PyTorch参考，不是已通过的Core ML对齐。
2. 未压缩转换对齐后再做INT8 block32，先用FP32计算、CPU_ONLY、最低iOS18作为已有路线的实验起点；V2是否适用由实际结果决定。本轮优先完成一种量化方案。
3. 复用包内FP32候选分数，报告量化前后Top-1、分歧、分数误差及固定续写。2000条标签仍为草稿；不修改标签、不借后验别名提高结果，不使用blind选模型。INT8严格logits误差和实际排序质量分别记录。
4. Vime客户端以Mac现有仓库为准，更新资源并检查分词、取消/回退、冷启动、排序、下一词以及真实键盘扩展内存/延迟。测试宿主与扩展、驻留内存与footprint分开报告。达标后形成独立V2发布候选，V1用于回退。

每种转换／压缩使用新的`artifacts/deployment/tiny-ja-v2.1-*/`目录，报告放`outputs/deployment/v21-*/`。只做改动相关的数值对齐、同池量化评测和必要真机检查；不重复完整Python回归、训练、旧模型冒烟或全目录SHA256。关键模型／tokenizer身份在首次导出或资源变更时绑定一次，后续复用既有记录。

## Git与产物回传

| 仓库 | 分支与职责 |
| --- | --- |
| VimeML | `codex/mac-v21-coreml`：转换、量化、工具与模型报告；完成后提交并push此分支，建立PR回到`main` |
| Vime客户端 | 在Mac现有Git仓库从最新基点建`codex/v21-coreml-client`；资源接入独立提交和PR，不混进VimeML |
| Windows | 审阅VimeML PR后合并；本地保持干净时`git pull --ff-only origin main`，不再解压源码覆盖仓库 |

提交只包含源码、配置、说明、必要fixture及测试；`artifacts/outputs/runs/handoff`沿用忽略规则。Mac结果另打一个`vimeml-v21-mac-results.zip`，仅放新增V2模型、报告、必要compiled资源和顶层`handoff.json`（两个仓库的base/head commit、PR链接、环境、文件路径/大小、通过与失败项）。不回传全量训练输入、所有旧结果或客户端源码快照。

W&B账号／run链接只保存在本地监控资料，不写入待提交的源码、配置或报告。现有Git历史保留，当前版本已移除旧链接。

产物按原相对路径恢复到Windows忽略目录；同名目标发生冲突时另存版本。来源identity复用各模型manifest，ZIP按目录清单和大小检查，不进行每条文件重复SHA256。两个PR互相记录关联，代码由Git合并，模型与trace由结果ZIP迁移。

相关记录：[追加结果](reports/v2-20261007/v21-extend.md)、[Core ML指南](coreml.md)、[V1详细实测](reference/coreml-v1.md)、[本地产物](artifacts.md)。
