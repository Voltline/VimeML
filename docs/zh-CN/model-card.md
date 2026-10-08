# VimeML模型卡

主要模型卡为[MODEL_CARD.md](../../MODEL_CARD.md)。公开版本为[V1](https://huggingface.co/Voltline/vimeml-tiny-ja-v1)和[V2.1](https://huggingface.co/Voltline/vimeml-tiny-ja-v2.1)；V2.0、V3 A/B为实验checkpoint。

## 架构与用途

| 属性 | V1 | V2／V2.1／V3 |
| --- | --- | --- |
| 参数量 | 7,386,624 | 12,537,920 |
| 词表／context | 16,384／128 | 16,384／128 |
| Hidden／层／头 | 256／4／4 | 320／6／5 |
| FFN | GELU，1024 | SwiGLU，832 |
| 归一化 | Pre-LayerNorm | Pre-RMSNorm，FP32方差 |
| 位置／输出 | 可学习位置、共享embedding/head | 可学习位置、共享embedding/head |

两代均为decoder-only因果LM，dropout为0，无KV cache。分词器为Unigram、byte fallback，特殊ID为0/1/2/3。V2关闭dummy prefix，词表ID与V1不兼容；联合编码仍可能发生边界重切分。

主要用途是对AzooKey真实候选池按左文重排。context与候选联合分词，以共同token前缀后的logP sum评分；该代理不精确等于字符串条件概率。检索、分词、稳定排序、回退与生成搜索在应用侧实现。短文本续写与下一词联想仍属实验能力，重复与语义错误未消除，未验证通用聊天或事实问答能力。

## 数据与训练

V1/V2冻结语料来自9个FineWeb2-Edu日语分片和Tatoeba，共25,713,003唯一句子，按文档与精确重复分组切分。全面语义近重复与基准污染审计未完成。V1从头训练1轮；V2.0从头4轮；V2.1 restart1从V2.0续训2轮并重建AdamW；extend5携带优化器状态追加3.273轮，部署选定step40000。V2训练有30%合格首窗口prefix crop，验证不裁剪。

V3使用当前8个JpnMix分片的清洗抽样池及两套真人聊天，B另含旧V2语料。聊天为裁剪后有效tokens的5%，包含重复曝光，不等于新增独立语料。A/B未形成稳定替换收益，V2.1仍为部署版本。

## 评测与部署边界

V2.1 FP32旧验证BPC为3.0802686，AJIMEE144/200、历史dev122/137、草稿dev1487/2000；INT8对应144、122、1481。Standard development：V2.1／A／B为505／508／507，分母728；已暴露固定reference为535／546／533，分母725；旧500草稿回归为374／368／364。

Standard标签为冻结AI专家审核版，不是母语人工gold。已暴露reference不能用于重新选型或宣称新盲测；旧1000 blind未评分。逐case配对统计未充分校正作者／用户相关性、多重比较与development选型，属于探索性证据。

V2.1 Core ML包14.33 MB，INT8 block32存储、FP32计算、CPU_ONLY、iOS18以上；输入int32[1,T]、T≤128，输出float32[1,T,16384]原始logits。未压缩转换通过对齐，INT8最大logits误差1.00949144，严格对齐失败。小集命中数相同不代表分布或完整排序等效，也未证实该失败是模型规模或架构的固有属性。

短时真机测量不能认证长期内存、功耗或普遍延迟。代码与发布权重采用GPL-2.0；源语料和第三方组件保留独立许可。公开发布仅包含聚合报告，不含受限原始基准文本或训练优化器。
