> 详细参考与 FP32 基线。当前阶段入口见 [文档索引](../index.md)，量化后的结果见 [Core ML 指南](../coreml.md)。

# AJIMEE：真实转换候选评测

首轮输入转换、Mac 候选导出和本地评分已完成。AJIMEE 测假名→汉字转换，不测开放式短语联想。当前正式文件位于 `artifacts/benchmarks/ajimee-jwtd-v2-v1/`。

## 固定数据和版本

- [azooKey/AJIMEE-Bench](https://github.com/azooKey/AJIMEE-Bench)，数据 commit `401666cd56d1a570c2021798b64b6da4396bfd45`。
- `JWTD_v2/v1/evaluation_items.json`：200 条，100 条有左文、100 条无左文，83 条多个可接受答案，33 条长输入分割方案。
- 数据基于 JWTD v2 测试集，读音与切分经人工核对；数据 CC-BY-SA 3.0，官方工具代码 CC0 1.0。原数据及 `NOTICE.md` 留在本地产物，评测不加入训练语料。
- 转换器 `d59a28e4c7ca049aef04f29a91eae9677a7753f2`；主字典 `4d418525b090cf49c219819d05a7e3cc2a4346eb`；emoji 字典 `67b822603391b01238d7b80b8b61b63f966cf357`。

## Python 输入适配

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/benchmarks/prepare_ajimee.py --input "C:\Users\Zhang\Downloads\evaluation_items.json"
```

默认输出 `artifacts/benchmarks/ajimee-jwtd-v2-v1/`：原始文件按字节保存、`ajimee-input.json`、`case-map.json`、manifest/hash 和 NOTICE。相同输入重复运行检查后复用；不覆盖其他版本。

| 官方字段 | CLI 字段 | 处理 |
| --- | --- | --- |
| `input` | `query` | 片假名读音原样 |
| `context_text` | `left_context` | 仅给定左文，不读取原全文补答案 |
| `expected_output` | `answer` | 完整可接受答案数组，不作为候选池 |
| `index` | `tag` 与独立映射 | CLI 不保留 tag，按位置/query/context 再核验 |

CLI 读取 JSON 数组。右文设 null，未提供用户字典，不正规化或拆分长输入。33 条分割方案未参与本轮，不能把前段标准答案当已提交文字冒充真实上下文。

## Mac 构建与导出

以下命令针对固定版本；用户已在 Mac 成功导出候选。需要 Swift6.1+、macOS13+ 与开发工具；先检查 `xcrun swift --version`。新机器缺工具时安装 Command Line Tools / Xcode。

```bash
mkdir -p ~/Sources
cd ~/Sources
git clone https://github.com/azooKey/AzooKeyKanaKanjiConverter.git AzooKeyKanaKanjiConverter-ajimee
cd AzooKeyKanaKanjiConverter-ajimee
git checkout --detach d59a28e4c7ca049aef04f29a91eae9677a7753f2
git submodule update --init --recursive --jobs 8
swift build -c release --product CliTool -Xcxx -xobjective-c++
.build/release/CliTool evaluate --help
```

把 Python 生成的输入复制到 `~/Downloads/ajimee-input.json`，在转换器仓库根目录执行：

```bash
mkdir -p ajimee-results
cp ~/Downloads/ajimee-input.json ajimee-results/ajimee-input.json
git rev-parse HEAD > ajimee-results/converter-version.txt
git submodule status --recursive > ajimee-results/dictionary-versions.txt
swift --version > ajimee-results/swift-version.txt
.build/release/CliTool evaluate ajimee-results/ajimee-input.json \
  --config_n_best 20 --config_typo_mode off \
  --output ajimee-results/azookey-candidates.json
```

直接运行 CliTool 保留 SwiftPM 字典资源；已安装同版本 anco 可使用等价 `anco evaluate` 命令。不启用 Zenzai，不加 `--stable`（该版本会将 score 取整）。预测、学习和错字校正关闭，每条输入独立停止 composition；经典转换原排序不描述为已使用 Tiny LM 的左文。

把整个结果目录复制到 Windows 的 `artifacts/benchmarks/ajimee-jwtd-v2-v1/ajimee-results/`。开发集导出用单独目录，固定相同转换器、字典和 N-best；未来 Vime 版本不同需另建实验。

## 本地评分与指标

```powershell
# 新结果目录；现有正式结果不覆盖
.\.venv\Scripts\python.exe -X utf8 -u scripts/benchmarks/evaluate_ajimee.py --output outputs/ime-eval/ajimee-recheck
```

默认 CPU4线程FP32、`tiny-ja-v1/best.pt`；`--baseline-only` 不加载模型。工具核对源/hash/映射/Mac 输入、逐条 query/context/全部答案、候选数量与版本，不增删候选。

评分把 context+candidate 联合分词，在公共 token 前缀之后求原始词表 logP sum，不加 EOS；接缝可能重新分词，因此是 token 后缀似然代理，不精确等于字符串条件概率。不能分别 encode 后拼接。mean 是固定的次要长度诊断，不据结果重选主评分。

超 context128 或无法完整计分时整条回退原排序，保留全分母，不静默截断。官方左文可能含句界，本轮使用原给定左文；Vime 句内裁剪应另命名实验。

- Top-1 / Top-5：任一可接受答案精确匹配；不静默统一表记宽度。
- Recall@20：实际候选池覆盖，是排序精确命中的上限。
- MinCER：对每个参考答案计算字符编辑距离 / 参考长度，取最小后按样本平均，沿用[官方定义](https://github.com/azooKey/AJIMEE-Bench/blob/401666cd56d1a570c2021798b64b6da4396bfd45/utils.py)。
- 单列有/无左文、覆盖子集、纠正/改坏、回退与空池；空池仍计失败，并列保持原顺序。

原排序 87/200，纯 contextual LM 124/200；覆盖 162/200，无空池或长度回退。完整固定组合结果见 [基线结果](results.md)，系数只在独立开发集选择，见 [组合排序](hybrid-ranking.md)。公开评测已用于错误分析，不是完全盲测；与训练文本的重叠尚未全面审计，开发机耗时不等于 iOS 延迟。
