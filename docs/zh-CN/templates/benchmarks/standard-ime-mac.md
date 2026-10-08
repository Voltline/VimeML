# Standard Japanese IME：Mac候选交接

版本：`__VERSION__`。
输入：development __DEVELOPMENT__条，blind __BLIND__条，共__TOTAL__条。
标签等级：`__LABEL_QUALITY__`；是否母语人工gold：`__FORMAL_GOLD__`。具体冻结状态见manifest。
旧FineWeb500条已有真实候选，在Windows端复用，不需要再次导出。

## 候选导出范围

在本包中完成真实AzooKey候选导出，生成`ime-standard-ja-v1-azookey-results.zip`。
复用此前成功导出候选的`AzooKeyKanaKanjiConverter-ajimee` checkout及CliTool：

- Converter：`d59a28e4c7ca049aef04f29a91eae9677a7753f2`。
- 主字典：`4d418525b090cf49c219819d05a7e3cc2a4346eb`。
- Emoji字典：`67b822603391b01238d7b80b8b61b63f966cf357`。
- `n_best=20`、`typo_mode=off`、Zenzai disabled、stable off；这一步只收集候选，不运行任何LM。

标签、读音、左文、case ID和输入顺序按包内文件原样使用；保持两组独立导出。
版本不同则使用独立checkout恢复上述快照，保留正在使用的仓库。已有CliTool可直接运行，缺失时按脚本构建一次。
任务不需要Core ML、模型权重、GPU训练、重复SHA256或测试套件，也不涉及GitHub提交。

## 运行

解压并进入本包目录：

```bash
bash run_mac.sh "$HOME/Sources/AzooKeyKanaKanjiConverter-ajimee"
```

路径不同则将参数替换成已有转换器仓库路径。脚本构建缺失的CliTool、顺序导出两组真实候选，然后打包结果。
若保留了未完成的候选文件，使用新的结果目录，原文件留存：

```bash
bash run_mac.sh "/actual/path/AzooKeyKanaKanjiConverter-ajimee" "azookey-results-retry1"
```

## 返回结果

返回本包目录内生成的 **`ime-standard-ja-v1-azookey-results.zip`**，无需回传Git分支或模型。
压缩包内部固定为：

```text
azookey-results/
  development/
  blind/
```

每组保留`ajimee-input.json`、`case-map.json`、`azookey-candidates.json`、converter/字典/Swift版本、flags、export.log和completed.txt。
候选未召回答案也保留原始结果，不手动添加答案到候选。脚本异常时保留日志与现有结果；恢复后采用新目录完成导出。
Windows端负责验收、固定FP32评分及预登记盲测。包内文本与派生题库仅用于本次私有研究交接，来源归属及许可见NOTICE。
