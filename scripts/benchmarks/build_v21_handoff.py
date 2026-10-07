"""Package prepared IME inputs with provenance and a Mac export command."""

import argparse
import collections
import json
import re
import shutil
import zipfile
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[2]


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "artifacts/benchmarks/ime-expanded-v21-final"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "handoff/ime-v21-3000-20261007")
    args = parser.parse_args()
    archive = args.output.with_suffix(".zip")
    if args.output.exists() or archive.exists():
        parser.error("Keep existing handoffs; use a fresh output.")
    rows = json.loads((args.input / "evaluation_items.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.input / "manifest.json").read_text(encoding="utf-8"))
    splits = {
        role: json.loads((args.input / role / "evaluation_items.json").read_text(encoding="utf-8"))
        for role in ("development", "blind")
    }
    assert len(rows) == 3000 and len(splits["development"]) == 2000 and len(splits["blind"]) == 1000
    assert rows == splits["development"] + splits["blind"]
    for field in ("index", "input"):
        assert len({r[field] for r in rows}) == 3000
    for field in ("group_id", "source_url", "original_sentence"):
        assert len({r["provenance"][field] for r in rows}) == 3000
    for row in rows:
        p = row["provenance"]
        assert p["original_sentence"] == row["context_text"] + row["expected_output"][0]
        assert p["span_start"] == len(row["context_text"]) and p["span_end"] == len(
            p["original_sentence"]
        )
        assert re.fullmatch("[ァ-ヺー。、！？]+", row["input"])
        assert row["expected_output"] and row["review"]["not_formal_gold"]
    maximum = {}
    for version in ("v1", "v2"):
        tokenizer = spm.SentencePieceProcessor(
            model_file=str(ROOT / f"artifacts/tokenizers/ja-unigram-16k-{version}/tokenizer.model")
        )
        maximum[version] = max(
            len(tokenizer.encode(r["context_text"] + a)) + 1
            for r in rows
            for a in r["expected_output"]
        )
        assert maximum[version] <= 128
    for role, data in {"all": rows, **splits}.items():
        folder = args.input if role == "all" else args.input / role
        cli = json.loads((folder / "ajimee-input.json").read_text(encoding="utf-8"))
        mapping = json.loads((folder / "case-map.json").read_text(encoding="utf-8"))
        assert len(cli) == len(data) and len(mapping) == len(data)
        for case, converted in zip(data, cli):
            assert (
                converted["query"] == case["input"]
                and converted["left_context"] == case["context_text"]
            )
            assert (
                converted["answer"] == case["expected_output"]
                and converted["right_context"] is None
            )
    qa = {
        "cases": 3000,
        "development": 2000,
        "blind": 1000,
        "unique_ids": 3000,
        "unique_readings": 3000,
        "unique_source_groups": 3000,
        "unique_source_urls": 3000,
        "span_and_cli_mapping_checked": True,
        "maximum_joint_reference_tokens_including_bos": maximum,
        "candidate_token_lengths_checked": False,
        "actual_candidate_export_complete": False,
        "native_label_review_complete": False,
        "models_scored": False,
        "sources": dict(collections.Counter(r["provenance"]["source"] for r in rows)),
        "with_context": sum(bool(r["context_text"]) for r in rows),
        "validation_scope": "Actual 3000 files: counts, unique fields, spans, CLI mapping, reference token budgets. No repeated SHA256 or test suite.",
    }
    shutil.copytree(args.input, args.output)
    script = (ROOT / "scripts/benchmarks/export_azookey_v21.sh").read_text(encoding="utf-8")
    (args.output / "export_azookey.sh").write_bytes(script.encode("utf-8"))
    write(args.output / "handoff-validation.json", qa)
    manifest["handoff_validation"] = qa
    manifest["status"] = "candidate_collection_inputs_ready_labels_provisional"
    write(args.output / "manifest.json", manifest)
    (args.output / "NOTICE.md").write_text(
        """# Source attribution

This package contains derived conversion spans from the frozen VimeML corpus
validation and test splits. Source text is unchanged; kana and provisional
orthographic alternatives are derived. Each evaluation item records its original
sentence, source URL, document ID, local source file/row and conversion span.

- FineWeb2 Edu Japanese: Yuichi Tateno (2025), built on HuggingFaceFW FineWeb2.
  https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese
  Dataset card: https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese/raw/main/README.md
  Dataset license: Open Data Commons Attribution 1.0 (ODC-By), also subject to
  Common Crawl terms: https://commoncrawl.org/terms-of-use . Underlying web content
  retains its original rights; original website URLs are recorded per case.
- Tatoeba sentence contributors: https://tatoeba.org/
  Terms: https://tatoeba.org/en/terms_of_use ; default text license CC BY 2.0 FR.
  Per-sentence URLs/IDs are recorded so original contributors and licenses can be
  resolved; contributor names were not included in the local three-column TSV.

This is a private research handoff, not a publication of a fully adjudicated
benchmark. License/author attribution should follow the source records if shared
publicly. No corpus, model, SSH or W&B credentials are included.
""",
        encoding="utf-8",
    )
    readme = f"""# V2.1：3,000 条 AzooKey 候选收集输入

2026-10-07。本包是实际 JSON 输入，开发集 **2,000 条**，盲集 **1,000 条**。
可直接使用之前成功导出 AJIMEE 的 Mac 转换器 checkout。

## 一条命令导出

解压 zip，进入 `ime-v21-3000-20261007`，将下方路径换成你的转换器仓库路径：

```bash
bash export_azookey.sh "/实际路径/AzooKeyKanaKanjiConverter"
```

若使用此前指南的仓库：

```bash
bash export_azookey.sh "$HOME/Sources/AzooKeyKanaKanjiConverter-ajimee"
```

脚本按 development、blind 顺序导出真实候选；已成功完成且输入一致的部分会保留。
使用 `n_best=20`、`typo_mode=off`、不启用 Zenzai、不使用 `--stable`。
记录 converter commit、字典子模块版本、Swift 版本、flags、日志和完成标记。
转换器固定为 `d59a28e4c7ca049aef04f29a91eae9677a7753f2`，不会自动修改你的 checkout。
需要该 checkout 已有 `.build/release/CliTool`；如尚未构建，在转换器仓库执行：

```bash
swift build -c release --product CliTool -Xcxx -xobjective-c++
```

## 返回哪些文件

导出结束后，把整个 **`azookey-results` 目录**发回，两个 split 都保留：

```text
azookey-results/
  development/
    ajimee-input.json
    case-map.json
    azookey-candidates.json
    converter-version.txt
    dictionary-versions.txt
    swift-version.txt
    export-flags.txt
    export.log
    completed.txt
  blind/
    （同上）
```

脚本第二个参数可指定结果目录。若上次中断留下未标记完成的候选文件，脚本保留它并提示改用新目录，避免误覆盖。
根目录还有合并的 `ajimee-input.json`（3,000 条），方便检查；正式交接优先分别导出两套。

## 输入检查与用途

- 冻结 corpus validation 原句用于 development，test 原句用于 blind；未读训练 split 生成样本。
- 全部读音经 UniDic Lite + Sudachi 两词典、原句上下文读音一致性和词边界检查；避免从复合名词、片假名词内部截断。
- 3,000 个读音、源文档组、原句和来源 URL 各自唯一；剔除与现有 200/137 条完全相同的 query/context。
- 共 {qa["with_context"]:,} 条有左文、{3000 - qa["with_context"]:,} 条无左文；来源 FineWeb {qa["sources"].get("fineweb", 0):,} 条、Tatoeba {qa["sources"].get("tatoeba", 0):,} 条。
- 每条保留完整原句、转换区间和来源，开发/盲集按来源隔离；模型得分和候选召回不参与选样。
- 所有参考答案与左文联合分词，含 BOS 最大 V1={maximum["v1"]}、V2={maximum["v2"]} tokens，低于 context128。真实候选的 token 长度要等导出后再检查。

**当前标签是草稿，不是已经验收的正式 3,000 条 gold。** 双词典一致仍可能选错读音；网页原句可能有噪声，同读音表记不保证穷尽，尚未完成全部母语者审核。部分高风险多读音词和已发现的提取噪声已过滤，但不是同音词/专名等类别均衡的最终集。近重复来源也未全面审计。

返回真实候选后，再复核标签、隔离歧义、记录 Recall@20 和类别/候选数分层；保留全池与召回失败诊断，另报告 covered subset。
盲集在模型选择阶段不进行 LM 计分；只在开发集确定模型之后执行最终盲测。本次读取 test 文本仅准备输入，没有算 test BPC 或运行盲集模型。
原计划目标是 3,000 条**有效正式样本**，这次先交付 3,000 条候选收集输入；验收如有剔除，会从相同来源规则补齐后另建正式版本。

## 包内文件

每套有 `evaluation_items.json`、`ajimee-input.json`、`case-map.json`、`stats.json`。
`manifest.json` 记录方法、限制和版本；`handoff-validation.json` 是本次实际文件检查；`NOTICE.md` 和逐条 provenance 记录来源。
没有假候选，没有模型/大语料文件，也没有 SSH 或 W&B 凭据。

脚本已通过 Bash 语法检查；实际 macOS Swift 导出由你在现有 AzooKey 环境运行。
"""
    (args.output / "README_CN.md").write_text(readme, encoding="utf-8")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for path in sorted(args.output.rglob("*")):
            if path.is_file():
                package.write(path, path.relative_to(args.output.parent).as_posix())
    with zipfile.ZipFile(archive) as package:
        assert len(json.loads(package.read(args.output.name + "/ajimee-input.json"))) == 3000
        assert all(
            json.loads(package.read(args.output.name + "/" + role + "/ajimee-input.json"))
            == json.loads((args.output / role / "ajimee-input.json").read_text(encoding="utf-8"))
            for role in splits
        )
    print(
        json.dumps(
            {
                "directory": str(args.output),
                "archive": str(archive),
                "zip_bytes": archive.stat().st_size,
                "validation": qa,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
