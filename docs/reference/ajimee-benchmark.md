# AJIMEE real-candidate evaluation

AJIMEE evaluates kana-to-kanji conversion rather than open-ended completion. The fixed input is `JWTD_v2/v1/evaluation_items.json` from [AJIMEE-Bench](https://github.com/azooKey/AJIMEE-Bench), revision `401666cd56d1a570c2021798b64b6da4396bfd45`: 200 cases, half with left context, 83 with multiple accepted outputs. The 33 optional long-input segmentation plans are not used in this experiment.

Data is CC-BY-SA 3.0; official tool code is CC0 1.0. Private inputs and notices reside in `artifacts/benchmarks/ajimee-jwtd-v2-v1/`, outside training data.

## Input adaptation

```bash
python scripts/benchmarks/prepare_ajimee.py --input /path/to/evaluation_items.json
```

Official `input`, `context_text`, `expected_output`, and `index` map to query, left context, accepted answers, and a separate case map. Readings and all accepted outputs are preserved. Right context is null; reference text is never introduced as a candidate or fabricated committed context.

## Fixed converter export

Converter revision is `d59a28e4c7ca049aef04f29a91eae9677a7753f2`, dictionary `4d418525b090cf49c219819d05a7e3cc2a4346eb`, and emoji dictionary `67b822603391b01238d7b80b8b61b63f966cf357`. Mac export requires the matching Swift checkout and resources.

```bash
git clone https://github.com/azooKey/AzooKeyKanaKanjiConverter.git AzooKeyKanaKanjiConverter-ajimee
cd AzooKeyKanaKanjiConverter-ajimee
git checkout --detach d59a28e4c7ca049aef04f29a91eae9677a7753f2
git submodule update --init --recursive --jobs 8
swift build -c release --product CliTool -Xcxx -xobjective-c++
.build/release/CliTool evaluate /path/to/ajimee-input.json \
  --config_n_best 20 --config_typo_mode off \
  --output /path/to/azookey-candidates.json
```

Zenzai and stable mode are disabled. Stable mode in this revision rounds scores. Candidate text, order, flags, converter/dictionary revisions, and Swift version accompany the export. Development exports use separate directories with identical converter settings.

## Metrics

`scripts/benchmarks/evaluate_ajimee.py` defaults to CPU FP32 and V1; explicit checkpoint/tokenizer arguments select another model. It validates case alignment and preserves the actual candidate pool. Primary scoring, fallback, exact accepted references, and statistical boundaries are defined in [evaluation](../evaluation.md).

MinCER is the minimum normalized character edit distance over references, then averaged over cases. Recall bounds reranking success; misses remain in the denominator. Original order scores 87/200, V1 contextual LM 124/200, with answer coverage 162/200 and no empty/length-fallback cases. [V1 results](results.md) include hybrid and context diagnostics. This exposed public set is a fixed comparison, not fresh blind evidence; training overlap has not been fully audited.
