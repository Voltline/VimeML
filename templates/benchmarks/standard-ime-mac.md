# Standard Japanese IME candidate-export specification

Version: `__VERSION__`. Inputs: __DEVELOPMENT__ development cases and __BLIND__ originally blind cases, __TOTAL__ total. Label quality: `__LABEL_QUALITY__`; native-speaker gold status: `__FORMAL_GOLD__`. The manifest records the frozen split/consumption role. The historical 500-case web track already has candidates and is not re-exported.

## Export scope

The output is `ime-standard-ja-v1-azookey-results.zip`, produced from the supplied unchanged case IDs, readings, left contexts, references, and order. Development and the original blind split remain separate export directories. No language model runs during candidate export.

| Component | Fixed revision / setting |
| --- | --- |
| Converter | `d59a28e4c7ca049aef04f29a91eae9677a7753f2` |
| Main dictionary | `4d418525b090cf49c219819d05a7e3cc2a4346eb` |
| Emoji dictionary | `67b822603391b01238d7b80b8b61b63f966cf357` |
| Export settings | N-best 20, typo off, Zenzai disabled, stable off |

The existing matching CliTool can be reused; the script builds it once if absent. A mismatched converter requires a separate checkout, preserving existing work. Core ML resources, model weights, GPU training, full-file checksum repetition, and test suites are outside the export scope.

## Command

```bash
bash run_mac.sh "$HOME/Sources/AzooKeyKanaKanjiConverter-ajimee"
```

A distinct retry output preserves any incomplete export:

```bash
bash run_mac.sh "/path/to/AzooKeyKanaKanjiConverter-ajimee" "azookey-results-retry1"
```

## Result layout

```text
azookey-results/
  development/
  blind/
```

Each directory retains input JSON, case map, actual candidates, converter/dictionary/Swift versions, flags, export log, and completion marker. Retrieval misses remain unchanged; references are never added to candidates. Failures retain existing results and logs. Return packaging contains export artifacts rather than source branches or models.

Fixed FP32 evaluation follows separately under the registered benchmark plan. After initial consumption, the original blind directory is an exposed reference, not a new blind test. Text and derived items are private research materials; source attribution and licensing remain in the archive notice.
