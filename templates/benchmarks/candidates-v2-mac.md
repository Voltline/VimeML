# IME candidate export v2 (Mac)

Exports AzooKey n-best candidate pools for the tracks in `tracks.tsv`. No language model runs; the results are returned for local FP32 reranking under the registered metric policy.

| Track | n-best | Purpose |
| --- | --- | --- |
| `standard-v2-blind` | 20, 50 | New Standard blind release (labels frozen before export) |
| `homophone-v1` | 20, 50 | Written-text homophone challenge (Japanese Wikipedia) |
| `standard-v1-development` | 20, 50 | n50 recall gain; n20 repeats the original export (determinism) |
| `standard-v1-development-nocontext` | 20 | Same inputs, empty left context: do pools depend on context? |
| `incremental-proto1` | 20, 50 | Segmented-input prototype on development messages |

Fixed setup (same as every earlier export):

| Component | Revision / setting |
| --- | --- |
| Converter | `d59a28e4c7ca049aef04f29a91eae9677a7753f2` |
| Main dictionary | `4d418525b090cf49c219819d05a7e3cc2a4346eb` |
| Emoji dictionary | `67b822603391b01238d7b80b8b61b63f966cf357` |
| Flags | typo off, Zenzai disabled, stable off; only `--config_n_best` changes |

## Command

```bash
bash run_candidates_mac.sh "$HOME/Sources/AzooKeyKanaKanjiConverter-ajimee"
```

Return `ime-candidates-v2-azookey-results.zip`. A retry into a fresh directory keeps earlier output:

```bash
bash run_candidates_mac.sh "$HOME/Sources/AzooKeyKanaKanjiConverter-ajimee" azookey-results-retry1
```

Each `azookey-results/<track>/n<k>/` keeps the input, case map, candidates, converter/dictionary/Swift versions, flags, log and completion marker. References are never added to candidates. The data are private research material; see `NOTICE.md`.
