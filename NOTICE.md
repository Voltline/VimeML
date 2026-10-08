# Licensing and attribution

## Project and models

Original VimeML code and published V1/V2.1 weights use GPL-2.0; the full license is preserved in [LICENSE](LICENSE). A dataset's source license is independent of the project license. This repository does not redistribute the private training corpora or benchmark text.

## Third-party software

The vendored [SentencePiece bridge](examples/ios/VimeSentencePiece/UPSTREAM.md) is based on SentencePiece 0.2.1 and retains Apache-2.0 and its dependency notices. Upstream license files and generated sources are preserved. AzooKey is an external converter dependency; converter and dictionary revisions are recorded in benchmark manifests.

## Data sources

| Source | Recorded use and attribution |
| --- | --- |
| [FineWeb2-Edu Japanese](https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese) | Frozen V1/V2 web sentence corpus; source data card governs provenance and terms |
| [Tatoeba](https://tatoeba.org/en/terms_of_use) | Japanese sentence corpus; attribution and license metadata remain with local source records |
| [AdaMLLab/JpnMix](https://huggingface.co/datasets/AdaMLLab/JpnMix) | Selected `minhash_deduped` pool for V3; mixed upstream sources retain their respective terms |
| [RealPersonaChat](https://github.com/nu-dialogue/real-persona-chat) | Real-dialogue V3 training source, CC-BY-SA 4.0 |
| [Multi-Relational Multi-Party Chat Corpus](https://github.com/nu-dialogue/multi-relational-multi-party-chat-corpus) | Real-dialogue V3 training source, CC-BY-SA 4.0 |
| [AJIMEE-Bench](https://github.com/azooKey/AJIMEE-Bench) | JWTD v2 evaluation text, CC-BY-SA 3.0; official tool code, CC0 1.0 |
| [WRIME](https://github.com/ids-cv/wrime) | Standard IME evaluation source, CC-BY-NC-ND 4.0 |
| JMultiWOZ | Standard IME evaluation source, CC-BY-SA 4.0; local source notices record attribution |

WRIME-derived and JMultiWOZ-derived benchmark texts are restricted to the recorded private research exchange and are not published with this repository or model releases. Local notices and source-level attribution accompany private artifacts. Aggregate scores do not confer permission to redistribute underlying texts or candidate exports. The project GPL license does not relicense upstream datasets or vendored components.
