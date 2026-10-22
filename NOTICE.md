# Licensing and attribution

## Project and models

Original VimeML code and published V1/V2.1/V4.2 weights, tokenizers and Core ML packages use GPL-3.0-or-later with an additional permission under GPLv3 section 7 for conveyance through application distribution services such as the App Store. The License is in [LICENSE](LICENSE); the copyright notice, additional permission and the source form of model weights are defined in [LICENSE-EXCEPTION](LICENSE-EXCEPTION). Until 2026-10-22 the project used GPL-2.0-only; copies received under those terms keep them.

V4.2 was trained in part on RealPersonaChat and Multi-Relational Multi-Party Chat Corpus dialogue text (CC BY-SA 4.0, nu-dialogue). Its model card attributes both sources. GPLv3 is a Creative Commons designated compatible license for BY-SA 4.0 adaptations, so V4.2 releases keep that attribution with the GPLv3 terms. A dataset's source license is independent of the project license. This repository does not redistribute the private training corpora or benchmark text.

## Attribution for deployed models

Applications that bundle V2.1 or V4.2 must show the model license and the following attribution; the Vime client shows it under About Vime → Open Source Licenses.

| Model | Training sources and terms |
| --- | --- |
| V2.1 | FineWeb2-Edu Japanese (Yuichi Tateno; derived from FineWeb2 and Common Crawl), ODC-By 1.0 and Common Crawl Terms of Use; Tatoeba Japanese sentences (Tatoeba contributors), CC BY 2.0 FR unless marked CC0 |
| V4.2 | The V2.1 sources, plus RealPersonaChat and Multi-Relational Multi-Party Chat Corpus (nu-dialogue), CC BY-SA 4.0 |

Evaluation-only sources in the table below do not enter model weights.

## Third-party software

The vendored [SentencePiece bridge](examples/ios/VimeSentencePiece/UPSTREAM.md) is based on SentencePiece 0.2.1 and retains Apache-2.0 and its dependency notices. Upstream license files and generated sources are preserved. AzooKey is an external converter dependency; converter and dictionary revisions are recorded in benchmark manifests.

## Data sources

| Source | Recorded use and attribution |
| --- | --- |
| [FineWeb2-Edu Japanese](https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese) | Frozen V1/V2 web sentence corpus, also used by V4; ODC-By 1.0, subject to the Common Crawl Terms of Use |
| [Tatoeba](https://tatoeba.org/en/terms_of_use) | Japanese sentence corpus; CC BY 2.0 FR unless marked CC0; per-sentence attribution remains with local source records |
| [AdaMLLab/JpnMix](https://huggingface.co/datasets/AdaMLLab/JpnMix) | Selected `minhash_deduped` pool for V3; mixed upstream sources retain their respective terms |
| [RealPersonaChat](https://github.com/nu-dialogue/real-persona-chat) | Real-dialogue V3/V4 training source, CC-BY-SA 4.0 |
| [Multi-Relational Multi-Party Chat Corpus](https://github.com/nu-dialogue/multi-relational-multi-party-chat-corpus) | Real-dialogue V3/V4 training source, CC-BY-SA 4.0 |
| [AJIMEE-Bench](https://github.com/azooKey/AJIMEE-Bench) | JWTD v2 evaluation text, CC-BY-SA 3.0; official tool code, CC0 1.0 |
| [WRIME](https://github.com/ids-cv/wrime) | Standard IME evaluation source, CC-BY-NC-ND 4.0 |
| JMultiWOZ | Standard IME evaluation source, CC-BY-SA 4.0; local source notices record attribution |

WRIME-derived and JMultiWOZ-derived benchmark texts are restricted to the recorded private research exchange and are not published with this repository or model releases. Local notices and source-level attribution accompany private artifacts. Aggregate scores do not confer permission to redistribute underlying texts or candidate exports. The project GPL license does not relicense upstream datasets or vendored components.
