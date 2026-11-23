# Licensing and attribution

## Project and models

Original VimeML code and published V1/V2.1/V4.2, vimeml-tiny-zh-v1 and vimeml-zh-v2 weights, tokenizers, Core ML packages and the vimeml-zh-v2 pinyin reading dictionary use GPL-3.0-or-later with an additional permission under GPLv3 section 7 for conveyance through application distribution services such as the App Store. The License is in [LICENSE](LICENSE); the copyright notice, additional permission and the source form of model weights are defined in [LICENSE-EXCEPTION](LICENSE-EXCEPTION). Until 2026-10-22 the project used GPL-2.0-only; copies received under those terms keep them.

V4.2 was trained in part on RealPersonaChat and Multi-Relational Multi-Party Chat Corpus dialogue text (CC BY-SA 4.0, nu-dialogue). Its model card attributes both sources. GPLv3 is a Creative Commons designated compatible license for BY-SA 4.0 adaptations, so V4.2 releases keep that attribution with the GPLv3 terms. A dataset's source license is independent of the project license. This repository does not redistribute the private training corpora or benchmark text.

## Attribution for deployed models

Applications that bundle V2.1, V4.2, vimeml-tiny-zh-v1 or vimeml-zh-v2-20m must show the model license and the following attribution; the Vime client shows it under About Vime → Open Source Licenses.

| Model | Training sources and terms |
| --- | --- |
| V2.1 | FineWeb2-Edu Japanese (Yuichi Tateno; derived from FineWeb2 and Common Crawl), ODC-By 1.0 and Common Crawl Terms of Use; Tatoeba Japanese sentences (Tatoeba contributors), CC BY 2.0 FR unless marked CC0 |
| V4.2 | The V2.1 sources, plus RealPersonaChat and Multi-Relational Multi-Party Chat Corpus (nu-dialogue), CC BY-SA 4.0 |
| vimeml-tiny-zh-v1 | FineWeb2 `cmn_Hani` (Hugging Face; derived from Common Crawl), ODC-By 1.0 and Common Crawl Terms of Use; LCCC-base (thu-coai; Wang et al., *A Large-Scale Chinese Short-Text Conversation Dataset*, NLPCC 2020), MIT, with a README disclaimer that the dataset and pretrained dialogue models are for research purposes |
| vimeml-zh-v2-20m | The vimeml-tiny-zh-v1 sources, plus colloquial post-training dialogue: Douban Conversation Corpus (Wu et al., ACL 2017), no license stated, citation requested; MMChat (Zheng et al., LREC 2022), Weibo dialogues, non-commercial use per its dataset card; Baidu Tieba Sun Xiaochuan Comments Dataset (Zheng and zyw, 2025), CC BY-NC 4.0 in the dataset card body (CC BY 4.0 in its metadata). The v2 Hugging Face repository's `NOTICE.md` gives the full attribution |

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
| [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) `cmn_Hani` | vimeml-tiny-zh-v1 and vimeml-zh-v2 web text (v1: 21 of 370 train files, cleaned and reviewed; the desktop 800M adds 60 more files, cleaned); also the source of the v2-20m colloquial paragraphs; ODC-By 1.0, subject to the Common Crawl Terms of Use |
| [LCCC-base](https://github.com/thu-coai/CDial-GPT) | vimeml-tiny-zh-v1 and vimeml-zh-v2 dialogue text (about 5% of v1's supervised tokens); MIT; the README disclaimer states research-only use of the dataset and pretrained dialogue models |
| [Douban Conversation Corpus](https://github.com/MarkWuNLP/MultiTurnResponseSelection) | vimeml-zh-v2-20m colloquial post-training dialogue turns, after rule cleaning and an LLM quality screen; no license stated, the authors ask users to cite the paper |
| [MMChat](https://github.com/silverriver/MMChat) | vimeml-zh-v2-20m colloquial post-training (the authors' LCCC-filtered dialogues), cleaned and quality-screened; collected from Weibo, the dataset card asks users to follow Weibo's policy and restrict use to non-commercial purposes |
| [Baidu Tieba Sun Xiaochuan Comments](https://huggingface.co/datasets/Orphanage/Baidu_Tieba_SunXiaochuan) | vimeml-zh-v2-20m colloquial post-training, cleaned and quality-screened; CC BY-NC 4.0 in the dataset card body (its metadata says CC BY 4.0) |
| [CrossWOZ](https://github.com/thu-coai/CrossWOZ), [RiSAWOZ](https://github.com/terryqj0107/RiSAWOZ), [KdConv](https://github.com/thu-coai/KdConv) | Chinese IME and pinyin-generation evaluation sources; Apache-2.0, MIT, Apache-2.0 |
| [CPED](https://github.com/scutcyr/CPED) | Chinese prediction and pinyin-generation evaluation source; Apache-2.0 repository license; television dialogue, kept local |

The colloquial pinyin benchmark `ime-pinyin-zh-colloq-v1` is drawn from held-out splits of LCCC, the FineWeb2 colloquial paragraphs, Douban, MMChat and Tieba; those splits never enter training.

WRIME-derived and JMultiWOZ-derived benchmark texts are restricted to the recorded private research exchange and are not published with this repository or model releases. Local notices and source-level attribution accompany private artifacts. Aggregate scores do not confer permission to redistribute underlying texts or candidate exports. The project GPL license does not relicense upstream datasets or vendored components.
