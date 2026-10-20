# Chinese native-speaker review packet — 2026-10-17

Status: prepared for the user's native-speaker annotation. No human labels have been received, and no human-validated accuracy is claimed.

The user requested approximately 100 cases to independently examine the earlier high-looking results and selected a local, one-case-at-a-time annotation page with saving and export. The resulting packet contains all 60 original sentence inputs, preserved verbatim, plus 40 newly authored diagnostics. It uses a deterministic shuffled order and blind IDs H001–H100.

The new diagnostics comprise ten context/homophone cases, five conversational full-pinyin sentences, five first-letter abbreviations, five mixed abbreviations, five authored key errors, and ten context-free ambiguous strings. They were authored and frozen without running models or using model output to select cases. This is a diagnostic packet, not a sample from real user traffic. Its two source groups must be scored separately.

The annotation page and exported packet contain only blind ID, committed context and raw key input. Original labels, authored seed text, source IDs, model names, scores and candidate rankings are excluded. The private mapping is retained separately in the ignored artifact directory. Newly authored seed text is explicitly marked as a construction aid, not human gold.

The reviewer independently transcribes acceptable Chinese spellings, one per line, with the preferred spelling first. Decisions distinguish one natural answer, multiple natural answers, insufficient information and malformed/uninterpretable input. The latter two can be confirmed without supplying an answer. Drafts and confirmed annotations are distinct; modifying a confirmed answer returns it to draft status. A multi-answer decision requires at least two spellings. Lists need not exhaust all possible proper names or grammatical interpretations; unlisted model alternatives may require subsequent human adjudication rather than automatic rejection.

The page is a self-contained offline HTML document, with no external dependencies, analytics, network submissions or model requests. Drafts, completion state and current position are saved in the current browser's local storage. Export produces a JSON snapshot containing all 100 inputs and annotation records, including partial progress. Import verifies dataset identity, IDs, field types and confirmed-answer consistency before replacing current work. A replacement prompt protects an existing local session. The page displays a warning if browser storage is unavailable and still supports JSON export.

Local deliverable: `outputs/chinese-human-review-20261017/dist/index.html`. A Chinese usage guide is beside `dist/`. Private source mapping, input-authoring script and frozen manifest remain under `artifacts/chinese-human-review-20261017/`; datasets and annotation records are not tracked in Git.

Packet identity: `zh-human-100-20261017-4c0543baa2d76573`. The complete canonical input hash and source-file hashes are in the private manifest. This packet is separate from all previously scored candidate and label files. Existing model snapshots, scores and reports remain unchanged.

Implementation:

- [`chinese_review.py`](../../src/vimeml/benchmarks/chinese_review.py): frozen packet builder; separates public inputs from private source mapping.
- [`prepare_chinese_human_review.py`](../../scripts/benchmarks/prepare_chinese_human_review.py): thin build entry point.
- [`chinese-ime-human-review.html`](../../templates/benchmarks/chinese-ime-human-review.html): offline annotation UI, saving, import and export.

Validation checked all 100 public records for the ID/context/input-only schema, preserved all 60 original inputs, verified the 40 new cases and absence of duplicate context/input pairs, checked source syntax and reproducible packet identity, and exercised confirmation validation, multiple answers, uncertain cases, draft persistence, reload restoration, JSON export/import, wrong-dataset rejection and confirmation invalidation using a dependency-free synthetic DOM. The validation creates no browser or human annotation records.

Browser visual QA was unavailable: the in-app browser security policy rejected the `file://` URL protocol. No alternate browser surface or indirect browser-opening command was used to bypass that restriction. The handoff is the offline document for the user to open in their own system browser; functionality checks do not constitute a browser rendering test.

Once labels arrive, freeze their exact export before scoring. Report the legacy audit and new diagnostics separately, disclose ambiguity/exclusion counts, and preserve multi-answer labels. Any references previously used to select or revise models remain reference sets, even after the labels are improved.
