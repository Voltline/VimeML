# Historical development generation and review

The 137-case V1 development set is an AI-reviewed generated dataset used for historical hybrid-score calibration. It is not native-speaker gold and is separate from the newer Standard IME source benchmark.

## Generation and verification

`scripts/benchmarks/generate_development.py` requests 200 cases across ten topics, in batches of ten, with approximately half using left context. DeepSeek is the default generator. A separate verification request receives context and reading without the answer or local LM scores; the default same-model judge can share systematic errors. Labels require recorded review rather than JSON validity alone.

```bash
python scripts/benchmarks/generate_development.py --dry-run
python scripts/benchmarks/generate_development.py --output artifacts/benchmarks/ime-dev-generation-new
```

Credentials are supplied through environment variables such as `SJTU_API_KEY`. Defaults limit each account to four concurrent requests, 10 RPM, and 80K TPM, shared by generation, verification, and retry. Independent account names may be configured; literal keys remain outside commands and artifacts. Success caches retain model, endpoint, payload, and version identity. Failed batches remain pending rather than being reported as complete.

Context is committed text; the reading describes only the conversion target. Conversion readings preserve Japanese orthographic particles rather than phonetic substitutions. Accepted outputs share the intended reading and meaning; unrelated homophones are not automatically equivalent. Real candidates are exported by AzooKey, never fabricated from reference answers.

## Completed historical set

Eighteen of twenty batches yield 180 drafts; missing batches are not replenished. Offline export uses `export_cached_development.py`. AI-assisted review retains 137 cases, quarantines 43, and removes reading-incompatible alternatives from nine cases without changing input/context. Sixty-three retained cases have context and 74 do not. Review decisions reside in `artifacts/benchmarks/ime-dev-reviewed-v2/`.

Outputs preserve drafts, quarantine, verification evidence, raw responses, failures, pending jobs, parameters, and distribution records. New generation/prompt/model versions use distinct directories. External seed parameters do not guarantee deterministic API responses.

`hybrid.py prepare-dev --labels-reviewed` records a review declaration; it does not perform review or certify human gold quality. [Hybrid ranking](hybrid-ranking.md) records the completed calibration.

## Optional synthetic diagnostics

`generate.py` and `evaluate.py` retain early controlled homophone/phrase experiments with separate generation and verification models. They do not supply real AzooKey pools or measure unrestricted completion quality. Incomplete historical synthetic caches are excluded from the current model baseline.
