# Historical V1 conversion command reference

This pre-measurement V1 workflow is retained as historical tooling context. Current V2.1 assets and runtime restrictions appear in [Core ML](../coreml.md).

`scripts/deployment/coreml.py` exposes V1 export/reference/conversion/validation/compression/evaluation/phrase/timing operations. The checkpoint, tokenizer, and manifest identity must match. Runtime conversion and prediction require macOS; Windows preserves reports but does not reproduce Core ML predictions.

```bash
python scripts/deployment/coreml.py --help
```

Original outputs distinguish uncompressed conversion from quantized variants, retain failed alignment records, and preserve generation/ranking measurements. Later V2 tools use `coreml_v2.py` and `package_v2_ios.py`; historical V1 manifests are not retroactively regenerated with updated tool fingerprints.

Initial FP16/palette expectations were design hypotheses. Recorded V1 deployment uses INT8 weight compression, FP32 calculation, and CPU-only execution. Candidate score equivalence, model-level numerical alignment, package size, and device memory remain distinct measurements.
