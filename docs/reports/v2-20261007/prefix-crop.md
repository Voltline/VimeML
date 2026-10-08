# Deterministic prefix crop — 2026-10-07

`PrefixCropWindowDataset` affects V2 training only. With probability 0.30, an eligible first window is sliced from a nonzero offset while retaining at least eight prediction targets. The endpoint is unchanged; no retokenization or artificial BOS occurs. Short windows, later long-sentence windows, validation, and test remain unchanged.

Crop decisions use a deterministic seed/epoch/sample-index integer mix. Persistent workers receive epoch-tagged indices; bucket length uses the same crop rule. Resume starts at the committed batch cursor.

```text
seen windows = completed epochs × windows per epoch
trained tokens + dropped tokens = completed epochs × prediction pairs per epoch
```

Original targeted evidence uses 4,099 queried windows: 3,759 eligible, with 1,141/1,139 cropped in epochs 0/1 (30.354%/30.301%). It verifies bucket length, worker consistency, cursor resume, and a small interrupted-versus-continuous fixture. This query sample is not a full-corpus rate estimate.

Reports reside in `outputs/model-checks/v2-prefix-crop/report.json`. Online `prefix_crop_fraction` uses all windows as its denominator, so it can be below the configured eligible-window probability. These completed checks need not be repeated for documentation maintenance.
