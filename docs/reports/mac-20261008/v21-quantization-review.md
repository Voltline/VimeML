# V2.1 quantization alignment review — 2026-10-08

The INT8 deployment package fails the original strict logits criterion while uncompressed conversion passes. The recorded tolerance is absolute 0.0003 plus relative 0.0003 × |reference|. Maximum errors are 0.0000457764 uncompressed and 1.00949144 INT8; T=128 INT8 mean absolute error is 0.108517, RMSE 0.139438.

## Interpretation

Weight compression perturbs distributions and candidate scores. A low-margin decision can change under modest score perturbations, while aggregate hit counts remain stable. Thirty changed expanded-set first choices have median FP32 margin approximately 0.070 and maximum 0.3628; 27 have margin at most 0.25. The 1,910 unchanged multi-candidate cases have median margin approximately 3.0718; 60 single-candidate cases are excluded from that margin comparison.

V1 INT8 also fails strict alignment, with maximum error approximately 3.66. Recurring failures do not establish an unavoidable small-model or architecture limitation. Uncompressed V2.1 agreement argues against a general conversion failure, but does not prove every compressed operation equivalent or identify a single cause.

## Task-level evidence

AJIMEE and historical-development hit counts stay at 144/200 and 122/137. AJIMEE includes three corrections and three regressions; one development accepted spelling changes. Expanded draft Top-1 falls 1,487 → 1,481, eight corrections/fourteen regressions, net -6 or -0.3 percentage points, exact paired p=0.286279. Nonsignificance is not a formal equivalence result; draft labels and limited coverage constrain the conclusion.

Fixed next-token choices match 12/12, but 32-token greedy trajectories match only 8/12. Candidate reranking and free generation have different perturbation sensitivity. Package size falls 50.36 → 14.33 MB. The compression is accepted for the recorded input-method deployment, with failure records preserved.

## Remaining boundaries

Numerical alignment, exact task labels, ranking changes, device response time, and long-term stability are separate measurements. No additional quantization method is established as universally inferior or superior. Limited real-device experience does not eliminate the approximately 1.93-second UI publication tail or demonstrate a sustained memory plateau. [Core ML record](v21-coreml.md) and [iPhone report](v21-iphone.md) retain evidence without attributing unmeasured mechanisms.
