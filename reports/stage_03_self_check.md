# Stage 03 Self Check

## Scope

This stage scales the joined feature pipeline from smoke tests to a usable
DeepFM baseline. It also adds stable train/valid splitting and true validation
metrics for LR Hashing and PyTorch models.

## Feasibility Decisions

- The 10k joined sample was used first to validate runtime and file size.
- The 10k joined file was about 130 MB, so 100k was estimated at roughly 1.3 GB
  and considered feasible.
- The 100k joined generation completed quickly because the required common ids
  were found within the first 730,191 rows of `common_features_train.csv`.
- CUDA is available on this machine: NVIDIA GeForce RTX 4070 Laptop GPU.
- CTR is the reliable Stage 03 target. CVR/CTCVR remain sparse even at 100k:
  valid has only 8 conversion positives.

## New Utilities

- `src.data.split_joined`: stable train/valid split by `sample_id` hash.
- `src.train.train_lr_hashing_eval`: train LR on joined train and evaluate on
  joined valid.
- `src.train.train_torch`: now supports train/valid paths, validation AUC,
  validation LogLoss, checkpoint saving, and joined feature input.

## 10k Joined Probe

- joined rows: 10,000
- join coverage: 1.0
- train rows: 8,015
- valid rows: 1,985
- valid clicks: 83
- valid conversions: 0

Conclusion: 10k is useful for CTR smoke tests but too small for conversion
evaluation.

## 100k Joined Dataset

- joined rows: 100,000
- required common ids: 1,948
- matched common ids: 1,948
- scanned common rows: 730,191
- join coverage: 1.0

Stable split:

- train rows: 79,878
- valid rows: 20,122
- train click rate: 0.044643
- valid click rate: 0.041994
- train conversions: 36
- valid conversions: 8

## CTR Baseline Results

All runs use `ad + common` joined features with `--normalize-l2 --value-clip 10`.

| Model | Hash Size | Epochs | Valid AUC | Valid LogLoss | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| LR Hashing | 1,048,576 | 1 online pass | 0.610696 | 0.172164 | Best calibrated baseline |
| DeepFM | 1,048,576 | 1 | 0.573749 | 0.176364 | Under-trained ranking |
| DeepFM | 1,048,576 | 3 | 0.624124 | 0.237966 | Best ranking, poor calibration |
| DeepFM low LR | 1,048,576 | 2 | 0.568588 | 0.177887 | Too conservative |
| ESMM + DCN-V2 smoke | 1,048,576 | 1 | 0.569483 | 0.177492 | Entry verified only |

## Implementation Correction

DeepFM initially performed poorly because PyTorch embedding defaults are too
large for this high-dimensional sparse FM interaction. The model now uses:

- zero initialization for linear embedding weights
- small normal initialization for sparse embeddings
- Xavier initialization for dense layers

This changed DeepFM from an unstable smoke-test model into a usable ranking
baseline.

## Self-Check Result

- 100k joined data generation is feasible.
- Stable split has adequate CTR positives.
- LR Hashing remains the best probability-calibrated baseline.
- DeepFM now beats LR on AUC at 100k, so it is a valid neural ranking baseline.
- ESMM + DCN-V2 training entry is functional, but Stage 03 data is still too
  sparse for serious CVR/CTCVR evaluation.

## Next Stage

Stage 04 should focus on ESMM + DCN-V2 properly:

- expand to a larger joined sample, likely 500k or 1M if disk/time allow
- add explicit CTCVR and CVR-on-clicked validation metrics
- add class imbalance handling or loss weighting for CTCVR
- compare ESMM + DCN-V2 against DeepFM on CTR AUC and CTCVR metrics

