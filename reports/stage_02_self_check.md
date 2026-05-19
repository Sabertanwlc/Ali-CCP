# Stage 02 Self Check

## Scope

This stage adds `common_features_*` to the feature pipeline. The goal is to
validate that `sample_skeleton_*` can be joined with `common_features_*` through
`common_feature_id` and that `ad + common` features improve the LR Hashing
baseline after sane feature scaling.

## Feasibility Decisions

- Full common join cannot load `common_features_train.csv` into memory.
- `src.data.bucket_join` implements the scalable path: split both files by
  `hash(common_feature_id) % bucket_count`, then join one bucket at a time.
- `src.data.selective_join` implements the validation path: collect the required
  common ids for a small skeleton sample, then scan common features until all are
  found.
- Joined rows keep the same label columns and replace the final sparse feature
  string with `ad_features + common_features`.
- Sample-level L2 normalization and value clipping are needed because common
  rows have hundreds of features. Without normalization, online LR can become
  unstable.

## Join Checks

- Bucket smoke test:
  - skeleton rows: 1,000
  - common rows scanned: 10,000
  - joined rows: 0
  - interpretation: expected limitation of truncating common rows; the first
    1,000 skeleton rows reference common ids that appear later in the common file.
- Selective join validation:
  - skeleton rows: 1,000
  - required common ids: 13
  - matched common ids: 13
  - scanned common rows: 707,394
  - joined rows: 1,000
  - join coverage: 1.0

## LR Hashing Comparison

Raw joined features were intentionally tested first and failed the quality check:

- `ad-only`, no normalization:
  - AUC: 0.504584
  - LogLoss: 0.178282
- `ad + common`, no normalization:
  - AUC: 0.436096
  - LogLoss: 1.663458

After value clipping and sample-level L2 normalization:

- `ad-only`, `--value-clip 10 --normalize-l2`:
  - AUC: 0.535996
  - LogLoss: 0.177392
- `ad + common`, `--value-clip 10 --normalize-l2`:
  - AUC: 0.583419
  - LogLoss: 0.166655

## PyTorch Pipeline Check

- DeepFM joined smoke test:
  - rows: 64
  - feature source: joined
  - normalization: enabled
  - avg loss: 0.370266

## Self-Check Result

- Join key is valid.
- Joined format is parseable by the shared feature stream.
- `common_features_*` improves LR Hashing once feature scale is controlled.
- DeepFM can consume joined samples, so Stage 03 can proceed to neural baseline
  training.

## Next Stage

Stage 03 should scale from 1,000 joined samples to a larger joined sample, then
train DeepFM with `ad + common` features and compare it against normalized LR
Hashing on the same sample.

