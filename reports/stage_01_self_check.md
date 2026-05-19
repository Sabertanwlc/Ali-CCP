# Stage 01 Self Check

## Scope

This stage verifies that the project can parse Ali-CCP raw files, inspect labels
and sparse fields, and run small-sample training for the planned model ladder:

- LR Hashing
- DeepFM
- ESMM + DCN-V2

## Feasibility Decisions

- Raw files are around 40GB total, so all scripts use streaming reads.
- The first baseline uses only `sample_skeleton_*` ad-side features. This is a
  deliberate first step because joining `common_features_*` should be handled
  separately with buckets or external sorting.
- Feature hashing is used before building a full feature dictionary. This avoids
  a memory-heavy global vocabulary pass.
- DeepFM and ESMM share the same hashed sparse input so that model comparisons
  are controlled by model structure rather than different feature pipelines.

## Completed Checks

- Parser synthetic self-check: passed.
- `sample_skeleton_train.csv` first 1,000 rows: parsed 1,000 rows, 0 errors.
- `common_features_train.csv` first 1,000 rows: parsed 1,000 rows, 0 errors.
- Label check on first 1,000 train skeleton rows:
  - click rate: 0.032
  - conversion rate: 0.0
  - conversion without click: 0
- LR Hashing on first 1,000 rows:
  - AUC: 0.504584
  - LogLoss: 0.178282
- DeepFM smoke test on first 64 rows: completed one epoch on CPU.
- ESMM + DCN-V2 smoke test on first 64 rows: completed one epoch on CPU.

## Current Limitations

- LR baseline currently uses ad-side features only.
- DeepFM and ESMM smoke tests only validate tensor flow and loss computation.
- No full validation split has been materialized yet.
- `common_features_*` join is not implemented in this stage.

## Next Stage

Implement bucketed join for `sample_skeleton_*` and `common_features_*`, then
rerun LR Hashing with ad + common features before scaling neural models.

