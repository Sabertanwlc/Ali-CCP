# Stage 04 Self Check

## Scope

This stage moves from the DeepFM CTR baseline to the final-model path:
`ESMM + DCN-V2`. It adds explicit CTR, CTCVR, and CVR-on-clicked validation
metrics, then scales joined data to 500k rows.

## Feasibility Decisions

- Stage 03 showed 100k had only 8 conversion positives in valid, so ESMM needed
  a larger sample before CVR/CTCVR metrics were meaningful.
- 500k joined generation was feasible and completed with full join coverage.
- `train_torch.py` initially defaulted to `--max-lines 100000`, which truncated
  large training runs. This was fixed by changing the default to no truncation.
- CTCVR task weighting alone was not enough because the conversion target is
  extremely sparse. Stage 04 added CTCVR positive weighting.

## 500k Joined Dataset

- joined rows: 500,000
- required common ids: 9,438
- matched common ids: 9,438
- join coverage: 1.0
- train rows: 399,871
- valid rows: 100,129
- train clicks: 18,633
- valid clicks: 4,513
- train conversions: 94
- valid conversions: 29
- valid CTCVR positives: 27

## Validation Results

All runs use joined `ad + common` features with `--normalize-l2 --value-clip 10`.

| Model | Epochs | CTR AUC | CTR LogLoss | CTCVR AUC | CTCVR LogLoss | CVR AUC clicked | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LR Hashing | 1 online pass | 0.607462 | 0.181300 | n/a | n/a | n/a | Strong calibrated CTR baseline |
| DeepFM | 2 | 0.624314 | 0.186955 | n/a | n/a | n/a | Best CTR AUC |
| ESMM + DCN-V2, w1 | 2 | 0.618970 | 0.189038 | 0.525128 | 0.003111 | 0.494733 | Balanced task weight |
| ESMM + DCN-V2, w5 | 2 | 0.623049 | 0.183871 | 0.534461 | 0.003343 | 0.493709 | Best ESMM CTR |
| ESMM + DCN-V2, posw100 | 2 | 0.599613 | 0.203014 | 0.554700 | 0.024985 | 0.471244 | Best CTCVR AUC, poor calibration |

## Self-Check Result

- ESMM + DCN-V2 now trains on the full 500k joined sample.
- ESMM w5 nearly matches DeepFM on CTR AUC: 0.623049 vs 0.624314.
- DeepFM remains the best CTR ranker at this stage.
- Positive weighting improves CTCVR AUC but damages CTR and probability
  calibration, so `posw100` is not the default final setting.
- CVR-on-clicked AUC is not yet reliable because valid has only 27 clicked
  conversions.

## Current Best Models

- Best CTR AUC: DeepFM 500k, AUC 0.624314.
- Best ESMM CTR tradeoff: ESMM + DCN-V2 w5, CTR AUC 0.623049.
- Best CTCVR AUC: ESMM + DCN-V2 posw100, CTCVR AUC 0.554700, but with worse
  calibration.

## Next Stage

Stage 05 should scale ESMM evaluation further and improve conversion learning:

- generate 1M joined rows if disk budget allows
- use more reliable CTCVR/CVR validation positives
- try moderate positive weights such as 10, 25, and 50 instead of 100
- add PR-AUC for CTCVR because conversion positives are extremely sparse
- consider separate early stopping criteria for CTR and CTCVR

