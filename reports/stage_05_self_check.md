# Stage 05 Self Check

## Scope

This stage adds the final evaluation metrics needed for model selection and
scales joined training data to 1M rows. The goal is to choose between LR
Hashing, DeepFM, and ESMM + DCN-V2 using CTR ranking, CTCVR ranking, Lift, and
calibration.

## New Metrics

- ROC-AUC for ranking quality.
- PR-AUC for sparse positive classes.
- `rate@1%/5%/10%` and `lift@1%/5%/10%` for top-ranked business value.
- Calibration bins for probability sanity checks.

## 1M Joined Dataset

- joined rows: 1,000,000
- required common ids: 18,827
- matched common ids: 18,827
- join coverage: 1.0
- train rows: 799,567
- valid rows: 200,433
- train clicks: 37,449
- valid clicks: 9,304
- train conversions: 213
- valid conversions: 64
- valid CTCVR positives: 62

## CTR Results

All models use joined `ad + common` features with `--normalize-l2 --value-clip 10`.

| Model | CTR AUC | CTR PR-AUC | CTR LogLoss | CTR Lift@1% | CTR Lift@5% | CTR Lift@10% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LR Hashing | 0.606167 | 0.072442 | 0.189058 | 2.805707 | 2.096009 | 1.848695 |
| DeepFM | 0.619959 | 0.074665 | 0.187340 | 2.870206 | 2.190598 | 1.894912 |
| ESMM + DCN-V2 w5 | 0.619733 | 0.075187 | 0.192130 | 2.934705 | 2.121806 | 1.932531 |
| ESMM + DCN-V2 posw25 | 0.618270 | 0.072177 | 0.197740 | 2.343464 | 2.085260 | 1.858368 |

## CTCVR Results

| Model | CTCVR AUC | CTCVR PR-AUC | CTCVR LogLoss | CTCVR Lift@1% | CTCVR Lift@5% | CTCVR Lift@10% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ESMM + DCN-V2 w5 | 0.580375 | 0.000800 | 0.003643 | 6.452675 | 2.258211 | 1.935513 |
| ESMM + DCN-V2 posw25 | 0.575332 | 0.000590 | 0.008045 | 4.839506 | 1.935609 | 1.935513 |

## CVR On Clicked Results

| Model | CVR AUC | CVR PR-AUC | CVR LogLoss | CVR Lift@1% | CVR Lift@5% | CVR Lift@10% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ESMM + DCN-V2 w5 | 0.554150 | 0.009575 | 0.057968 | 4.840791 | 1.613597 | 1.290878 |
| ESMM + DCN-V2 posw25 | 0.550387 | 0.011079 | 0.137448 | 1.613597 | 1.613597 | 1.774957 |

## Calibration Notes

- LR Hashing is conservative: almost all CTR predictions are below 0.1.
- DeepFM has better spread and the best CTR LogLoss among trained 1M neural
  models.
- ESMM w5 has strong top-1% CTR Lift and the best CTCVR Lift, but CTR LogLoss is
  worse than DeepFM.
- ESMM posw25 over-amplifies conversion probability. It improves one CVR PR-AUC
  number but damages CTR Lift@1%, CTR LogLoss, and CTCVR LogLoss.

## Self-Check Result

- 1M data makes CTR evaluation stable and gives a more useful, though still
  sparse, conversion validation set.
- DeepFM remains the strongest pure CTR probability/ranking baseline.
- ESMM + DCN-V2 w5 is the best final-model candidate because it nearly matches
  DeepFM on CTR and clearly beats posw25 on CTCVR Lift and calibration.
- Accuracy is still not useful: valid click rate is 4.64%, so a model predicting
  all negatives would get about 95.36% accuracy.

## Current Recommendation

- Use DeepFM if the project only needs CTR ranking.
- Use ESMM + DCN-V2 w5 as the final project model because the stated objective
  includes CTR, CVR, and CTCVR.
- Do not use posw25 as default. It is too poorly calibrated even though it
  increases some sparse conversion metrics.

## Next Work

- Add early stopping on validation CTR AUC and CTCVR PR-AUC.
- Try moderate CTCVR positive weights: 5, 10, and 15.
- Add model score export for offline error analysis.
- Consider expanding to more than 1M only if disk budget allows and conversion
  metrics remain too noisy.

