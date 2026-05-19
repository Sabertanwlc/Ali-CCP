# Stage 06-C Self Check

## Scope

This stage exports validation predictions from the best available DeepFM and
ESMM w5 checkpoints, then performs error analysis and score-distribution
analysis.

## Exported Prediction Files

- DeepFM seed2025 predictions:
  `D:\Ali-CCP\reports\predictions_deepfm_seed2025_valid.csv`
- ESMM w5 seed2025 predictions:
  `D:\Ali-CCP\reports\predictions_esmm_w5_seed2025_valid.csv`

Each row contains:

- `sample_id`
- `click`
- `conversion`
- `ctcvr`
- `common_feature_id`
- `feature_count`
- `pctr`
- `pcvr` for ESMM
- `pctcvr` for ESMM

## Analysis Outputs

DeepFM:

- `D:\Ali-CCP\reports\analysis_deepfm_seed2025\summary.json`
- `high_pctr_not_clicked.csv`
- `low_pctr_clicked.csv`

ESMM w5:

- `D:\Ali-CCP\reports\analysis_esmm_w5_seed2025\summary.json`
- `high_pctr_not_clicked.csv`
- `low_pctr_clicked.csv`
- `high_pctcvr_not_converted.csv`
- `low_pctcvr_converted.csv`
- `low_pcvr_clicked_converted.csv`

## DeepFM Summary

- rows: 200,433
- clicks: 9,304
- conversions: 64
- avg feature count: 617.77
- CTR AUC: 0.629902
- CTR PR-AUC: 0.077298
- CTR LogLoss: 0.183460
- CTR Lift@1%: 2.687459
- CTR Lift@5%: 2.201347
- CTR Lift@10%: 2.049687

Calibration:

- Most samples are in prediction bin [0.0, 0.1).
- Bin [0.0, 0.1): avg prediction 0.04858, true CTR 0.04519.
- Bin [0.1, 0.2): avg prediction 0.10887, true CTR 0.11877.

Interpretation:

- DeepFM is reasonably calibrated for CTR.
- It is the strongest pure CTR model so far.
- Its top-ranked samples are more than 2.6x richer in clicks than the average
  validation population.

## ESMM w5 Summary

- rows: 200,433
- clicks: 9,304
- conversions: 64
- CTCVR positives: 62
- avg feature count: 617.77

CTR:

- CTR AUC: 0.615697
- CTR PR-AUC: 0.074529
- CTR LogLoss: 0.194537
- CTR Lift@1%: 3.181951
- CTR Lift@5%: 2.106757
- CTR Lift@10%: 1.887389

CTCVR:

- CTCVR AUC: 0.547937
- CTCVR PR-AUC: 0.000645
- CTCVR LogLoss: 0.003832
- CTCVR Lift@1%: 6.452675
- CTCVR Lift@5%: 2.903414
- CTCVR Lift@10%: 2.258098

CVR on clicked samples:

- clicked rows: 9,304
- clicked conversions: 62
- CVR AUC: 0.508461
- CVR PR-AUC: 0.009322
- CVR LogLoss: 0.060915
- CVR Lift@1%: 4.840791

Interpretation:

- ESMM w5 is weaker than DeepFM on general CTR AUC.
- ESMM w5 is better for top-of-list chain objectives: CTR Lift@1% and CTCVR
  Lift@1% are both strong.
- CVR ranking remains weak by AUC because clicked conversions are still sparse,
  but top 1% clicked CVR lift is meaningfully above random.

## Error Analysis Interpretation

High pCTR but not clicked:

- These rows represent false positives for click ranking.
- In recommendation logs, many are likely exposure noise or attractive ad/user
  combinations that did not receive attention.
- These should be inspected for repeated common ids or unusually high feature
  counts.

Low pCTR but clicked:

- These rows represent missed click opportunities.
- They are useful for identifying feature groups where the model under-scores
  a specific user/ad pattern.

High pCTCVR but not converted:

- These are conversion false positives.
- They matter because over-ranking them can waste recommendation slots.
- ESMM w5 keeps CTCVR probabilities low overall, so this list is mainly useful
  for qualitative inspection rather than threshold-based deployment.

Low pCTCVR but converted:

- These are the most valuable missed conversion cases.
- Given only 62 validation CTCVR positives, each row can have a large effect on
  CVR/CTCVR metrics.

## Stage Result

- Prediction export works for DeepFM and ESMM.
- Error sample files are available for manual inspection.
- DeepFM remains best for CTR probability/ranking.
- ESMM w5 remains best for final chain-oriented ranking because CTCVR Lift@1%
  is much stronger than the base rate.

## Next Stage

Stage 06-D should generate the final project report:

- consolidate metric tables
- include Lift@K interpretation
- include calibration interpretation
- include error-analysis findings
- state final model selection and limitations

