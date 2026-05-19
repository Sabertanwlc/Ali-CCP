# Stage 06-B Self Check

## Scope

This stage runs additional seed-based stability checks on the 1M joined split.

The goal is to validate whether the Stage 05 conclusion is robust:

- DeepFM is the best pure CTR model.
- ESMM + DCN-V2 w5 is the best final chain model for CTR/CVR/CTCVR.

## Completed Runs

### DeepFM seed 2025

- monitor: `valid_ctr_auc`
- early stopping patience: 1
- epochs completed: 2
- best epoch: 1
- best CTR AUC: 0.629902
- best CTR PR-AUC: 0.077298
- best CTR LogLoss: 0.183460
- best CTR Lift@1%: 2.687459
- checkpoint: `D:\Ali-CCP\models\deepfm_stage06b_1m_seed2025.pt`

Stage 05 DeepFM seed 2024 reference:

- CTR AUC: 0.619959
- CTR PR-AUC: 0.074665
- CTR LogLoss: 0.187340
- CTR Lift@1%: 2.870206

Interpretation:

- DeepFM is stable directionally: seed 2025 is not worse than seed 2024.
- CTR AUC improves with seed 2025, while Lift@1% is lower than seed 2024.
- The conclusion that DeepFM is a strong CTR baseline remains valid.

### ESMM + DCN-V2 w5 seed 2025

- monitor: `valid_ctcvr_pr_auc`
- early stopping patience: 1
- epochs completed: 3
- best epoch: 2
- best CTR AUC: 0.615697
- best CTR PR-AUC: 0.074529
- best CTR LogLoss: 0.194537
- best CTR Lift@1%: 3.181951
- best CTCVR AUC: 0.547937
- best CTCVR PR-AUC: 0.000645
- best CTCVR Lift@1%: 6.452675
- best CVR-on-clicked AUC: 0.508461
- checkpoint: `D:\Ali-CCP\models\esmm_w5_stage06b_1m_seed2025.pt`

Stage 05 ESMM w5 seed 2024 reference:

- CTR AUC: 0.619733
- CTR PR-AUC: 0.075187
- CTR LogLoss: 0.192130
- CTR Lift@1%: 2.934705
- CTCVR AUC: 0.580375
- CTCVR PR-AUC: 0.000800
- CTCVR Lift@1%: 6.452675
- CVR-on-clicked AUC: 0.554150

Interpretation:

- ESMM w5 is stable on CTCVR Lift@1%, which stays at 6.45.
- ESMM w5 is less stable on CTCVR AUC and CVR-on-clicked AUC.
- This is expected because valid has only 62 CTCVR positives.
- The final claim should emphasize Lift@K and PR-AUC rather than overclaiming
  CVR AUC stability.

## Current Stability Judgment

- DeepFM CTR conclusion is stable enough for project reporting.
- ESMM w5 remains the best chain-model candidate, but conversion metrics are
  still sparse and seed-sensitive.
- One more ESMM seed is useful if time allows, but not strictly required for a
  project report if limitations are stated clearly.

## Recommendation

Proceed to Stage 06-C:

- export predictions for DeepFM seed 2025 and ESMM w5 seed 2025
- run error analysis
- generate final report tables

Optional:

- run ESMM w5 seed 2026 if stronger stability evidence is required.

