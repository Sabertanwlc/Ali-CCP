# Stage 06-B Stability Experiment Plan

## Goal

Validate whether the current model-selection conclusion is stable across random
seeds.

Current conclusion to verify:

- DeepFM is the best pure CTR model.
- ESMM + DCN-V2 w5 is the best final CTR/CVR/CTCVR chain model.

## Data

Use the 1M joined split:

- train: `D:\Ali-CCP\processed\joined_stage05_1m\train.csv`
- valid: `D:\Ali-CCP\processed\joined_stage05_1m\valid.csv`

## Seeds

- 2024
- 2025
- 2026

## DeepFM Runs

Monitor `valid_ctr_auc`.

```powershell
C:\Users\WLC\AppData\Local\Programs\Python\Python313\python.exe -m src.train.train_torch `
  --model deepfm `
  --feature-source joined `
  --train-path D:\Ali-CCP\processed\joined_stage05_1m\train.csv `
  --valid-path D:\Ali-CCP\processed\joined_stage05_1m\valid.csv `
  --hash-size 1048576 `
  --embedding-dim 16 `
  --hidden-units 256,128,64 `
  --batch-size 2048 `
  --epochs 5 `
  --seed <SEED> `
  --learning-rate 0.001 `
  --device cuda `
  --normalize-l2 `
  --value-clip 10 `
  --early-stop-metric valid_ctr_auc `
  --early-stop-mode max `
  --early-stop-patience 1 `
  --output D:\Ali-CCP\reports\stage_06b_deepfm_1m_seed<SEED>.json `
  --checkpoint-output D:\Ali-CCP\models\deepfm_stage06b_1m_seed<SEED>.pt
```

## ESMM + DCN-V2 w5 Runs

Monitor `valid_ctcvr_pr_auc`.

```powershell
C:\Users\WLC\AppData\Local\Programs\Python\Python313\python.exe -m src.train.train_torch `
  --model esmm_dcnv2 `
  --feature-source joined `
  --train-path D:\Ali-CCP\processed\joined_stage05_1m\train.csv `
  --valid-path D:\Ali-CCP\processed\joined_stage05_1m\valid.csv `
  --hash-size 1048576 `
  --embedding-dim 16 `
  --hidden-units 256,128,64 `
  --cross-layers 3 `
  --batch-size 2048 `
  --epochs 5 `
  --seed <SEED> `
  --learning-rate 0.001 `
  --device cuda `
  --normalize-l2 `
  --value-clip 10 `
  --ctr-loss-weight 1 `
  --ctcvr-loss-weight 5 `
  --early-stop-metric valid_ctcvr_pr_auc `
  --early-stop-mode max `
  --early-stop-patience 1 `
  --output D:\Ali-CCP\reports\stage_06b_esmm_w5_1m_seed<SEED>.json `
  --checkpoint-output D:\Ali-CCP\models\esmm_w5_stage06b_1m_seed<SEED>.pt
```

## Success Criteria

- DeepFM CTR AUC remains near the Stage 05 value around 0.620.
- ESMM w5 CTR AUC remains close to DeepFM.
- ESMM w5 CTCVR Lift@1% and CTCVR PR-AUC remain directionally better than
  alternative ESMM settings from Stage 05.
- If seed variance is large, final reporting must weaken the claim and frame the
  result as preliminary.

