# Stage 06-A Self Check

## Scope

This stage adds the training controls needed before stability validation:

- random seed control
- per-epoch validation
- early stopping
- best-checkpoint saving

## Implemented Changes

- `train_torch.py` now supports `--seed`.
- Validation runs after every epoch when `--valid-path` is provided.
- `--early-stop-metric` selects the metric to monitor.
- `--early-stop-mode` supports `max` and `min`.
- `--early-stop-patience` stops training after non-improving epochs.
- `--checkpoint-output` now saves the best checkpoint, not just the final one.
- Report JSON contains full `history` plus `best_*` metrics.

## Smoke Checks

### DeepFM 10k

Command validated:

- model: DeepFM
- data: `joined_stage03_10k`
- seed: 2024
- monitor: `valid_ctr_auc`
- patience: 1

Result:

- early stopping loop completed normally
- best epoch: 3
- best `valid_ctr_auc`: 0.534545
- checkpoint saved to `D:\Ali-CCP\models\deepfm_stage06a_10k_smoke.pt`

### ESMM + DCN-V2 100k

Command validated:

- model: ESMM + DCN-V2
- data: `joined_stage03_100k`
- seed: 2024
- monitor: `valid_ctcvr_pr_auc`
- patience: 1

Result:

- early stopping loop completed normally
- best epoch: 3
- best `valid_ctcvr_pr_auc`: 0.001342
- checkpoint saved to `D:\Ali-CCP\models\esmm_stage06a_100k_smoke.pt`

## Self-Check Result

Stage 06-A is complete. Multi-seed stability experiments can now use best
checkpoints and per-epoch validation instead of fixed-epoch final checkpoints.

## Recommended Stage 06-B Runs

Use the 1M joined split and run three seeds:

- DeepFM: monitor `valid_ctr_auc`
- ESMM + DCN-V2 w5: monitor `valid_ctcvr_pr_auc`

Suggested seeds:

- 2024
- 2025
- 2026

