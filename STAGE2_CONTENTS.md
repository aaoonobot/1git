# Stage 2: Core modeling and diagnostics

This package adds the current xPass and PCTV analysis modules to the `pipeline-update` branch.

## Files

- `scripts/06_xpass.py`: locked independent-test evaluation, full-sample match-level OOF cross-fitting, and OOF verification.
- `scripts/07_xpass_reporting.py`: independent-test tables, reliability figures, and high-risk tail calibration summaries.
- `scripts/08_paired_bootstrap_differences.py`: paired match-cluster bootstrap differences between the full xPass model and its comparators.
- `scripts/09_grouped_feature_importance.py`: within-match grouped permutation importance for the locked full xPass model.
- `scripts/10_threat_value_pctv.py`: match-level cross-fitted pass-chain threat and pass-value construction.
- `scripts/11_pctv_calibration_diagnostics.py`: pooled OOF PCTV calibration intercept, slope, ECE, and reliability curves for H=1, H=3, and H=5.

## Important

Do not delete the older scripts yet. They will be removed only after the downstream team, player, and robustness modules have been uploaded and the README has been replaced.

Do not upload any `__pycache__` directory or `.pyc` files.
