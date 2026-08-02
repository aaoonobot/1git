# Pipeline update: stage 1

This package contains the validated upstream portion of the revised analysis pipeline.

## Scripts

1. `scripts/02_prepare_pass_data.py`
   - Parses StatsBomb coordinates.
   - Retains all original columns.
   - Adds primary and strict open-play indicators.
   - Writes coordinate and sample-flow audits.

2. `scripts/03_audit_xpass_target.py`
   - Audits the pass-completion target and class balance.
   - Produces match-level and subgroup summaries.

3. `scripts/04_create_match_splits.py`
   - Creates the deterministic 70/10/10/10 match-level split.
   - Uses competition gender and match-date quantile bands for stratification.

4. `scripts/05_tune_xpass.py`
   - Tunes the main and no-team xPass variants.
   - Uses training and tuning matches only.
   - Does not read calibration or test matches.

## Reproducibility records

- `data/manifests/competitions_selected.csv`
- `data/manifests/match_level_target_audit.csv`
- `data/manifests/xpass_match_split_manifest.csv`
- `config/xpass_main_tuning_selection.json`
- `config/xpass_no_team_tuning_selection.json`
- `supplementary_outputs/*.json`
- `SHA256SUMS.txt`

Local absolute paths have been removed from the public copies of the metadata files.
The exact 3,926-match study cohort is fixed by the match-level manifest.
