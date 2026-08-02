# Football Passing Risk–Reward Analysis

This repository provides a reproducible event-data pipeline for separating four components of football passing performance:

1. pass-completion probability (`xPass`);
2. successful-state gain;
3. failure exposure;
4. execution relative to model expectation.

The primary analysis contains **3,396,912 open-play passes from 3,926 matches**. Predictive models are split and cross-fitted at match level. The downstream applications include full-sample risk–reward patterns, pressure and pass-height comparisons, three prespecified team-seasons, player-level matched contrasts, and sensitivity analyses.

## Repository structure

```text
.
├── config/                 # Locked xPass tuning selections
├── data/
│   ├── manifests/          # Fixed competition, match-split, and team-season manifests
│   ├── raw/                # Large raw files are generated locally and not committed
│   └── processed/          # Large intermediate files are generated locally and not committed
├── docs/
│   └── reproducibility.md
├── reference_outputs/      # Compact audit and summary outputs
├── scripts/                # Numbered analysis modules
├── .gitignore
├── README.md
└── requirements.txt
```

## Data source

Raw events are obtained from StatsBomb Open Data, subject to the provider's terms of use. Large raw and pass-level intermediate files are not stored in this repository. The fixed manifests in `data/manifests/` identify the competition-seasons, matches, model splits, and three-team application used in the analysis.

## Environment

Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The recorded software versions used for the frozen analysis are documented in [`docs/reproducibility.md`](docs/reproducibility.md).

## Reproduction workflow

The commands below use `outputs/` for generated files. That directory is ignored by Git.

### 1. Download the fixed match sample

For exact sample reconstruction, use both committed manifests:

```bash
python scripts/01_fetch_statsbomb_passes.py \
  --output_dir data/raw \
  --output_csv passes_all_matches_fixed.csv \
  --competitions_manifest data/manifests/competitions_selected.csv \
  --match_manifest data/manifests/xpass_match_split_manifest.csv \
  --expected_matches 3926 \
  --expected_pass_rows 3806977
```

A small smoke test can instead use `--max_matches 20` and omit the expected-count arguments.

### 2. Prepare the primary and strict open-play samples

```bash
python scripts/02_prepare_pass_data.py \
  --input_csv data/raw/passes_all_matches_fixed.csv \
  --output_dir outputs/02_preprocessing
```

Expected output:

```text
outputs/02_preprocessing/passes_preprocessed_with_open_play_flags.csv
```

### 3. Audit the xPass target

```bash
python scripts/03_audit_xpass_target.py \
  --input_csv outputs/02_preprocessing/passes_preprocessed_with_open_play_flags.csv \
  --competitions_csv data/manifests/competitions_selected.csv \
  --output_dir outputs/03_target_audit
```

The committed `data/manifests/xpass_match_split_manifest.csv` is the locked split used in the study. It contains 2,747 training, 395 tuning, 394 calibration, and 390 test matches.

To regenerate the deterministic split from the target audit:

```bash
python scripts/04_create_match_splits.py \
  --match_audit_csv outputs/03_target_audit/match_level_target_audit.csv \
  --output_dir outputs/04_match_splits
```

### 4. Tune the main and no-team xPass models

```bash
python scripts/05_tune_xpass.py \
  --variant main \
  --input_csv outputs/02_preprocessing/passes_preprocessed_with_open_play_flags.csv \
  --split_manifest data/manifests/xpass_match_split_manifest.csv \
  --output_dir outputs/05_tune_main

python scripts/05_tune_xpass.py \
  --variant no-team \
  --input_csv outputs/02_preprocessing/passes_preprocessed_with_open_play_flags.csv \
  --split_manifest data/manifests/xpass_match_split_manifest.csv \
  --output_dir outputs/05_tune_no_team
```

The frozen selections are also provided in `config/`.

### 5. Run the locked independent-test xPass evaluation

```bash
python scripts/06_xpass.py evaluate \
  --input_csv outputs/02_preprocessing/passes_preprocessed_with_open_play_flags.csv \
  --split_manifest data/manifests/xpass_match_split_manifest.csv \
  --main_tuning_json config/xpass_main_tuning_selection.json \
  --no_team_tuning_json config/xpass_no_team_tuning_selection.json \
  --competitions_csv data/manifests/competitions_selected.csv \
  --output_dir outputs/06_xpass_evaluation \
  --bootstrap_reps 1000 \
  --confirm_test_evaluation
```

Generate evaluation tables, reliability results, paired model differences, and grouped permutation importance:

```bash
python scripts/07_xpass_reporting.py \
  --evaluation_dir outputs/06_xpass_evaluation \
  --output_dir outputs/07_xpass_reporting

python scripts/08_paired_bootstrap_differences.py \
  --predictions_csv outputs/06_xpass_evaluation/final_test_event_predictions.csv \
  --evaluation_record_json outputs/06_xpass_evaluation/final_test_evaluation_record.json \
  --output_dir outputs/08_paired_bootstrap

python scripts/09_grouped_feature_importance.py \
  --xpass_script scripts/06_xpass.py \
  --model_bundle outputs/06_xpass_evaluation/xpass_final_model_bundle.joblib \
  --input_csv outputs/02_preprocessing/passes_preprocessed_with_open_play_flags.csv \
  --split_manifest data/manifests/xpass_match_split_manifest.csv \
  --reference_predictions_csv outputs/06_xpass_evaluation/final_test_event_predictions.csv \
  --output_dir outputs/09_grouped_importance \
  --repetitions 30 \
  --permutation_scope within_match

python scripts/12_plot_grouped_permutation_importance.py \
  --input_csv outputs/09_grouped_importance/grouped_permutation_importance_summary.csv \
  --output_dir outputs/12_grouped_importance_figure
```

### 6. Generate match-level out-of-fold xPass predictions

```bash
python scripts/06_xpass.py crossfit \
  --input_csv outputs/02_preprocessing/passes_preprocessed_with_open_play_flags.csv \
  --split_manifest data/manifests/xpass_match_split_manifest.csv \
  --main_tuning_json config/xpass_main_tuning_selection.json \
  --no_team_tuning_json config/xpass_no_team_tuning_selection.json \
  --output_dir outputs/06_xpass_oof \
  --outer_folds 5

python scripts/06_xpass.py verify \
  --oof_dir outputs/06_xpass_oof
```

### 7. Build H=1, H=3, and H=5 PCTV pass-value files

```bash
python scripts/10_threat_value_pctv.py build \
  --pass_csv outputs/02_preprocessing/passes_preprocessed_with_open_play_flags.csv \
  --oof_csv outputs/06_xpass_oof/oof_xpass_predictions.csv \
  --output_dir outputs/10_pctv_h1 \
  --horizon 1 \
  --output_format csv.gz

python scripts/10_threat_value_pctv.py build \
  --pass_csv outputs/02_preprocessing/passes_preprocessed_with_open_play_flags.csv \
  --oof_csv outputs/06_xpass_oof/oof_xpass_predictions.csv \
  --output_dir outputs/10_pctv_h3 \
  --horizon 3 \
  --output_format csv.gz

python scripts/10_threat_value_pctv.py build \
  --pass_csv outputs/02_preprocessing/passes_preprocessed_with_open_play_flags.csv \
  --oof_csv outputs/06_xpass_oof/oof_xpass_predictions.csv \
  --output_dir outputs/10_pctv_h5 \
  --horizon 5 \
  --output_format csv.gz
```

PCTV calibration diagnostics:

```bash
python scripts/11_pctv_calibration_diagnostics.py \
  --h1 outputs/10_pctv_h1/pass_value_oof.csv.gz \
  --h3 outputs/10_pctv_h3/pass_value_oof.csv.gz \
  --h5 outputs/10_pctv_h5/pass_value_oof.csv.gz \
  --output_dir outputs/11_pctv_calibration
```

### 8. Full-sample and three-team applications

```bash
python scripts/17_full_sample_analysis.py \
  --pass_value_csv outputs/10_pctv_h3/pass_value_oof.csv.gz \
  --pass_metadata_csv outputs/02_preprocessing/passes_preprocessed_with_open_play_flags.csv \
  --output_dir outputs/17_full_sample \
  --bootstrap_reps 1000

python scripts/18_three_team_analysis.py \
  --pass_value_csv outputs/10_pctv_h3/pass_value_oof.csv.gz \
  --pass_metadata_csv outputs/02_preprocessing/passes_preprocessed_with_open_play_flags.csv \
  --cohort_manifest_csv data/manifests/three_team_match_manifest.csv \
  --output_dir outputs/18_three_team \
  --bootstrap_reps 1000
```

### 9. Team and player summaries

```bash
python scripts/13_team_player_analysis.py \
  --pass_value outputs/10_pctv_h3/pass_value_oof.csv.gz \
  --cohort_csv outputs/18_three_team/three_team_event_cohort.csv.gz \
  --output_dir outputs/13_team_player \
  --bootstrap_reps 1000

python scripts/14_finalize_team_player_tables.py \
  --team_input outputs/13_team_player/team_summary.csv \
  --player_input outputs/13_team_player/player_summary.csv \
  --output_dir outputs/14_team_player_final

python scripts/19_prepare_player_events.py \
  --event_cohort outputs/18_three_team/three_team_event_cohort.csv.gz \
  --metadata_csv outputs/02_preprocessing/passes_preprocessed_with_open_play_flags.csv \
  --output_dir outputs/19_player_events

python scripts/20_player_decision_value_analysis.py \
  --player_event_csv outputs/19_player_events/three_team_player_event_data.csv.gz \
  --player_summary_csv outputs/14_team_player_final/player_summary_final.csv \
  --output_dir outputs/20_player_analysis
```

### 10. Sensitivity and robustness analysis

```bash
python scripts/15_robustness_analysis.py \
  --h1_pass_value outputs/10_pctv_h1/pass_value_oof.csv.gz \
  --h3_pass_value outputs/10_pctv_h3/pass_value_oof.csv.gz \
  --h5_pass_value outputs/10_pctv_h5/pass_value_oof.csv.gz \
  --cohort_csv outputs/18_three_team/three_team_event_cohort.csv.gz \
  --output_dir outputs/15_robustness \
  --player_thresholds 100,200,300

python scripts/16_robustness_reporting.py \
  --robustness_summary_json outputs/15_robustness/robustness_summary.json \
  --robustness_stability_csv outputs/15_robustness/robustness_stability.csv \
  --robustness_player_results_csv outputs/15_robustness/robustness_player_results.csv \
  --robustness_team_results_csv outputs/15_robustness/robustness_team_results.csv \
  --selected_player_pairs_csv outputs/20_player_analysis/selected_player_pairs.csv \
  --output_dir outputs/16_robustness_reporting
```

## Primary definitions

For pass `i`:

```text
success value       = T_end − T_start
failure exposure    = T_start + T_opponent
expected success    = p_success × success value
expected failure    = (1 − p_success) × failure exposure
expected net value  = expected success − expected failure
execution residual  = observed success − p_success
```

The risk-bin curves are descriptive. No universal or prescriptive passing-risk threshold is inferred.

## Reference outputs and integrity checks

Compact summaries and audit records are included under `reference_outputs/`. Most modules also write JSON audit records, software versions, seeds, input/output hashes, and SHA-256 manifests. See [`docs/reproducibility.md`](docs/reproducibility.md) for expected counts and verification guidance.

## Citation and license

Please cite the associated study when using this workflow. No software license has been assigned in this repository unless a `LICENSE` file is added explicitly by the repository owner. The underlying StatsBomb data remain subject to their own terms of use.
