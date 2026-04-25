# Football Passing Risk-Reward Reproducibility Package

This repository contains the code used to reproduce the data access, xPass estimation, out-of-fold xPass prediction, and pass-level risk-reward analysis reported in the manuscript.

The workflow is split into five command-line scripts:

1. `scripts/01_fetch_statsbomb_passes.py` downloads StatsBomb Open Data match metadata and pass events.
2. `scripts/02_train_xpass.py` trains and evaluates the final xPass model.
3. `scripts/03_generate_oof_xpass.py` generates match-level out-of-fold xPass probabilities for downstream analysis.
4. `scripts/04_compute_risk_reward.py` computes the risk-reward metrics, including risk, gain on success, failure cost, and expected reward.
5. `scripts/05_sensitivity_analysis.py` runs structural and formula-level sensitivity analyses.

The scripts do not contain local computer paths. All inputs and outputs are controlled with command-line arguments.

## 1. Data source and access

The raw event data used in the manuscript were derived from publicly available StatsBomb Open Data, subject to the provider's terms of use. Raw data are not included in this repository.

The data-access script downloads the available competition-season catalogue, retrieves match metadata, downloads event data match by match, and keeps pass events with the columns required by the downstream xPass and risk-reward pipeline.

The recommended output file for downstream analysis is:

```text
data/raw/passes_all_matches_fixed.csv
```

## 2. Environment

A minimal Python environment can be installed with:

```bash
pip install -r requirements.txt
```

The scripts were written for Python 3.10 or later. The main dependencies are:

- numpy
- pandas
- statsbombpy
- scikit-learn
- matplotlib
- joblib
- torch, only for the GRU continuation-value component in `04_compute_risk_reward.py`

## 3. Recommended project structure

```text
football-pass-risk-reward/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── passes_all_matches_fixed.csv
│   └── processed/
├── outputs/
│   ├── xpass_final/
│   ├── xpass_oof/
│   └── risk_reward/
└── scripts/
    ├── 01_fetch_statsbomb_passes.py
    ├── 02_train_xpass.py
    ├── 03_generate_oof_xpass.py
    ├── 04_compute_risk_reward.py
    └── 05_sensitivity_analysis.py
```

## 4. Step-by-step reproduction

### Step 0. Download and prepare pass-event data

This step downloads StatsBomb Open Data match metadata and extracts pass events from match event files. It also saves competition metadata, match metadata, excluded matches, and a download log.

```bash
python scripts/01_fetch_statsbomb_passes.py \
  --output_dir data/raw \
  --output_csv passes_all_matches_fixed.csv \
  --min_season_year 2000
```

Main outputs:

```text
data/raw/passes_all_matches_fixed.csv
data/raw/statsbomb_competitions_selected.csv
data/raw/statsbomb_matches_selected.csv
data/raw/statsbomb_matches_excluded.csv
data/raw/statsbomb_competition_download_log.csv
data/raw/statsbomb_event_download_log.csv
```

For a quick test run, use a small match cap:

```bash
python scripts/01_fetch_statsbomb_passes.py \
  --output_dir data/raw_test \
  --output_csv passes_sample.csv \
  --min_season_year 2000 \
  --max_matches 20
```

To resume an interrupted download:

```bash
python scripts/01_fetch_statsbomb_passes.py \
  --output_dir data/raw \
  --output_csv passes_all_matches_fixed.csv \
  --min_season_year 2000 \
  --resume
```

If a separate holdout case study is needed, matches involving a specific team, competition, and season can be excluded during the data download step. For example:

```bash
python scripts/01_fetch_statsbomb_passes.py \
  --output_dir data/raw \
  --output_csv passes_all_matches_fixed.csv \
  --min_season_year 2000 \
  --exclude_team "Bayer Leverkusen" \
  --exclude_competition_contains "Bundesliga" \
  --exclude_season "2023/2024"
```

The default manuscript workflow does not require this exclusion unless a holdout design is explicitly used and reported.

### Step 1. Train the final xPass model

This step trains the final xPass model, evaluates discrimination and calibration, and saves model files and test-set metrics.

```bash
python scripts/02_train_xpass.py \
  --input_csv data/raw/passes_all_matches_fixed.csv \
  --output_dir outputs/xpass_final
```

Main outputs:

```text
outputs/xpass_final/xpass_pipeline_auc.pkl
outputs/xpass_final/xpass_pipeline_calibrated.pkl
outputs/xpass_final/xpass_test_metrics.csv
outputs/xpass_final/xpass_test_predictions.csv
outputs/xpass_final/xpass_val_curve.csv
outputs/xpass_final/passes_filtered.csv
outputs/xpass_final/plots/
```

The file `passes_filtered.csv` contains the open-play pass sample after filtering set-piece and restart contexts.

### Step 2. Generate out-of-fold xPass probabilities

This step produces leakage-controlled xPass probabilities. Matches are held out by fold, and each held-out match receives predictions from a model that was not trained on that match.

```bash
python scripts/03_generate_oof_xpass.py \
  --input_csv outputs/xpass_final/passes_filtered.csv \
  --output_dir outputs/xpass_oof \
  --n_splits 5 \
  --epochs 50
```

Main outputs:

```text
outputs/xpass_oof/passes_with_xpass_oof.csv
outputs/xpass_oof/xpass_oof_predictions.csv
outputs/xpass_oof/xpass_oof_metrics.csv
outputs/xpass_oof/xpass_oof_fold_metrics.csv
```

The downstream risk-reward script prioritizes `xpass_pred_calib_oof` when this column is available.

### Step 3. Compute pass-level risk-reward metrics

This step computes the spatial value proxy, conditional continuation value, passer marginal contribution, failure cost, gain on success, risk, and expected reward.

```bash
python scripts/04_compute_risk_reward.py \
  --input_csv outputs/xpass_oof/passes_with_xpass_oof.csv \
  --output_dir outputs/risk_reward
```

Main outputs:

```text
outputs/risk_reward/passes_with_appended_new_cols_v4.csv
outputs/risk_reward/risk_reward_per_pass_v4.csv
outputs/risk_reward/cv_predictions_v4.csv
outputs/risk_reward/passer_marginal_oof_v4.csv
outputs/risk_reward/frontier_v4.png
outputs/risk_reward/gru_loss_curve_v4.png
```


### Step 4. Run sensitivity analyses

This step summarizes the robustness of the xPass model and the risk-reward framework under alternative feature, optimizer, continuation-value, passer-marginal, and formula-weight settings. It is computationally heavier than the baseline workflow because it re-runs the relevant scripts across multiple scenarios.

A full run can be launched with:

```bash
python scripts/05_sensitivity_analysis.py \
  --xpass_input_csv data/raw/passes_all_matches_fixed.csv \
  --risk_input_csv outputs/xpass_oof/passes_with_xpass_oof.csv \
  --baseline_risk_csv outputs/risk_reward/passes_with_appended_new_cols_v4.csv \
  --output_dir outputs/sensitivity
```

If only formula-level sensitivity is needed after the baseline risk-reward output has already been generated, use:

```bash
python scripts/05_sensitivity_analysis.py \
  --skip_xpass \
  --skip_risk_structural \
  --baseline_risk_csv outputs/risk_reward/passes_with_appended_new_cols_v4.csv \
  --output_dir outputs/sensitivity_formula_only
```

Main outputs:

```text
outputs/sensitivity/xpass_structural/xpass_sensitivity_summary.csv
outputs/sensitivity/risk_structural/risk_structural_sensitivity_summary.csv
outputs/sensitivity/risk_formula/risk_formula_sensitivity_summary.csv
```

The formula-level summary reports the mean and median expected reward, the positive-reward rate, the estimated zero-crossing risk, and the Spearman rank correlation against the baseline expected-reward ranking. These outputs are intended to support the manuscript's robustness checks and supplementary statistical output.

## 6. Expected input columns

The downstream scripts expect a StatsBomb-style event table containing pass events or full event data. The following columns are recommended:

- `match_id`
- `type`
- `location`
- `pass_end_location`
- `pass_outcome`
- `pass_length`
- `pass_angle`
- `minute`
- `second`
- `under_pressure`
- `pass_height`
- `pass_type`
- `pass_body_part`
- `play_pattern`
- `team`
- `player`
- `position`
- `possession`

If `pass_length` and `pass_angle` are absent, they are reconstructed from `location` and `pass_end_location`.

## 7. Main parameter defaults

The default parameters are aligned with the manuscript workflow:

```text
Minimum season start year: 2000
xPass random seed: 42
xPass test size: 0.20
xPass validation size: 0.25
xPass epochs: 50
xPass batch size: 96,000
xPass learning rate: 2.5e-4
xPass penalty: elasticnet
xPass L1 ratio: 0.25
xPass alpha: 3e-6
OOF folds: 5
Continuation-value horizon H: 3
Continuation discount gamma: 0.95
GRU window length: 5
Continuation-value weight lambda: 0.30
Passer marginal weight: 0.05
Failure-cost proxy weight on intermediate turnover point: 0.70
```

## 8. Notes on reproducibility and interpretation

- The data-access script replaces the exploratory notebook workflow. It does not depend on in-memory notebook variables such as `all_matches`, `kept`, or `skipped`.
- `statsbomb_event_download_log.csv` should be retained as part of the reproducibility record. It documents successful and failed match downloads.
- The final xPass model and the out-of-fold xPass predictions serve different purposes. The final model is used for model evaluation and future deployment, while the out-of-fold predictions are recommended for constructing residual-based or downstream pass-value metrics.
- The risk-reward script gives priority to out-of-fold xPass columns when available.
- The spatial value function in this package is an EPV-like location-based proxy rather than a full transition-matrix xT model. If a manuscript describes the value function as xT, the implementation should be replaced with a transition-based xT estimator or the wording should be revised.
- All coordinates should be oriented consistently so that the attacking direction is left to right before spatial values are interpreted.
- The failure-cost component contains a heuristic intermediate turnover-point proxy. Sensitivity analyses are recommended for the coefficients used in this component.

## 9. Code and data availability statement template

Example wording for a manuscript:

```text
The analysis code used for data acquisition, data cleaning, xPass model estimation, out-of-fold probability generation, risk-reward metric construction, and figure generation is available in this repository. The raw event data were derived from publicly available StatsBomb Open Data, subject to the provider's terms of use. Full statistical outputs, including model performance metrics, out-of-fold prediction metrics, pass-level risk-reward outputs, and sensitivity-analysis summaries, are provided in the repository outputs or supplementary materials.
```
