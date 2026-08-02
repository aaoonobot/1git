# Stage 3: Downstream analysis and reporting

This package adds the downstream modules required to reproduce the full-sample,
three-team, player-level, and robustness applications.

## Replacement files

These files replace files already uploaded in Stage 2:

- `scripts/07_xpass_reporting.py`
  - uses 20 fixed-width reliability bins;
  - generates Supplementary Figure S1 without conflicting with the main-text Figure 1;
  - retains high-risk tail-calibration summaries.
- `scripts/11_pctv_calibration_diagnostics.py`
  - reports pooled OOF calibration intercept, slope, ECE, and reliability data for H=1, H=3, and H=5;
  - writes PNG, PDF, SVG, and an audit JSON.

## New files

- `scripts/publication_style.py`: shared figure style and export settings.
- `scripts/12_plot_grouped_permutation_importance.py`: Supplementary Figure S2.
- `scripts/13_team_player_analysis.py`: team and player summaries for the 110-match cohort.
- `scripts/14_finalize_team_player_tables.py`: centered values, eligibility checks, and corrected ranks.
- `scripts/15_robustness_analysis.py`: H=1/H=5, no-team, strict-open-play, failure-exposure, and player-threshold sensitivity analyses.
- `scripts/16_robustness_reporting.py`: Supplementary Figure S4 and interpretation tables.
- `scripts/17_full_sample_analysis.py`: full-sample Figure 2 and Supplementary Figure S3.
- `scripts/18_three_team_analysis.py`: three-team Figure 3 and fixed event-cohort export.
- `scripts/19_prepare_player_events.py`: player and position metadata for the three-team cohort.
- `scripts/20_player_decision_value_analysis.py`: player-level Figure 4 and matched-player profiles.

## Frozen manifest and reference outputs

- `data/manifests/three_team_match_manifest.csv`: 110 locked matches
  (34 Bayer Leverkusen, 38 Athletic Club, 38 AS Monaco).
- `reference_outputs/three_team/`: compact reference tables used to verify
  the cleaned scripts.
- `STAGE3_VALIDATION.json`: syntax, language, manifest, and table checks.

## Important

Upload the contents of this package to the repository root. Do not upload the
ZIP itself and do not create an additional `pipeline_stage3/` directory.

The root README, requirements, obsolete-script removal, and final repository
cleanup are handled in the next stage.
