# Reproducibility and verification

## Frozen analytical sample

The committed manifests and audit records define the published analytical sample:

- 3,926 matches;
- 3,806,977 downloaded pass events before open-play filtering;
- 3,396,912 primary open-play passes;
- 1,754,809 strict open-play passes;
- 743,896 failures in the primary xPass sample;
- 110 matches and 54,795 focal-team passes in the three-team application;
- 80 players in the three-team data, of whom 55 meet the primary threshold of at least 200 passes and five matches.

The locked independent xPass split contains:

- training: 2,747 matches;
- tuning: 395 matches;
- calibration: 394 matches;
- test: 390 matches and 338,520 primary open-play passes.

All predictive splits and cross-fitting folds are defined at match level.

## Fixed manifests

- `data/manifests/competitions_selected.csv` fixes the 60 competition-season combinations.
- `data/manifests/xpass_match_split_manifest.csv` fixes all 3,926 match IDs and their model-development roles.
- `data/manifests/match_level_target_audit.csv` records match-level pass counts and outcomes.
- `data/manifests/three_team_match_manifest.csv` fixes the 34/38/38 matches used for Bayer Leverkusen 2023/24, Athletic Club 2015/16, and AS Monaco 2015/16.

Use both the competition and match manifests when downloading data. This prevents newly added open-data competitions or matches from silently changing the analytical sample.

## Model specifications

### xPass

The xPass models use elastic-net logistic regression trained with stochastic gradient descent. Hyperparameters were selected on the tuning matches. The main and no-team locked selections are stored in `config/`.

The independent test split must not be used for additional feature or hyperparameter selection. `06_xpass.py evaluate` therefore requires the explicit `--confirm_test_evaluation` flag.

For downstream analyses, `06_xpass.py crossfit` generates five-fold match-level out-of-fold probabilities. Each outer-fold development set reserves matches for Platt calibration.

### PCTV

The pass-chain threat model uses L2-regularized logistic regression with match-level cross-fitting. The primary horizon is H=3, defined as the current pass and the next two pass events in the same possession. H=1 and H=5 are sensitivity specifications.

Only the observed start-state prediction has a directly observed calibration target. Successful end states and opponent-view end states are model evaluations of alternative states and are not treated as separately labeled calibration samples.

### Pass-value decomposition

For each pass:

```text
success_value = threat_end − threat_start
failure_cost_start = threat_start
failure_cost_opponent = opponent_threat_end
failure_cost = failure_cost_start + failure_cost_opponent
expected_success_value = p_success × success_value
expected_failure_cost = (1 − p_success) × failure_cost
expected_net_value = expected_success_value − expected_failure_cost
realized_value = observed_success × success_value − observed_failure × failure_cost
execution_residual = observed_success − p_success
```

The failure-exposure components remain separate in the output file so that start-state-only and opponent-transition-only sensitivity analyses can be reproduced directly.

## Uncertainty and robustness

Reported confidence intervals use nonparametric match-cluster bootstrap resampling, normally with 1,000 repetitions.

The robustness module compares:

- H=1 and H=5 against the primary H=3 horizon;
- main xPass against no-team xPass;
- primary against strict open-play sampling;
- full failure exposure against start-state-only and opponent-transition-only definitions;
- player eligibility thresholds of 100, 200, and 300 passes.

Rank-correlation values of 0.80 and profile-agreement values of 0.70 are descriptive benchmarks, not significance thresholds. The substantive robustness conclusion is mixed because the opponent-transition-only specification materially changes the overall player decision-value ordering.

## Recorded software environment

Core modeling audit files recorded:

- Python 3.14.5;
- NumPy 2.5.1;
- pandas 3.0.3;
- SciPy 1.18.0;
- scikit-learn 1.9.0;
- joblib 1.5.3;
- Windows 11.

Reporting modules recorded NumPy versions from 2.4.6 to 2.5.1 and Matplotlib versions from 3.10.9 to 3.11.0. The version ranges in `requirements.txt` are intended to recreate a compatible environment; exact module-level versions remain available in the generated audit JSON files.

## Verification checklist

After a complete run, verify at minimum:

1. preprocessing reports 3,926 matches, 3,806,977 raw pass rows, 3,396,912 primary rows, and 1,754,809 strict rows;
2. the target audit reports 2,653,016 successes and 743,896 failures in the primary sample;
3. the split validation reports 2,747/395/394/390 matches;
4. xPass evaluation reports 390 test matches and 338,520 test passes;
5. the OOF xPass audit reports 3,396,912 predictions from 3,926 matches;
6. each H=1/H=3/H=5 PCTV output contains the full primary sample;
7. the three-team application reports 110 matches and 54,795 focal-team passes;
8. the team/player summary reports 3 teams, 80 players, and 55 primary-eligible players;
9. all module audit records report `PASS` where a technical execution status is used;
10. robustness reporting distinguishes technical completion (`PASS`) from its substantive conclusion (`MIXED`).

## Hashes and archived releases

Preserve the following together when freezing a release:

- source scripts;
- fixed manifests and tuning JSON files;
- input hashes and output hashes;
- module audit JSON files;
- SHA-256 manifests;
- terminal logs;
- compact reference outputs.

Large raw and pass-level files should be regenerated locally and should not be committed to GitHub.
