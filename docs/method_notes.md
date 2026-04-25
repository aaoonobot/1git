[method_notes.md](https://github.com/user-attachments/files/27084125/method_notes.md)
# Method Notes

This document explains several methodological choices used in the pass risk-reward framework. It is intended to accompany the reproducibility scripts and the statistical output tables.

## 1. Why out-of-fold xPass is used

The risk-reward framework uses xPass as the probability that a pass will be completed in its event context. This probability enters the analysis in two places. It defines risk as `1 - xpass_prob`, and it is also used to calculate expected reward. For this reason, the probability estimate should not be an in-sample fitted value for the same events used in the downstream analysis.

Out-of-fold xPass addresses this issue by predicting each event from a model that was not trained on the match containing that event. The split is grouped by `match_id`, so events from the same match do not appear simultaneously in the training and held-out prediction sets. This design reduces in-sample optimism and lowers the risk that match-specific patterns leak into the probability estimate.

OOF xPass is especially important for the passer marginal contribution term. That term is based on the residual `success - xpass_prob`. If xPass were calculated in sample, player residuals could partly reflect model overfitting rather than genuine long-run deviation from contextual expectation. Using match-level OOF predictions makes the residual-based adjustment more conservative and methodologically cleaner.

The final calibrated xPass model can still be trained and saved for deployment on new data. However, the analytical risk-reward results should preferably use OOF xPass when the same historical sample is being evaluated.

## 2. Why continuation value is conditional on successful passes

Receiver continuation value is designed to measure what happens after a pass has been completed. Its purpose is to capture the short-horizon attacking development made possible by a successful pass, beyond the immediate change in spatial value from start to end location.

For this reason, the observed continuation-value label is defined only for successful passes. Failed passes terminate or disrupt the possession sequence, so assigning them a continuation value of zero would mix two different concepts: the quality of post-success attacking development and the consequence of failure. In the present framework, those concepts are treated separately.

The reward side is represented by `GainOnSuccess`, which is conditional on completion. The failure side is represented by `CostOnFailure`, which describes the consequence of losing possession. This separation keeps the framework conceptually consistent. Completion probability determines how much weight each side receives in the expected reward equation.

In practical terms, `receiver_CV_cond` is estimated as a conditional continuation value. It can enter `GainOnSuccess` because it answers the question: if the pass is completed, how much short-horizon attacking value is likely to be generated after the receiver obtains the ball?

## 3. Why failure cost uses a turnover proxy point

Event data usually record the start and intended end locations of a pass, but they do not consistently provide the exact point at which an incomplete pass is intercepted, contested, or otherwise lost. Treating every failed pass as if the opponent gains possession at the intended endpoint would be too rigid. Treating every failed pass as if the turnover occurs at the start point would also be unrealistic.

The turnover proxy point is introduced as an intermediate estimate along the pass trajectory. It reflects the idea that failed passes often break down somewhere between the origin and the intended target. Its location is allowed to vary with pass completion probability, pressure status, pass length, and pass angle. Higher xPass values shift the proxy closer to the target because a failed easy pass is more likely to break down late in the passing lane. Longer, more diagonal, and pressured passes are treated more conservatively because the uncertainty and interception risk may occur earlier in the trajectory.

`CostOnFailure` is then calculated mainly from the opponent-oriented spatial value at this turnover proxy point, with smaller stabilizing weights on the opponent values at the start and end locations. The baseline formula is:

`CostOnFailure = 0.70 × EPV_opp_turnover + 0.15 × EPV_opp_start + 0.15 × EPV_opp_end`

This design should be interpreted as a spatial consequence proxy rather than a directly observed turnover location. For that reason, sensitivity analysis is necessary to check whether the main conclusions depend heavily on this specification.

## 4. Parameters examined in sensitivity analysis

The sensitivity analysis is designed to evaluate whether the main conclusions are stable under reasonable changes to model structure and formula weights. It includes three groups of checks.

### 4.1 xPass structural sensitivity

The xPass checks examine whether probability estimation remains stable when selected feature-engineering and optimization choices are modified. The tested scenarios include:

- removing directional flags
- removing region bins
- removing interaction features
- changing the learning-rate strategy
- using a logistic regression estimator with SAGA optimization as an alternative specification

The main outputs include AUC, calibrated log loss, Brier score, accuracy, and match-level performance summaries.

### 4.2 Risk-reward structural sensitivity

The risk-reward structural checks examine parameters that affect continuation value and passer marginal contribution. The tested scenarios include:

- shorter continuation horizon, such as `H_STEPS = 2` and `GAMMA = 0.90`
- longer continuation horizon, such as `H_STEPS = 5` and `GAMMA = 0.98`
- alternative sequence-window lengths, such as `WINDOW = 3` and `WINDOW = 7`
- alternative shrinkage strengths for passer marginal contribution, such as `PASSER_PRIOR_K = 20` and `PASSER_PRIOR_K = 100`

The main outputs include mean expected reward, median expected reward, positive-reward rate, mean gain, mean cost, estimated zero-crossing risk, and rank correlation with the baseline expected-reward values.

### 4.3 Formula-weight sensitivity

The formula-weight checks examine whether the expected reward ranking and threshold results are sensitive to the chosen reward and cost weights. The tested scenarios include:

- lower and higher continuation-value weight, such as `lambda_cv = 0.15` and `lambda_cv = 0.45`
- lower and higher passer marginal weight, such as `passer_weight = 0.02` and `passer_weight = 0.10`
- alternative failure-cost weights, such as `0.60 / 0.20 / 0.20` and `0.80 / 0.10 / 0.10` for turnover, start, and end components

The stability of the framework is assessed by comparing changes in the zero-crossing risk threshold, the positive expected-reward rate, and the Spearman rank correlation between each scenario and the baseline expected-reward values.

## 5. Interpretation of sensitivity outputs

The sensitivity outputs should be read as robustness evidence, not as a search for the single best parameter setting. The key question is whether the main qualitative conclusions remain stable: risk is right-skewed, gain on success tends to increase with risk, failure cost eventually dominates at higher risk, and expected reward crosses from positive to negative beyond a critical range.

If alternative specifications change absolute expected-reward values but preserve relative ranking and threshold patterns, this supports the robustness of the framework. If a specification substantially changes both rank ordering and threshold interpretation, it should be discussed as a limitation and used to refine the model.
