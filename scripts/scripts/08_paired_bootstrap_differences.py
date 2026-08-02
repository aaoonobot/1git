#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
08_paired_bootstrap_differences.py

Paired match-cluster bootstrap comparisons for the locked independent-test
xPass models.

The script does NOT refit or modify any model. It reads the already locked
event-level test predictions and repeatedly resamples complete matches.
All models are evaluated on the same resampled matches in every repetition.

Positive values always favor Full xPass:
- ROC-AUC improvement = Full - comparator
- Failure PR-AUC improvement = Full - comparator
- Log-loss reduction = Comparator - Full
- Brier-score reduction = Comparator - Full

Required inputs
---------------
1. final_test_event_predictions.csv
2. final_test_evaluation_record.json

Outputs
-------
paired_model_difference_bootstrap.csv
paired_model_difference_bootstrap_replicates.csv
paired_model_difference_summary.json
Figure_S_paired_ROC_AUC_improvement.png/.pdf/.svg
Figure_S_paired_failure_PR_AUC_improvement.png/.pdf/.svg
Figure_S_paired_log_loss_reduction.png/.pdf/.svg
Figure_S_paired_Brier_score_reduction.png/.pdf/.svg
paired_bootstrap_figure_captions.txt
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from pathlib import Path
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


EXPECTED_TEST_MATCHES = 390
EXPECTED_TEST_ROWS = 338_520

COMPARATORS = [
    "constant_probability",
    "geometry_calibrated",
    "no_team_calibrated",
]

MODEL_LABELS = {
    "constant_probability": "Constant-probability baseline",
    "geometry_calibrated": "Geometry-only xPass",
    "no_team_calibrated": "No-team xPass",
    "full_calibrated": "Full xPass",
}

PREDICTION_COLUMNS = {
    "geometry_calibrated": "p_success_geometry_calibrated",
    "no_team_calibrated": "p_success_no_team_calibrated",
    "full_calibrated": "p_success_full_calibrated",
}

METRIC_SPECS = {
    "roc_auc_improvement": {
        "label": "ROC-AUC improvement",
        "direction": "Full minus comparator",
        "file_stub": "ROC_AUC_improvement",
    },
    "failure_pr_auc_improvement": {
        "label": "Failure PR-AUC improvement",
        "direction": "Full minus comparator",
        "file_stub": "failure_PR_AUC_improvement",
    },
    "log_loss_reduction": {
        "label": "Log-loss reduction",
        "direction": "Comparator minus Full",
        "file_stub": "log_loss_reduction",
    },
    "brier_reduction": {
        "label": "Brier-score reduction",
        "direction": "Comparator minus Full",
        "file_stub": "Brier_score_reduction",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired match-cluster bootstrap confidence intervals for "
            "locked independent-test xPass model differences."
        )
    )
    parser.add_argument(
        "--predictions_csv",
        type=Path,
        required=True,
        help="final_test_event_predictions.csv",
    )
    parser.add_argument(
        "--evaluation_record_json",
        type=Path,
        required=True,
        help="final_test_evaluation_record.json",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--expected_test_matches",
        type=int,
        default=EXPECTED_TEST_MATCHES,
    )
    parser.add_argument(
        "--expected_test_rows",
        type=int,
        default=EXPECTED_TEST_ROWS,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except Exception:
            pass
    return text


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}\n"
            "Use --overwrite only when intentionally replacing these outputs."
        )
    if overwrite:
        for path in output_dir.iterdir():
            if path.is_file():
                path.unlink()


def metric_values(
    y_success: np.ndarray,
    p_success: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    y_success = np.asarray(y_success, dtype=np.int8)
    p_success = np.clip(
        np.asarray(p_success, dtype=np.float64),
        1e-7,
        1.0 - 1e-7,
    )
    y_failure = 1 - y_success
    p_failure = 1.0 - p_success

    return {
        "roc_auc_success": float(
            roc_auc_score(
                y_success,
                p_success,
                sample_weight=sample_weight,
            )
        ),
        "failure_pr_auc": float(
            average_precision_score(
                y_failure,
                p_failure,
                sample_weight=sample_weight,
            )
        ),
        "log_loss": float(
            log_loss(
                y_success,
                p_success,
                labels=[0, 1],
                sample_weight=sample_weight,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_success,
                p_success,
                sample_weight=sample_weight,
            )
        ),
    }


def difference_values(
    full_metrics: dict[str, float],
    comparator_metrics: dict[str, float],
) -> dict[str, float]:
    return {
        "roc_auc_improvement": (
            full_metrics["roc_auc_success"]
            - comparator_metrics["roc_auc_success"]
        ),
        "failure_pr_auc_improvement": (
            full_metrics["failure_pr_auc"]
            - comparator_metrics["failure_pr_auc"]
        ),
        "log_loss_reduction": (
            comparator_metrics["log_loss"]
            - full_metrics["log_loss"]
        ),
        "brier_reduction": (
            comparator_metrics["brier_score"]
            - full_metrics["brier_score"]
        ),
    }


def match_bootstrap_weights(
    inverse_match_index: np.ndarray,
    n_matches: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled = rng.integers(0, n_matches, size=n_matches)
    match_counts = np.bincount(sampled, minlength=n_matches)
    return match_counts[inverse_match_index].astype(np.float64)


def constant_metrics(
    y_success: np.ndarray,
    constant_probability: float,
    sample_weight: np.ndarray | None,
) -> dict[str, float]:
    p = np.full(len(y_success), constant_probability, dtype=np.float64)
    return metric_values(y_success, p, sample_weight)


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    output_dir: Path,
) -> None:
    selected = summary.loc[summary["metric"].eq(metric)].copy()
    selected = selected.sort_values("estimate", ascending=True)

    labels = selected["comparator_label"].tolist()
    estimates = selected["estimate"].to_numpy(dtype=float)
    lower = selected["ci_lower_2_5"].to_numpy(dtype=float)
    upper = selected["ci_upper_97_5"].to_numpy(dtype=float)

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    height = max(2.5, 0.58 * len(labels) + 1.25)
    figure, ax = plt.subplots(figsize=(6.3, height), constrained_layout=True)

    y_positions = np.arange(len(labels))
    xerr = np.vstack([estimates - lower, upper - estimates])

    ax.errorbar(
        estimates,
        y_positions,
        xerr=xerr,
        fmt="o",
        color="#1F4E79",
        ecolor="#1F4E79",
        markersize=5.2,
        markeredgecolor="white",
        markeredgewidth=0.7,
        elinewidth=1.1,
        capsize=2.5,
        zorder=3,
    )
    ax.axvline(
        0.0,
        color="#A8A8A8",
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        zorder=1,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(METRIC_SPECS[metric]["label"])
    ax.grid(
        True,
        axis="x",
        color="#D9D9D9",
        linewidth=0.45,
        alpha=0.65,
        zorder=0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    stem = f"Figure_S_paired_{METRIC_SPECS[metric]['file_stub']}"
    figure.savefig(
        output_dir / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_dir / f"{stem}.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()

    predictions_csv = args.predictions_csv.resolve()
    evaluation_record_json = args.evaluation_record_json.resolve()
    output_dir = args.output_dir.resolve()

    for path in [predictions_csv, evaluation_record_json]:
        if not path.exists():
            raise FileNotFoundError(path)

    if args.bootstrap_reps < 100:
        raise ValueError("Use at least 100 bootstrap repetitions.")

    prepare_output_dir(output_dir, args.overwrite)

    required_columns = {
        "match_id",
        "xpass_success",
        *PREDICTION_COLUMNS.values(),
    }
    predictions = pd.read_csv(
        predictions_csv,
        encoding="utf-8-sig",
        usecols=list(required_columns),
        low_memory=False,
    )
    missing = sorted(required_columns - set(predictions.columns))
    if missing:
        raise RuntimeError(
            "Prediction file is missing columns: " + ", ".join(missing)
        )

    with evaluation_record_json.open("r", encoding="utf-8") as file:
        evaluation_record = json.load(file)

    match_ids = predictions["match_id"].map(normalize_id).to_numpy()
    unique_matches, inverse_match_index = np.unique(
        match_ids,
        return_inverse=True,
    )
    y_success = predictions["xpass_success"].to_numpy(dtype=np.int8)

    probabilities = {
        model: predictions[column].to_numpy(dtype=np.float64)
        for model, column in PREDICTION_COLUMNS.items()
    }

    constant_probability = float(
        evaluation_record["primary_training_plus_tuning"][
            "success_prevalence"
        ]
    )
    probabilities["constant_probability"] = np.full(
        len(predictions),
        constant_probability,
        dtype=np.float64,
    )

    checks = {
        "evaluation_record_status_pass": (
            evaluation_record.get("status") == "PASS"
        ),
        "test_set_evaluated_once": bool(
            evaluation_record.get("test_set_evaluated_once", False)
        ),
        "rows_match_expected": len(predictions) == args.expected_test_rows,
        "matches_match_expected": (
            len(unique_matches) == args.expected_test_matches
        ),
        "binary_outcome": set(np.unique(y_success)).issubset({0, 1}),
        "probabilities_valid": all(
            np.isfinite(values).all()
            and ((values > 0.0) & (values < 1.0)).all()
            for values in probabilities.values()
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            "Input validation failed: " + ", ".join(failed)
        )

    # Point estimates on the full locked test sample.
    point_metrics = {
        model: metric_values(y_success, values)
        for model, values in probabilities.items()
    }
    point_differences = {
        comparator: difference_values(
            point_metrics["full_calibrated"],
            point_metrics[comparator],
        )
        for comparator in COMPARATORS
    }

    rng = np.random.default_rng(args.seed)
    replicate_rows: list[dict[str, object]] = []

    for repetition in range(1, args.bootstrap_reps + 1):
        weights = match_bootstrap_weights(
            inverse_match_index,
            len(unique_matches),
            rng,
        )

        # A valid bootstrap sample should contain both outcomes.
        positive_weight = float(weights[y_success == 1].sum())
        negative_weight = float(weights[y_success == 0].sum())
        if positive_weight == 0.0 or negative_weight == 0.0:
            continue

        sampled_metrics = {
            model: metric_values(y_success, values, weights)
            for model, values in probabilities.items()
        }

        for comparator in COMPARATORS:
            differences = difference_values(
                sampled_metrics["full_calibrated"],
                sampled_metrics[comparator],
            )
            for metric, value in differences.items():
                replicate_rows.append(
                    {
                        "bootstrap_rep": repetition,
                        "comparator": comparator,
                        "comparator_label": MODEL_LABELS[comparator],
                        "metric": metric,
                        "metric_label": METRIC_SPECS[metric]["label"],
                        "direction_definition": METRIC_SPECS[metric][
                            "direction"
                        ],
                        "difference": value,
                    }
                )

        if repetition % 100 == 0 or repetition == args.bootstrap_reps:
            print(
                f"[paired bootstrap] {repetition:,}/"
                f"{args.bootstrap_reps:,}"
            )

    replicates = pd.DataFrame(replicate_rows)
    if replicates.empty:
        raise RuntimeError("No valid bootstrap replicates were produced.")

    summary_rows: list[dict[str, object]] = []
    for comparator in COMPARATORS:
        for metric in METRIC_SPECS:
            values = replicates.loc[
                replicates["comparator"].eq(comparator)
                & replicates["metric"].eq(metric),
                "difference",
            ].to_numpy(dtype=np.float64)

            lower, upper = np.quantile(values, [0.025, 0.975])
            estimate = point_differences[comparator][metric]

            summary_rows.append(
                {
                    "comparison": (
                        f"Full xPass vs {MODEL_LABELS[comparator]}"
                    ),
                    "comparator": comparator,
                    "comparator_label": MODEL_LABELS[comparator],
                    "metric": metric,
                    "metric_label": METRIC_SPECS[metric]["label"],
                    "direction_definition": METRIC_SPECS[metric][
                        "direction"
                    ],
                    "estimate": estimate,
                    "ci_lower_2_5": float(lower),
                    "ci_upper_97_5": float(upper),
                    "bootstrap_reps_requested": args.bootstrap_reps,
                    "bootstrap_reps_valid": len(values),
                    "bootstrap_unit": "match_id",
                    "positive_favors_full": True,
                    "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
                    "proportion_bootstrap_values_above_zero": float(
                        np.mean(values > 0.0)
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        output_dir / "paired_model_difference_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    replicates.to_csv(
        output_dir / "paired_model_difference_bootstrap_replicates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for metric in METRIC_SPECS:
        plot_metric(summary, metric, output_dir)

    captions = """Paired model-performance difference figures. Points show the observed difference between Full xPass and each comparator on the locked independent test set; horizontal error bars show percentile 95% confidence intervals from paired match-cluster bootstrap resampling. Positive values favor Full xPass. ROC-AUC and failure PR-AUC are expressed as Full minus comparator. Log-loss and Brier-score reductions are expressed as comparator minus Full. The same resampled matches and event weights were used for both models in every bootstrap repetition.
"""
    (output_dir / "paired_bootstrap_figure_captions.txt").write_text(
        captions,
        encoding="utf-8",
    )

    summary_json = {
        "status": "PASS",
        "purpose": (
            "Paired match-cluster bootstrap confidence intervals for "
            "locked independent-test xPass model differences."
        ),
        "test_matches": len(unique_matches),
        "test_rows": len(predictions),
        "bootstrap_reps_requested": args.bootstrap_reps,
        "bootstrap_reps_valid_minimum": int(
            summary["bootstrap_reps_valid"].min()
        ),
        "bootstrap_seed": args.seed,
        "constant_probability_source": (
            "Training-plus-tuning pass-completion prevalence from the "
            "locked evaluation record."
        ),
        "checks": checks,
        "input_hashes": {
            "predictions_csv_sha256": sha256_file(predictions_csv),
            "evaluation_record_sha256": sha256_file(
                evaluation_record_json
            ),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "outputs": sorted(
            path.name
            for path in output_dir.iterdir()
            if path.is_file()
        ),
    }
    (
        output_dir / "paired_model_difference_summary.json"
    ).write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("Paired match-cluster bootstrap analysis completed")
    print("=" * 100)
    print(f"Test matches: {len(unique_matches):,}")
    print(f"Test passes: {len(predictions):,}")
    print(f"Bootstrap repetitions: {args.bootstrap_reps:,}")
    print("Status: PASS")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
