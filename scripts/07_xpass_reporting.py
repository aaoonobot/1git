# -*- coding: utf-8 -*-
"""
01_xpass_reporting.py

Reporting for the locked independent-test xPass evaluation.

This script does NOT refit or modify any model. It reads the output files
created by:

    01_xpass.py evaluate ...

Required files in --evaluation_dir:
    final_test_metrics.csv
    match_cluster_bootstrap_95ci.csv
    final_test_reliability_bins.csv
    final_test_event_predictions.csv
    final_test_evaluation_record.json

Main outputs:
    Table_1_xpass_independent_test_performance.csv
    Table_1_xpass_independent_test_performance_formatted.csv
    Table_S1_xpass_classification_metrics.csv
    Table_S2_xpass_tail_calibration.csv
    Figure_S1_xpass_reliability_comparison.png/.pdf/.svg
    manuscript_results_text.txt
    figure_captions.txt
    reporting_summary.json

Design principles:
- Locked 390-match independent test set is the primary performance evidence.
- OOF metrics are not mixed into this table.
- Match-cluster bootstrap CIs are used when available.
- Tail subsets are defined by the full model and are held fixed when
  comparing geometry-only, no-team, and full models.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# USER-ADJUSTABLE PUBLICATION SETTINGS
# =============================================================================

@dataclass(frozen=True)
class FigureStyle:
    # Figure geometry
    width_in: float = 6.30
    height_in: float = 5.15
    dpi: int = 600

    # Typography
    font_family: str = "Times New Roman"
    axis_label_size: float = 9.0
    tick_label_size: float = 8.0
    legend_size: float = 8.0

    # Lines and markers
    main_line_width: float = 1.8
    comparison_line_width: float = 1.45
    ideal_line_width: float = 1.0
    marker_size: float = 4.8
    error_line_width: float = 0.85
    cap_size: float = 2.2
    spine_width: float = 0.8

    # Color-blind-aware, restrained journal palette
    full_color: str = "#1F4E79"
    no_team_color: str = "#D97706"
    geometry_color: str = "#6B7280"
    ideal_color: str = "#A8A8A8"
    grid_color: str = "#D9D9D9"

    # Reliability settings
    main_reliability_bins: int = 20
    comparison_reliability_bins: int = 20
    reliability_binning: str = "fixed"  # "quantile" or "fixed"

    # Tail definitions based on full-model predicted failure risk
    tail_fractions: tuple[float, ...] = (0.05, 0.10, 0.20)


STYLE = FigureStyle()

EXPECTED_TEST_MATCHES = 390
EXPECTED_PRIMARY_TEST_ROWS = 338_520

PRIMARY_COHORT = "primary_open_play_test"

MODEL_ORDER = [
    "constant_probability",
    "geometry_calibrated",
    "no_team_calibrated",
    "full_calibrated",
]

MODEL_LABELS = {
    "constant_probability": "Constant-probability baseline",
    "geometry_calibrated": "Geometry-only xPass",
    "no_team_calibrated": "No-team xPass",
    "full_calibrated": "Full xPass",
}

LEARNED_MODEL_ORDER = [
    "geometry_calibrated",
    "no_team_calibrated",
    "full_calibrated",
]

PREDICTION_COLUMNS = {
    "geometry_calibrated": "p_success_geometry_calibrated",
    "no_team_calibrated": "p_success_no_team_calibrated",
    "full_calibrated": "p_success_full_calibrated",
}

BOOTSTRAP_METRICS = [
    "roc_auc_success",
    "pr_auc_failure",
    "log_loss",
    "brier_score",
    "calibration_slope",
]


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================


def json_default(value: object) -> object:
    """Convert common NumPy and pandas scalar values for JSON output."""
    if value is pd.NA:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-ready independent-test xPass tables, "
            "reliability figures, tail-calibration summaries, and manuscript text."
        )
    )
    parser.add_argument(
        "--evaluation_dir",
        type=Path,
        required=True,
        help="Directory created by `01_xpass.py evaluate`.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for publication-ready tables and figures.",
    )
    parser.add_argument(
        "--expected_test_matches",
        type=int,
        default=EXPECTED_TEST_MATCHES,
    )
    parser.add_argument(
        "--expected_test_rows",
        type=int,
        default=EXPECTED_PRIMARY_TEST_ROWS,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing reporting outputs.",
    )
    return parser.parse_args()


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        "Table_1_xpass_independent_test_performance.csv",
        "Table_1_xpass_independent_test_performance_formatted.csv",
        "Table_S1_xpass_classification_metrics.csv",
        "Table_S2_xpass_tail_calibration.csv",
        "Figure_S1_xpass_reliability_comparison.png",
        "Figure_S1_xpass_reliability_comparison.pdf",
        "Figure_S1_xpass_reliability_comparison.svg",
        "full_xpass_reliability_data.csv",
        "Figure_S1_xpass_reliability_comparison_data.csv",
        "manuscript_results_text.txt",
        "figure_captions.txt",
        "reporting_summary.json",
    ]
    existing = [output_dir / name for name in expected_outputs if (output_dir / name).exists()]
    if existing and not overwrite:
        listed = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Reporting outputs already exist. Use --overwrite to replace them:\n"
            + listed
        )


def safe_float(value: object) -> float:
    try:
        number = float(value)
    except Exception:
        return float("nan")
    return number


def format_number(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def format_metric_ci(
    estimate: float,
    lower: float | None,
    upper: float | None,
    digits: int = 3,
) -> str:
    if not math.isfinite(float(estimate)):
        return "—"
    estimate_text = f"{estimate:.{digits}f}"
    if (
        lower is None
        or upper is None
        or not math.isfinite(float(lower))
        or not math.isfinite(float(upper))
    ):
        return estimate_text
    return (
        f"{estimate:.{digits}f} "
        f"({float(lower):.{digits}f}–{float(upper):.{digits}f})"
    )


def configure_matplotlib(style: FigureStyle) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [style.font_family, "DejaVu Sans"],
            "font.size": style.tick_label_size,
            "axes.labelsize": style.axis_label_size,
            "axes.titlesize": style.axis_label_size,
            "xtick.labelsize": style.tick_label_size,
            "ytick.labelsize": style.tick_label_size,
            "legend.fontsize": style.legend_size,
            "axes.linewidth": style.spine_width,
            "xtick.major.width": style.spine_width,
            "ytick.major.width": style.spine_width,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_figure_all_formats(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> None:
    figure.savefig(
        output_dir / f"{stem}.png",
        dpi=dpi,
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


# =============================================================================
# RELIABILITY CALCULATION
# =============================================================================

def wilson_interval(
    successes: np.ndarray,
    totals: np.ndarray,
    z: float = 1.959963984540054,
) -> tuple[np.ndarray, np.ndarray]:
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)

    proportion = np.divide(
        successes,
        totals,
        out=np.full_like(successes, np.nan),
        where=totals > 0,
    )
    denominator = 1.0 + z**2 / totals
    center = (proportion + z**2 / (2.0 * totals)) / denominator
    half = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / totals
            + z**2 / (4.0 * totals**2)
        )
        / denominator
    )
    return center - half, center + half


def reliability_from_events(
    y_success: np.ndarray,
    p_success: np.ndarray,
    bins: int,
    method: str,
) -> pd.DataFrame:
    y = np.asarray(y_success, dtype=np.int8)
    p = np.clip(np.asarray(p_success, dtype=float), 1e-7, 1.0 - 1e-7)

    if method == "quantile":
        raw_edges = np.quantile(p, np.linspace(0.0, 1.0, bins + 1))
        edges = np.unique(raw_edges)
        if len(edges) < 3:
            raise RuntimeError("Too few unique prediction values for quantile binning.")
        edges[0] = 0.0
        edges[-1] = 1.0
    elif method == "fixed":
        edges = np.linspace(0.0, 1.0, bins + 1)
    else:
        raise ValueError("Reliability binning must be 'quantile' or 'fixed'.")

    bin_id = np.clip(
        np.digitize(p, edges, right=True) - 1,
        0,
        len(edges) - 2,
    )

    rows: list[dict[str, float | int]] = []
    for index in range(len(edges) - 1):
        mask = bin_id == index
        count = int(mask.sum())
        if count == 0:
            continue

        observed_successes = int(y[mask].sum())
        observed_rate = observed_successes / count
        predicted_mean = float(p[mask].mean())

        lower, upper = wilson_interval(
            np.array([observed_successes]),
            np.array([count]),
        )

        rows.append(
            {
                "bin": index + 1,
                "lower_probability": float(edges[index]),
                "upper_probability": float(edges[index + 1]),
                "count": count,
                "mean_predicted_success": predicted_mean,
                "observed_success_rate": float(observed_rate),
                "observed_ci_lower": float(lower[0]),
                "observed_ci_upper": float(upper[0]),
                "absolute_gap": abs(predicted_mean - observed_rate),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# TABLE GENERATION
# =============================================================================

def load_primary_metrics(path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "cohort",
        "model",
        "rows",
        "success_prevalence",
        "failure_prevalence",
        "roc_auc_success",
        "pr_auc_failure",
        "log_loss",
        "brier_score",
        "accuracy",
        "balanced_accuracy",
        "failure_precision",
        "failure_recall",
        "failure_f1",
        "calibration_intercept",
        "calibration_slope",
        "ece_20_bins",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise RuntimeError(
            "final_test_metrics.csv is missing columns: " + ", ".join(missing)
        )

    primary = metrics.loc[
        metrics["cohort"].eq(PRIMARY_COHORT)
        & metrics["model"].isin(MODEL_ORDER)
    ].copy()

    present = set(primary["model"])
    missing_models = [model for model in MODEL_ORDER if model not in present]
    if missing_models:
        raise RuntimeError(
            "Primary-test metrics are missing models: " + ", ".join(missing_models)
        )

    primary["model_order"] = primary["model"].map(
        {name: index for index, name in enumerate(MODEL_ORDER)}
    )
    primary = primary.sort_values("model_order").drop(columns="model_order")
    return primary


def load_bootstrap(path: Path) -> pd.DataFrame:
    bootstrap = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "cohort",
        "model",
        "metric",
        "estimate",
        "ci_lower_2_5",
        "ci_upper_97_5",
        "bootstrap_reps_requested",
        "bootstrap_reps_valid",
        "bootstrap_unit",
    }
    missing = sorted(required - set(bootstrap.columns))
    if missing:
        raise RuntimeError(
            "match_cluster_bootstrap_95ci.csv is missing columns: "
            + ", ".join(missing)
        )
    return bootstrap


def bootstrap_lookup(
    bootstrap: pd.DataFrame,
    model: str,
    metric: str,
) -> tuple[float | None, float | None]:
    selected = bootstrap.loc[
        bootstrap["cohort"].eq(PRIMARY_COHORT)
        & bootstrap["model"].eq(model)
        & bootstrap["metric"].eq(metric)
    ]
    if selected.empty:
        return None, None
    row = selected.iloc[0]
    return (
        safe_float(row["ci_lower_2_5"]),
        safe_float(row["ci_upper_97_5"]),
    )


def create_performance_tables(
    primary_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    test_matches: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict[str, object]] = []
    formatted_rows: list[dict[str, object]] = []
    classification_rows: list[dict[str, object]] = []

    for model in MODEL_ORDER:
        row = primary_metrics.loc[primary_metrics["model"].eq(model)].iloc[0]
        is_constant = model == "constant_probability"

        ci_values: dict[str, tuple[float | None, float | None]] = {}
        for metric in BOOTSTRAP_METRICS:
            ci_values[metric] = bootstrap_lookup(bootstrap, model, metric)

        raw_rows.append(
            {
                "Model": MODEL_LABELS[model],
                "Code model": model,
                "Test matches": test_matches,
                "Test passes": int(row["rows"]),
                "Success prevalence": float(row["success_prevalence"]),
                "Failure prevalence": float(row["failure_prevalence"]),
                "ROC-AUC": float(row["roc_auc_success"]),
                "ROC-AUC CI lower": ci_values["roc_auc_success"][0],
                "ROC-AUC CI upper": ci_values["roc_auc_success"][1],
                "Failure PR-AUC": float(row["pr_auc_failure"]),
                "Failure PR-AUC CI lower": ci_values["pr_auc_failure"][0],
                "Failure PR-AUC CI upper": ci_values["pr_auc_failure"][1],
                "Log loss": float(row["log_loss"]),
                "Log loss CI lower": ci_values["log_loss"][0],
                "Log loss CI upper": ci_values["log_loss"][1],
                "Brier score": float(row["brier_score"]),
                "Brier CI lower": ci_values["brier_score"][0],
                "Brier CI upper": ci_values["brier_score"][1],
                "Calibration intercept": (
                    np.nan if is_constant else float(row["calibration_intercept"])
                ),
                "Calibration slope": (
                    np.nan if is_constant else float(row["calibration_slope"])
                ),
                "Calibration slope CI lower": (
                    None if is_constant else ci_values["calibration_slope"][0]
                ),
                "Calibration slope CI upper": (
                    None if is_constant else ci_values["calibration_slope"][1]
                ),
                "ECE (20 fixed-width bins)": float(row["ece_20_bins"]),
            }
        )

        formatted_rows.append(
            {
                "Model": MODEL_LABELS[model],
                "Test matches": f"{test_matches:,}",
                "Test passes": f"{int(row['rows']):,}",
                "ROC-AUC (95% CI)": format_metric_ci(
                    float(row["roc_auc_success"]),
                    *ci_values["roc_auc_success"],
                ),
                "Failure PR-AUC (95% CI)": format_metric_ci(
                    float(row["pr_auc_failure"]),
                    *ci_values["pr_auc_failure"],
                ),
                "Log loss (95% CI)": format_metric_ci(
                    float(row["log_loss"]),
                    *ci_values["log_loss"],
                ),
                "Brier score (95% CI)": format_metric_ci(
                    float(row["brier_score"]),
                    *ci_values["brier_score"],
                ),
                "Calibration intercept": (
                    "—"
                    if is_constant
                    else format_number(float(row["calibration_intercept"]))
                ),
                "Calibration slope (95% CI)": (
                    "—"
                    if is_constant
                    else format_metric_ci(
                        float(row["calibration_slope"]),
                        *ci_values["calibration_slope"],
                    )
                ),
                "ECE (20 bins)": format_number(float(row["ece_20_bins"])),
            }
        )

        classification_rows.append(
            {
                "Model": MODEL_LABELS[model],
                "Test matches": test_matches,
                "Test passes": int(row["rows"]),
                "Accuracy": float(row["accuracy"]),
                "Balanced accuracy": float(row["balanced_accuracy"]),
                "Failure precision": float(row["failure_precision"]),
                "Failure recall": float(row["failure_recall"]),
                "Failure F1": float(row["failure_f1"]),
            }
        )

    return (
        pd.DataFrame(raw_rows),
        pd.DataFrame(formatted_rows),
        pd.DataFrame(classification_rows),
    )


def create_tail_calibration_table(
    event_predictions: pd.DataFrame,
    tail_fractions: Iterable[float],
) -> pd.DataFrame:
    required = {"match_id", "xpass_success", *PREDICTION_COLUMNS.values()}
    missing = sorted(required - set(event_predictions.columns))
    if missing:
        raise RuntimeError(
            "final_test_event_predictions.csv is missing columns: "
            + ", ".join(missing)
        )

    y_failure = 1 - event_predictions["xpass_success"].to_numpy(dtype=np.int8)
    full_failure = (
        1.0
        - event_predictions[
            PREDICTION_COLUMNS["full_calibrated"]
        ].to_numpy(dtype=float)
    )

    rows: list[dict[str, object]] = []
    n_total = len(event_predictions)

    for fraction in tail_fractions:
        if not 0 < fraction < 1:
            raise ValueError("Tail fractions must be between 0 and 1.")

        threshold = float(np.quantile(full_failure, 1.0 - fraction))
        mask = full_failure >= threshold
        n_tail = int(mask.sum())
        observed_failure = float(y_failure[mask].mean())

        for model in LEARNED_MODEL_ORDER:
            p_failure = (
                1.0
                - event_predictions[
                    PREDICTION_COLUMNS[model]
                ].to_numpy(dtype=float)
            )
            predicted_failure = float(p_failure[mask].mean())
            signed_gap = predicted_failure - observed_failure

            rows.append(
                {
                    "Tail definition": (
                        f"Highest {int(round(fraction * 100))}% "
                        "predicted failure risk"
                    ),
                    "Tail defined by": "Full xPass",
                    "Full-model failure-risk threshold": threshold,
                    "Model": MODEL_LABELS[model],
                    "Events": n_tail,
                    "Share of test events": n_tail / n_total,
                    "Mean predicted failure probability": predicted_failure,
                    "Observed failure rate": observed_failure,
                    "Signed calibration gap": signed_gap,
                    "Absolute calibration gap": abs(signed_gap),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def clean_axes(ax: plt.Axes, style: FigureStyle) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(style.spine_width)
    ax.spines["bottom"].set_linewidth(style.spine_width)
    ax.tick_params(direction="out")
    ax.grid(
        True,
        axis="both",
        linewidth=0.45,
        color=style.grid_color,
        alpha=0.55,
        zorder=0,
    )



def plot_model_comparison_reliability(
    event_predictions: pd.DataFrame,
    output_dir: Path,
    style: FigureStyle,
) -> None:
    y = event_predictions["xpass_success"].to_numpy(dtype=np.int8)

    plotting = [
        ("geometry_calibrated", style.geometry_color, "s"),
        ("no_team_calibrated", style.no_team_color, "^"),
        ("full_calibrated", style.full_color, "o"),
    ]

    figure, ax = plt.subplots(
        figsize=(style.width_in, style.height_in),
        constrained_layout=True,
    )

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle=(0, (4, 3)),
        color=style.ideal_color,
        linewidth=style.ideal_line_width,
        label="Ideal calibration",
        zorder=1,
    )

    output_rows: list[pd.DataFrame] = []

    for model, color, marker in plotting:
        p = event_predictions[PREDICTION_COLUMNS[model]].to_numpy(dtype=float)
        reliability = reliability_from_events(
            y,
            p,
            bins=style.comparison_reliability_bins,
            method=style.reliability_binning,
        )
        reliability.insert(0, "model", model)
        output_rows.append(reliability)

        ax.plot(
            reliability["mean_predicted_success"],
            reliability["observed_success_rate"],
            marker=marker,
            markersize=style.marker_size - 0.4,
            linewidth=(
                style.main_line_width
                if model == "full_calibrated"
                else style.comparison_line_width
            ),
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=MODEL_LABELS[model],
            zorder=3,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_xlabel("Predicted pass-completion probability", labelpad=6)
    ax.set_ylabel("Observed pass-completion rate", labelpad=6)

    clean_axes(ax, style)
    ax.legend(
        loc="upper left",
        frameon=False,
        handlelength=2.8,
        borderaxespad=0.4,
    )

    save_figure_all_formats(
        figure,
        output_dir,
        "Figure_S1_xpass_reliability_comparison",
        style.dpi,
    )
    plt.close(figure)

    pd.concat(output_rows, ignore_index=True).to_csv(
        output_dir / "Figure_S1_xpass_reliability_comparison_data.csv",
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# MANUSCRIPT TEXT
# =============================================================================

def get_metric_row(metrics: pd.DataFrame, model: str) -> pd.Series:
    selected = metrics.loc[metrics["model"].eq(model)]
    if selected.empty:
        raise RuntimeError(f"Metrics missing model: {model}")
    return selected.iloc[0]


def create_manuscript_text(
    primary_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    test_matches: int,
) -> str:
    full = get_metric_row(primary_metrics, "full_calibrated")
    geometry = get_metric_row(primary_metrics, "geometry_calibrated")
    no_team = get_metric_row(primary_metrics, "no_team_calibrated")
    constant = get_metric_row(primary_metrics, "constant_probability")

    full_auc_ci = bootstrap_lookup(
        bootstrap, "full_calibrated", "roc_auc_success"
    )
    full_pr_ci = bootstrap_lookup(
        bootstrap, "full_calibrated", "pr_auc_failure"
    )
    full_log_ci = bootstrap_lookup(
        bootstrap, "full_calibrated", "log_loss"
    )
    full_brier_ci = bootstrap_lookup(
        bootstrap, "full_calibrated", "brier_score"
    )
    full_slope_ci = bootstrap_lookup(
        bootstrap, "full_calibrated", "calibration_slope"
    )

    return f"""INDEPENDENT-TEST MODEL PERFORMANCE

The locked independent test set comprised {test_matches:,} matches and {int(full['rows']):,} primary open-play passes. The calibrated full xPass model achieved an ROC-AUC of {float(full['roc_auc_success']):.3f} (95% CI {full_auc_ci[0]:.3f}–{full_auc_ci[1]:.3f}) and a failure-class PR-AUC of {float(full['pr_auc_failure']):.3f} (95% CI {full_pr_ci[0]:.3f}–{full_pr_ci[1]:.3f}). Probability accuracy was reflected by a log loss of {float(full['log_loss']):.3f} (95% CI {full_log_ci[0]:.3f}–{full_log_ci[1]:.3f}) and a Brier score of {float(full['brier_score']):.3f} (95% CI {full_brier_ci[0]:.3f}–{full_brier_ci[1]:.3f}). The calibration intercept was {float(full['calibration_intercept']):.3f}, and the calibration slope was {float(full['calibration_slope']):.3f} (95% CI {full_slope_ci[0]:.3f}–{full_slope_ci[1]:.3f}); these estimates should be interpreted together with the reliability curve rather than as evidence of perfect calibration.

The full model improved on the constant-probability and geometry-only baselines. Relative to the geometry-only model, ROC-AUC increased from {float(geometry['roc_auc_success']):.3f} to {float(full['roc_auc_success']):.3f}, while log loss decreased from {float(geometry['log_loss']):.3f} to {float(full['log_loss']):.3f} and Brier score decreased from {float(geometry['brier_score']):.3f} to {float(full['brier_score']):.3f}. The constant-probability baseline had an ROC-AUC of {float(constant['roc_auc_success']):.3f}, a failure-class PR-AUC of {float(constant['pr_auc_failure']):.3f}, a log loss of {float(constant['log_loss']):.3f}, and a Brier score of {float(constant['brier_score']):.3f}.

Removing team and possession-team identity produced only a modest change in performance. The no-team model achieved an ROC-AUC of {float(no_team['roc_auc_success']):.3f}, a failure-class PR-AUC of {float(no_team['pr_auc_failure']):.3f}, a log loss of {float(no_team['log_loss']):.3f}, and a Brier score of {float(no_team['brier_score']):.3f}. Thus, team identifiers contributed incremental information but were not the principal source of the model's predictive performance.

REPORTING CAUTION

The comparisons above are descriptive unless confidence intervals for paired model-performance differences are separately estimated. Avoid the wording "significantly outperformed" based only on separate model-specific confidence intervals.
"""


def create_figure_captions(style: FigureStyle) -> str:
    binning_text = (
        "equal-frequency"
        if style.reliability_binning == "quantile"
        else "fixed-width"
    )
    return f"""Supplementary Figure S1. Reliability of the geometry-only, no-team, and full calibrated xPass models on the locked independent test set. Curves were calculated using {style.comparison_reliability_bins} {binning_text} bins. The dashed diagonal denotes ideal calibration. This figure is intended to compare calibration patterns; numerical discrimination and probability-accuracy metrics are reported separately.

Supplementary Table S2. Tail calibration on test events with the highest failure risk according to the calibrated full xPass model. The same full-model-defined event subsets were used to evaluate all learned models, ensuring that geometry-only, no-team, and full predictions were compared on identical high-risk events.
"""


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    evaluation_dir = args.evaluation_dir.resolve()
    output_dir = args.output_dir.resolve()

    files = {
        "metrics": evaluation_dir / "final_test_metrics.csv",
        "bootstrap": evaluation_dir / "match_cluster_bootstrap_95ci.csv",
        "reliability": evaluation_dir / "final_test_reliability_bins.csv",
        "predictions": evaluation_dir / "final_test_event_predictions.csv",
        "record": evaluation_dir / "final_test_evaluation_record.json",
    }

    for path in files.values():
        require_file(path)

    prepare_output_dir(output_dir, args.overwrite)
    configure_matplotlib(STYLE)

    primary_metrics = load_primary_metrics(files["metrics"])
    bootstrap = load_bootstrap(files["bootstrap"])
    event_predictions = pd.read_csv(
        files["predictions"],
        encoding="utf-8-sig",
        low_memory=False,
    )

    with files["record"].open("r", encoding="utf-8") as file:
        evaluation_record = json.load(file)

    actual_rows = len(event_predictions)
    actual_matches = int(
        event_predictions["match_id"].astype(str).nunique()
    )

    checks = {
        "evaluation_record_status_pass": (
            evaluation_record.get("status") == "PASS"
        ),
        "independent_test_flag_true": bool(
            evaluation_record.get("test_set_evaluated_once", False)
        ),
        "test_rows_match_expected": (
            actual_rows == args.expected_test_rows
        ),
        "test_matches_match_expected": (
            actual_matches == args.expected_test_matches
        ),
        "event_outcome_binary": set(
            event_predictions["xpass_success"].dropna().unique().tolist()
        ).issubset({0, 1}),
        "required_models_present": set(MODEL_ORDER).issubset(
            set(primary_metrics["model"])
        ),
        "required_prediction_columns_present": set(
            PREDICTION_COLUMNS.values()
        ).issubset(set(event_predictions.columns)),
        "bootstrap_unit_match_id": (
            set(
                bootstrap.loc[
                    bootstrap["cohort"].eq(PRIMARY_COHORT),
                    "bootstrap_unit",
                ].dropna().astype(str)
            )
            <= {"match_id"}
        ),
    }

    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(
            "Reporting validation failed: " + ", ".join(failed)
        )

    raw_table, formatted_table, classification_table = (
        create_performance_tables(
            primary_metrics,
            bootstrap,
            actual_matches,
        )
    )

    tail_table = create_tail_calibration_table(
        event_predictions,
        STYLE.tail_fractions,
    )

    raw_table.to_csv(
        output_dir / "Table_1_xpass_independent_test_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    formatted_table.to_csv(
        output_dir
        / "Table_1_xpass_independent_test_performance_formatted.csv",
        index=False,
        encoding="utf-8-sig",
    )
    classification_table.to_csv(
        output_dir / "Table_S1_xpass_classification_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tail_table.to_csv(
        output_dir / "Table_S2_xpass_tail_calibration.csv",
        index=False,
        encoding="utf-8-sig",
    )

    full_reliability = reliability_from_events(
        event_predictions["xpass_success"].to_numpy(dtype=np.int8),
        event_predictions[
            PREDICTION_COLUMNS["full_calibrated"]
        ].to_numpy(dtype=float),
        bins=STYLE.main_reliability_bins,
        method=STYLE.reliability_binning,
    )
    full_reliability.to_csv(
        output_dir / "full_xpass_reliability_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_model_comparison_reliability(
        event_predictions,
        output_dir,
        STYLE,
    )

    manuscript_text = create_manuscript_text(
        primary_metrics,
        bootstrap,
        actual_matches,
    )
    (output_dir / "manuscript_results_text.txt").write_text(
        manuscript_text,
        encoding="utf-8",
    )

    captions = create_figure_captions(STYLE)
    (output_dir / "figure_captions.txt").write_text(
        captions,
        encoding="utf-8",
    )

    summary = {
        "status": "PASS",
        "purpose": (
            "Reporting for the locked independent-test xPass evaluation."
        ),
        "evaluation_dir": str(evaluation_dir),
        "output_dir": str(output_dir),
        "primary_cohort": PRIMARY_COHORT,
        "test_matches": actual_matches,
        "test_passes": actual_rows,
        "figure_style": {
            key: value
            for key, value in STYLE.__dict__.items()
        },
        "checks": checks,
        "full_model_reliability_bins": len(full_reliability),
        "tail_fractions": list(STYLE.tail_fractions),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "outputs": sorted(
            path.name for path in output_dir.iterdir() if path.is_file()
        ),
    }

    (output_dir / "reporting_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    print("=" * 100)
    print("Independent-test xPass reporting completed")
    print("=" * 100)
    print(f"Test matches: {actual_matches:,}")
    print(f"Test passes: {actual_rows:,}")
    print("Status: PASS")
    print(f"Outputs: {output_dir}")
    print(
        "Main table: "
        f"{output_dir / 'Table_1_xpass_independent_test_performance_formatted.csv'}"
    )
    print(
        "Reliability figure: "
        f"{output_dir / 'Figure_S1_xpass_reliability_comparison.png'}"
    )


if __name__ == "__main__":
    main()
