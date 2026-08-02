#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
09_grouped_feature_importance.py

Grouped permutation importance for the locked Full xPass model on the
independent test set.

This script does NOT refit the model and does NOT use the results to modify
the model. It loads the locked model bundle, reconstructs the independent-test
feature matrix using the same functions as 06_xpass.py, verifies the resulting
probabilities against final_test_event_predictions.csv, and then permutes
pre-specified raw feature groups.

Default permutation strategy
----------------------------
All columns in a feature group are permuted together using the same row
permutation within each match. This preserves each match's marginal feature
distribution while breaking the event-level association between the feature
group and the pass outcome. The resulting values are descriptive grouped
permutation importance measures, not causal effects.

Positive values indicate that model performance worsened after permutation:
- Increase in log loss
- Increase in Brier score
- Decrease in ROC-AUC
- Decrease in failure PR-AUC

Required inputs
---------------
1. 06_xpass.py
2. xpass_final_model_bundle.joblib
3. passes_preprocessed_with_open_play_flags.csv
4. xpass_match_split_manifest.csv
5. final_test_event_predictions.csv

Outputs
-------
grouped_feature_definitions.csv
grouped_permutation_importance_replicates.csv
grouped_permutation_importance_summary.csv
grouped_permutation_importance_summary.json
Figure_S_grouped_feature_importance_log_loss.png/.pdf/.svg
grouped_feature_importance_caption.txt
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path
from types import ModuleType

import joblib
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

# Pre-specified groups based on raw variables used by the locked Full xPass.
# All variables in a group are moved together using the same permutation.
FEATURE_GROUPS: dict[str, dict[str, object]] = {
    "geometry": {
        "label": "Pass geometry",
        "columns": [
            "start_x_model",
            "start_y_model",
            "end_x_model",
            "end_y_model",
            "pass_length",
            "pass_angle",
        ],
        "description": (
            "Pass origin, intended endpoint, displacement, length, angle, "
            "and derived goal-distance features."
        ),
    },
    "time": {
        "label": "Match time",
        "columns": ["minute", "second"],
        "description": "Minute and second of the event.",
    },
    "pressure": {
        "label": "Pressure status",
        "columns": ["under_pressure"],
        "description": "StatsBomb event-level binary under-pressure indicator.",
    },
    "pass_action_flags": {
        "label": "Pass action flags",
        "columns": [
            "pass_cross",
            "pass_switch",
            "pass_through_ball",
            "pass_backheel",
            "pass_outswinging",
            "pass_inswinging",
            "pass_straight",
            "pass_cut_back",
        ],
        "description": (
            "Binary indicators describing crossing, switching, through balls, "
            "backheels, delivery shape, and cut-backs."
        ),
    },
    "pass_characteristics": {
        "label": "Pass characteristics",
        "columns": [
            "pass_height",
            "pass_body_part",
            "pass_type",
            "pass_technique",
        ],
        "description": (
            "Categorical pass height, body part, event type, and technique."
        ),
    },
    "tactical_context": {
        "label": "Tactical context",
        "columns": ["play_pattern", "position"],
        "description": "Play pattern and passer position.",
    },
    "team_identity": {
        "label": "Team identity",
        "columns": ["team", "possession_team"],
        "description": "Passing-team and possession-team identifiers.",
    },
}

METRICS = {
    "increase_log_loss": "Increase in log loss",
    "increase_brier_score": "Increase in Brier score",
    "decrease_roc_auc": "Decrease in ROC-AUC",
    "decrease_failure_pr_auc": "Decrease in failure PR-AUC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute grouped permutation importance for the locked Full xPass "
            "model on the independent test set."
        )
    )
    parser.add_argument(
        "--xpass_script",
        type=Path,
        required=True,
        help="Path to the exact 06_xpass.py used for model evaluation.",
    )
    parser.add_argument(
        "--model_bundle",
        type=Path,
        required=True,
        help="xpass_final_model_bundle.joblib",
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="passes_preprocessed_with_open_play_flags.csv",
    )
    parser.add_argument(
        "--split_manifest",
        type=Path,
        required=True,
        help="xpass_match_split_manifest.csv",
    )
    parser.add_argument(
        "--reference_predictions_csv",
        type=Path,
        required=True,
        help="final_test_event_predictions.csv",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--read_chunksize", type=int, default=75_000)
    parser.add_argument("--prediction_batch_size", type=int, default=75_000)
    parser.add_argument(
        "--permutation_scope",
        choices=["within_match", "global"],
        default="within_match",
    )
    parser.add_argument(
        "--sample_matches",
        type=int,
        default=0,
        help=(
            "Optional smoke-test mode. Use 0 for the full 390-match test set. "
            "A positive value randomly selects that many complete matches."
        ),
    )
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
    parser.add_argument(
        "--prediction_tolerance",
        type=float,
        default=1e-6,
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


def load_xpass_module(script_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "locked_xpass_module",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import xPass script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required_functions = [
        "load_split_map",
        "read_filtered_chunks",
        "build_matrix",
        "safe_predict",
        "apply_platt",
        "target_from_frame",
    ]
    missing = [
        name for name in required_functions
        if not hasattr(module, name)
    ]
    if missing:
        raise RuntimeError(
            "The supplied xPass script is missing required functions: "
            + ", ".join(missing)
        )
    return module


def load_test_frame(
    xpass_module: ModuleType,
    input_csv: Path,
    split_manifest: Path,
    chunksize: int,
) -> pd.DataFrame:
    split_map = xpass_module.load_split_map(split_manifest)

    parts = list(
        xpass_module.read_filtered_chunks(
            input_csv,
            split_map,
            {"test"},
            "primary_open_play",
            chunksize,
        )
    )
    if not parts:
        raise RuntimeError("No primary-open-play test rows were loaded.")

    frame = pd.concat(parts, ignore_index=True)
    frame["_event_id_norm"] = frame["id"].map(normalize_id)
    frame["_match_id_norm"] = frame["match_id"].map(normalize_id)

    if frame["_event_id_norm"].duplicated().any():
        duplicated = frame.loc[
            frame["_event_id_norm"].duplicated(),
            "_event_id_norm",
        ].head(10).tolist()
        raise RuntimeError(
            "Duplicate event IDs in reconstructed test data: "
            + ", ".join(duplicated)
        )

    return frame


def align_to_reference(
    frame: pd.DataFrame,
    reference_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "id",
        "match_id",
        "xpass_success",
        "p_success_full_calibrated",
    }
    reference = pd.read_csv(
        reference_path,
        encoding="utf-8-sig",
        usecols=list(required),
        low_memory=False,
    )
    missing = sorted(required - set(reference.columns))
    if missing:
        raise RuntimeError(
            "Reference prediction file is missing columns: "
            + ", ".join(missing)
        )

    reference["_event_id_norm"] = reference["id"].map(normalize_id)
    reference["_match_id_norm"] = reference["match_id"].map(normalize_id)

    if reference["_event_id_norm"].duplicated().any():
        raise RuntimeError("Reference prediction event IDs are not unique.")

    frame_ids = set(frame["_event_id_norm"])
    reference_ids = set(reference["_event_id_norm"])
    if frame_ids != reference_ids:
        raise RuntimeError(
            "Reconstructed test events and reference prediction events differ. "
            f"Only in raw frame: {len(frame_ids - reference_ids):,}; "
            f"only in reference: {len(reference_ids - frame_ids):,}."
        )

    frame = (
        frame.set_index("_event_id_norm", drop=False)
        .loc[reference["_event_id_norm"]]
        .reset_index(drop=True)
    )

    if not np.array_equal(
        frame["_match_id_norm"].to_numpy(),
        reference["_match_id_norm"].to_numpy(),
    ):
        raise RuntimeError(
            "Match IDs do not align after event-level ordering."
        )

    return frame, reference


def select_match_subset(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    sample_matches: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sample_matches <= 0:
        return frame, reference

    unique_matches = frame["_match_id_norm"].unique()
    if sample_matches >= len(unique_matches):
        return frame, reference

    rng = np.random.default_rng(seed)
    chosen = set(
        rng.choice(
            unique_matches,
            size=sample_matches,
            replace=False,
        ).tolist()
    )
    mask = frame["_match_id_norm"].isin(chosen).to_numpy()

    return (
        frame.loc[mask].reset_index(drop=True),
        reference.loc[mask].reset_index(drop=True),
    )


def predict_full_model(
    frame: pd.DataFrame,
    xpass_module: ModuleType,
    model: object,
    calibrator: object,
    encoder: object,
    categorical_features: list[str],
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []

    for start in range(0, len(frame), batch_size):
        end = min(start + batch_size, len(frame))
        batch = frame.iloc[start:end]

        matrix = xpass_module.build_matrix(
            batch,
            "full",
            encoder,
            categorical_features,
        )
        raw = xpass_module.safe_predict(model, matrix)
        calibrated = xpass_module.apply_platt(calibrator, raw)
        predictions.append(calibrated)

        del matrix, raw, calibrated
        gc.collect()

    return np.concatenate(predictions)


def metric_values(
    y_success: np.ndarray,
    p_success: np.ndarray,
) -> dict[str, float]:
    y_success = np.asarray(y_success, dtype=np.int8)
    p_success = np.clip(
        np.asarray(p_success, dtype=np.float64),
        1e-7,
        1.0 - 1e-7,
    )

    return {
        "roc_auc_success": float(
            roc_auc_score(y_success, p_success)
        ),
        "failure_pr_auc": float(
            average_precision_score(
                1 - y_success,
                1.0 - p_success,
            )
        ),
        "log_loss": float(
            log_loss(y_success, p_success, labels=[0, 1])
        ),
        "brier_score": float(
            brier_score_loss(y_success, p_success)
        ),
    }


def importance_values(
    baseline: dict[str, float],
    permuted: dict[str, float],
) -> dict[str, float]:
    return {
        "increase_log_loss": (
            permuted["log_loss"] - baseline["log_loss"]
        ),
        "increase_brier_score": (
            permuted["brier_score"] - baseline["brier_score"]
        ),
        "decrease_roc_auc": (
            baseline["roc_auc_success"]
            - permuted["roc_auc_success"]
        ),
        "decrease_failure_pr_auc": (
            baseline["failure_pr_auc"]
            - permuted["failure_pr_auc"]
        ),
    }


def permutation_index(
    match_ids: np.ndarray,
    rng: np.random.Generator,
    scope: str,
) -> np.ndarray:
    n = len(match_ids)
    if scope == "global":
        return rng.permutation(n)

    permutation = np.arange(n)
    order = np.argsort(match_ids, kind="stable")
    sorted_matches = match_ids[order]

    boundaries = np.flatnonzero(
        np.r_[True, sorted_matches[1:] != sorted_matches[:-1], True]
    )

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        positions = order[start:end]
        if len(positions) > 1:
            permutation[positions] = rng.permutation(positions)

    return permutation


def plot_log_loss_importance(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    selected = summary.loc[
        summary["metric"].eq("increase_log_loss")
    ].copy()
    selected = selected.sort_values("mean_importance", ascending=True)

    labels = selected["feature_group_label"].tolist()
    mean_values = selected["mean_importance"].to_numpy(dtype=float)
    lower = selected["percentile_2_5"].to_numpy(dtype=float)
    upper = selected["percentile_97_5"].to_numpy(dtype=float)

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

    height = max(3.6, 0.48 * len(labels) + 1.35)
    figure, ax = plt.subplots(figsize=(6.3, height), constrained_layout=True)

    y_positions = np.arange(len(labels))
    xerr = np.vstack([mean_values - lower, upper - mean_values])

    ax.errorbar(
        mean_values,
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
    ax.set_xlabel(
        "Increase in independent-test log loss after grouped permutation"
    )
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

    stem = "Figure_S_grouped_feature_importance_log_loss"
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

    paths = [
        args.xpass_script,
        args.model_bundle,
        args.input_csv,
        args.split_manifest,
        args.reference_predictions_csv,
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    if args.repetitions < 5:
        raise ValueError("Use at least 5 permutation repetitions.")
    if args.sample_matches == 0 and args.repetitions < 20:
        raise ValueError(
            "Use at least 20 repetitions for the full test analysis."
        )

    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, args.overwrite)

    xpass_module = load_xpass_module(args.xpass_script.resolve())
    bundle = joblib.load(args.model_bundle.resolve())

    required_bundle_keys = {
        "primary_models",
        "primary_calibrators",
        "main_encoder",
        "main_categories",
    }
    missing_bundle = sorted(required_bundle_keys - set(bundle))
    if missing_bundle:
        raise RuntimeError(
            "Model bundle is missing keys: " + ", ".join(missing_bundle)
        )

    full_model = bundle["primary_models"]["full"]
    full_calibrator = bundle["primary_calibrators"]["full"]
    main_encoder = bundle["main_encoder"]
    main_categories = list(bundle["main_categories"])

    test_frame = load_test_frame(
        xpass_module,
        args.input_csv.resolve(),
        args.split_manifest.resolve(),
        args.read_chunksize,
    )
    test_frame, reference = align_to_reference(
        test_frame,
        args.reference_predictions_csv.resolve(),
    )
    test_frame, reference = select_match_subset(
        test_frame,
        reference,
        args.sample_matches,
        args.seed,
    )

    actual_rows = len(test_frame)
    actual_matches = test_frame["_match_id_norm"].nunique()

    if args.sample_matches == 0:
        if actual_rows != args.expected_test_rows:
            raise RuntimeError(
                f"Expected {args.expected_test_rows:,} rows, "
                f"found {actual_rows:,}."
            )
        if actual_matches != args.expected_test_matches:
            raise RuntimeError(
                f"Expected {args.expected_test_matches:,} matches, "
                f"found {actual_matches:,}."
            )

    y_success_raw = xpass_module.target_from_frame(test_frame)
    y_success_reference = reference["xpass_success"].to_numpy(dtype=np.int8)

    if not np.array_equal(y_success_raw, y_success_reference):
        raise RuntimeError(
            "Reconstructed target does not match the locked prediction file."
        )

    baseline_probability = predict_full_model(
        test_frame,
        xpass_module,
        full_model,
        full_calibrator,
        main_encoder,
        main_categories,
        args.prediction_batch_size,
    )

    reference_probability = reference[
        "p_success_full_calibrated"
    ].to_numpy(dtype=np.float64)
    max_abs_difference = float(
        np.max(np.abs(baseline_probability - reference_probability))
    )
    if max_abs_difference > args.prediction_tolerance:
        raise RuntimeError(
            "Reconstructed Full xPass probabilities differ from the locked "
            f"reference predictions. Maximum absolute difference = "
            f"{max_abs_difference:.3e}; tolerance = "
            f"{args.prediction_tolerance:.3e}."
        )

    baseline_metrics = metric_values(
        y_success_reference,
        baseline_probability,
    )

    # Validate feature definitions against the actual raw columns.
    missing_feature_columns = sorted(
        {
            column
            for specification in FEATURE_GROUPS.values()
            for column in specification["columns"]
            if column not in test_frame.columns
        }
    )
    if missing_feature_columns:
        raise RuntimeError(
            "Feature definitions refer to missing raw columns: "
            + ", ".join(missing_feature_columns)
        )

    feature_definition_rows = []
    for key, specification in FEATURE_GROUPS.items():
        feature_definition_rows.append(
            {
                "feature_group": key,
                "feature_group_label": specification["label"],
                "raw_columns": "; ".join(specification["columns"]),
                "description": specification["description"],
                "permutation_scope": args.permutation_scope,
            }
        )
    pd.DataFrame(feature_definition_rows).to_csv(
        output_dir / "grouped_feature_definitions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    match_ids = test_frame["_match_id_norm"].to_numpy()
    rng = np.random.default_rng(args.seed)
    replicate_rows: list[dict[str, object]] = []

    # Save each raw source column once. All columns in a group use the same
    # permutation index, preserving their within-group relationships.
    original_columns = {
        column: test_frame[column].to_numpy(copy=True)
        for specification in FEATURE_GROUPS.values()
        for column in specification["columns"]
    }

    for group_index, (group_key, specification) in enumerate(
        FEATURE_GROUPS.items(),
        start=1,
    ):
        columns = list(specification["columns"])
        print(
            "=" * 100
            + f"\nFeature group {group_index}/{len(FEATURE_GROUPS)}: "
            f"{specification['label']}\n"
            + "=" * 100
        )

        for repetition in range(1, args.repetitions + 1):
            permutation = permutation_index(
                match_ids,
                rng,
                args.permutation_scope,
            )

            for column in columns:
                test_frame[column] = original_columns[column][permutation]

            permuted_probability = predict_full_model(
                test_frame,
                xpass_module,
                full_model,
                full_calibrator,
                main_encoder,
                main_categories,
                args.prediction_batch_size,
            )
            permuted_metrics = metric_values(
                y_success_reference,
                permuted_probability,
            )
            importance = importance_values(
                baseline_metrics,
                permuted_metrics,
            )

            for metric, value in importance.items():
                replicate_rows.append(
                    {
                        "feature_group": group_key,
                        "feature_group_label": specification["label"],
                        "repetition": repetition,
                        "metric": metric,
                        "metric_label": METRICS[metric],
                        "importance": value,
                        "permutation_scope": args.permutation_scope,
                        "test_matches": actual_matches,
                        "test_rows": actual_rows,
                    }
                )

            # Restore original values before the next permutation.
            for column in columns:
                test_frame[column] = original_columns[column]

            del permuted_probability
            gc.collect()

            if repetition % 5 == 0 or repetition == args.repetitions:
                print(
                    f"[{specification['label']}] "
                    f"{repetition}/{args.repetitions}"
                )

    replicates = pd.DataFrame(replicate_rows)
    replicates.to_csv(
        output_dir / "grouped_permutation_importance_replicates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows: list[dict[str, object]] = []
    for group_key, specification in FEATURE_GROUPS.items():
        for metric, metric_label in METRICS.items():
            values = replicates.loc[
                replicates["feature_group"].eq(group_key)
                & replicates["metric"].eq(metric),
                "importance",
            ].to_numpy(dtype=np.float64)

            lower, upper = np.quantile(values, [0.025, 0.975])
            summary_rows.append(
                {
                    "feature_group": group_key,
                    "feature_group_label": specification["label"],
                    "metric": metric,
                    "metric_label": metric_label,
                    "mean_importance": float(np.mean(values)),
                    "median_importance": float(np.median(values)),
                    "standard_deviation": float(np.std(values, ddof=1)),
                    "percentile_2_5": float(lower),
                    "percentile_97_5": float(upper),
                    "repetitions": len(values),
                    "proportion_positive": float(np.mean(values > 0.0)),
                    "permutation_scope": args.permutation_scope,
                    "positive_means_performance_worsened": True,
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_dir / "grouped_permutation_importance_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_log_loss_importance(summary, output_dir)

    caption = (
        "Supplementary Figure. Grouped permutation importance for the locked "
        "Full xPass model on the independent test set. Points show the mean "
        "increase in log loss after jointly permuting all raw variables in a "
        "pre-specified feature group; horizontal intervals show the 2.5th and "
        "97.5th percentiles across repeated permutations. By default, "
        "permutations were performed within matches to preserve each match's "
        "marginal feature distribution. Positive values indicate poorer "
        "probability predictions after permutation. The intervals describe "
        "variability across random permutations and are not sampling-based "
        "confidence intervals or causal-effect estimates."
    )
    (output_dir / "grouped_feature_importance_caption.txt").write_text(
        caption,
        encoding="utf-8",
    )

    summary_json = {
        "status": "PASS",
        "purpose": (
            "Grouped permutation importance for the locked Full xPass model "
            "on the independent test set."
        ),
        "test_matches": int(actual_matches),
        "test_rows": int(actual_rows),
        "sample_matches_mode": int(args.sample_matches),
        "repetitions": int(args.repetitions),
        "random_seed": int(args.seed),
        "permutation_scope": args.permutation_scope,
        "baseline_metrics": baseline_metrics,
        "baseline_prediction_validation": {
            "maximum_absolute_difference_from_locked_predictions": (
                max_abs_difference
            ),
            "tolerance": args.prediction_tolerance,
            "passed": max_abs_difference <= args.prediction_tolerance,
        },
        "interpretation": (
            "Importance values are descriptive performance changes after "
            "grouped permutation. They are not causal effects. Percentile "
            "intervals summarize random-permutation variability rather than "
            "sampling uncertainty."
        ),
        "input_hashes": {
            "xpass_script_sha256": sha256_file(
                args.xpass_script.resolve()
            ),
            "model_bundle_sha256": sha256_file(
                args.model_bundle.resolve()
            ),
            "input_csv_sha256": sha256_file(args.input_csv.resolve()),
            "split_manifest_sha256": sha256_file(
                args.split_manifest.resolve()
            ),
            "reference_predictions_sha256": sha256_file(
                args.reference_predictions_csv.resolve()
            ),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "joblib": joblib.__version__,
        },
        "outputs": sorted(
            path.name
            for path in output_dir.iterdir()
            if path.is_file()
        ),
    }
    (
        output_dir / "grouped_permutation_importance_summary.json"
    ).write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("Grouped permutation importance completed")
    print("=" * 100)
    print(f"Test matches: {actual_matches:,}")
    print(f"Test passes: {actual_rows:,}")
    print(f"Repetitions per feature group: {args.repetitions}")
    print(
        "Maximum baseline prediction difference: "
        f"{max_abs_difference:.3e}"
    )
    print("Status: PASS")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
