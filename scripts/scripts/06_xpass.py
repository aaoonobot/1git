#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified xPass modeling, evaluation, and cross-fitting pipeline.

This is the final consolidated xPass script for the study. It replaces the
separate development scripts used for model finalization and full-sample OOF
prediction generation.

Commands
--------
evaluate
    Refit locked geometry, main, no-team, and strict-open-play models on the
    locked training+tuning matches; fit Platt calibration on calibration
    matches only; evaluate the independent test matches; and write baselines,
    reliability tables, subgroup results, match-cluster bootstrap intervals,
    coefficients, model bundles, hashes, and software-environment records.

crossfit
    Generate one calibrated match-level out-of-fold xPass probability for every
    primary-open-play pass using five outer folds. Each held-out match is absent
    from both model fitting and calibration. Main and no-team probabilities are
    exported for downstream PCTV, team, and player analyses.

verify
    Audit an existing OOF result directory without retraining a model.

Important reporting rule
------------------------
The locked independent test analysis is the primary model-performance result.
The full-sample OOF metrics are supplementary diagnostics and the OOF
probabilities are used for downstream pass-value analyses.

Target
------
blank pass_outcome -> success = 1
nonblank pass_outcome -> failure = 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder


GEOMETRY_NUMERIC_FEATURES = [
    "start_x_norm",
    "start_y_norm",
    "end_x_norm",
    "end_y_norm",
    "delta_x_norm",
    "delta_y_norm",
    "pass_length_norm",
    "sin_angle",
    "cos_angle",
    "start_goal_distance_norm",
    "end_goal_distance_norm",
]

FULL_EXTRA_NUMERIC_FEATURES = [
    "minute_norm",
    "second_norm",
    "under_pressure_num",
    "pass_cross_num",
    "pass_switch_num",
    "pass_through_ball_num",
    "pass_backheel_num",
    "pass_outswinging_num",
    "pass_inswinging_num",
    "pass_straight_num",
    "pass_cut_back_num",
]

RAW_COLUMNS = [
    "id",
    "match_id",
    "competition_id",
    "season_id",
    "pass_outcome",
    "primary_open_play",
    "strict_open_play",
    "start_x_model",
    "start_y_model",
    "end_x_model",
    "end_y_model",
    "pass_length",
    "pass_angle",
    "minute",
    "second",
    "under_pressure",
    "pass_cross",
    "pass_switch",
    "pass_through_ball",
    "pass_backheel",
    "pass_outswinging",
    "pass_inswinging",
    "pass_straight",
    "pass_cut_back",
    "pass_height",
    "pass_body_part",
    "pass_type",
    "play_pattern",
    "position",
    "pass_technique",
    "team",
    "possession_team",
]

BOOTSTRAP_METRICS = [
    "roc_auc_success",
    "pr_auc_failure",
    "log_loss",
    "brier_score",
    "calibration_slope",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified reproducible xPass evaluation, cross-fitting, and "
            "verification pipeline."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate = commands.add_parser(
        "evaluate",
        help=(
            "Run the locked independent-test evaluation. This reads the test "
            "set and should not be used for further model selection."
        ),
    )
    evaluate.add_argument("--input_csv", type=Path, required=True)
    evaluate.add_argument("--split_manifest", type=Path, required=True)
    evaluate.add_argument("--main_tuning_json", type=Path, required=True)
    evaluate.add_argument("--no_team_tuning_json", type=Path, required=True)
    evaluate.add_argument("--competitions_csv", type=Path, required=True)
    evaluate.add_argument("--output_dir", type=Path, required=True)
    evaluate.add_argument("--chunksize", type=int, default=75_000)
    evaluate.add_argument("--seed", type=int, default=20260723)
    evaluate.add_argument("--bootstrap_reps", type=int, default=1000)
    evaluate.add_argument("--bootstrap_seed", type=int, default=20260724)
    evaluate.add_argument("--overwrite", action="store_true")
    evaluate.add_argument(
        "--confirm_test_evaluation",
        action="store_true",
        help=(
            "Required acknowledgement that the independent test split will be "
            "read and must not be used for additional model selection."
        ),
    )
    evaluate.add_argument(
        "--unlock_test",
        action="store_true",
        help=(
            "Required, together with --overwrite, only to replace an existing "
            "test-evaluation record in the same output directory."
        ),
    )

    crossfit = commands.add_parser(
        "crossfit",
        help=(
            "Generate calibrated five-fold match-level OOF xPass predictions "
            "for downstream PCTV and team/player analyses."
        ),
    )
    crossfit.add_argument("--input_csv", type=Path, required=True)
    crossfit.add_argument("--split_manifest", type=Path, required=True)
    crossfit.add_argument("--main_tuning_json", type=Path, required=True)
    crossfit.add_argument("--no_team_tuning_json", type=Path, required=True)
    crossfit.add_argument("--output_dir", type=Path, required=True)
    crossfit.add_argument("--outer_folds", type=int, default=5)
    crossfit.add_argument(
        "--calibration_fraction_of_development",
        type=float,
        default=0.125,
        help=(
            "Within each outer-fold development set, fraction of matches "
            "reserved for Platt calibration."
        ),
    )
    crossfit.add_argument("--chunksize", type=int, default=75_000)
    crossfit.add_argument("--seed", type=int, default=20260723)
    crossfit.add_argument("--expected_matches", type=int, default=3926)
    crossfit.add_argument(
        "--expected_primary_rows",
        type=int,
        default=3_396_912,
    )
    crossfit.add_argument(
        "--save_fold_models",
        action="store_true",
        help="Optionally save the five fitted fold bundles.",
    )
    crossfit.add_argument("--overwrite", action="store_true")

    verify = commands.add_parser(
        "verify",
        help="Audit an existing OOF output directory without refitting models.",
    )
    verify.add_argument("--oof_dir", type=Path, required=True)
    verify.add_argument("--expected_matches", type=int, default=3926)
    verify.add_argument(
        "--expected_primary_rows",
        type=int,
        default=3_396_912,
    )

    return parser.parse_args()

def json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): json_native(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_native(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except Exception:
            pass
    return text


def normalize_category_frame(
    frame: pd.DataFrame,
    categorical_features: list[str],
) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for column in categorical_features:
        out[column] = (
            frame[column]
            .astype("string")
            .fillna("<MISSING>")
            .str.strip()
            .replace("", "<MISSING>")
        )
    return out


def bool_to_float(series: pd.Series) -> np.ndarray:
    text = (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )
    result = np.zeros(len(series), dtype=np.float32)
    result[text.isin(["true", "1", "1.0", "yes"])] = 1.0
    return result


def derive_numeric(frame: pd.DataFrame, full: bool) -> np.ndarray:
    sx = pd.to_numeric(
        frame["start_x_model"], errors="coerce"
    ).to_numpy(dtype=np.float32)
    sy = pd.to_numeric(
        frame["start_y_model"], errors="coerce"
    ).to_numpy(dtype=np.float32)
    ex = pd.to_numeric(
        frame["end_x_model"], errors="coerce"
    ).to_numpy(dtype=np.float32)
    ey = pd.to_numeric(
        frame["end_y_model"], errors="coerce"
    ).to_numpy(dtype=np.float32)

    length = pd.to_numeric(
        frame["pass_length"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=np.float32)
    angle = pd.to_numeric(
        frame["pass_angle"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=np.float32)

    sx = np.nan_to_num(sx, nan=0.0)
    sy = np.nan_to_num(sy, nan=0.0)
    ex = np.nan_to_num(ex, nan=0.0)
    ey = np.nan_to_num(ey, nan=0.0)

    goal_norm = math.sqrt(120.0**2 + 40.0**2)

    features = [
        sx / 120.0,
        sy / 80.0,
        ex / 120.0,
        ey / 80.0,
        (ex - sx) / 120.0,
        (ey - sy) / 80.0,
        length / 120.0,
        np.sin(angle),
        np.cos(angle),
        np.sqrt((120.0 - sx) ** 2 + (40.0 - sy) ** 2)
        / goal_norm,
        np.sqrt((120.0 - ex) ** 2 + (40.0 - ey) ** 2)
        / goal_norm,
    ]

    if full:
        minute = pd.to_numeric(
            frame["minute"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=np.float32)
        second = pd.to_numeric(
            frame["second"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=np.float32)

        features.extend(
            [
                minute / 130.0,
                second / 60.0,
                bool_to_float(frame["under_pressure"]),
                bool_to_float(frame["pass_cross"]),
                bool_to_float(frame["pass_switch"]),
                bool_to_float(frame["pass_through_ball"]),
                bool_to_float(frame["pass_backheel"]),
                bool_to_float(frame["pass_outswinging"]),
                bool_to_float(frame["pass_inswinging"]),
                bool_to_float(frame["pass_straight"]),
                bool_to_float(frame["pass_cut_back"]),
            ]
        )

    return np.column_stack(features).astype(np.float32)


def target_from_frame(frame: pd.DataFrame) -> np.ndarray:
    outcome = (
        frame["pass_outcome"]
        .astype("string")
        .fillna("")
        .str.strip()
    )
    return outcome.eq("").astype(np.int8).to_numpy()


def load_split_map(path: Path) -> dict[str, str]:
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"match_id": "string"},
        low_memory=False,
    )
    required = {"match_id", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"Split manifest missing columns: {missing}"
        )

    mapping: dict[str, str] = {}
    for match_id, split in zip(frame["match_id"], frame["split"]):
        match_text = normalize_id(match_id)
        split_text = str(split).strip()
        if not match_text:
            raise RuntimeError("Empty match_id in split manifest.")
        if match_text in mapping:
            raise RuntimeError(
                f"Duplicate match_id in split manifest: {match_text}"
            )
        mapping[match_text] = split_text

    expected_splits = {
        "training",
        "tuning",
        "calibration",
        "test",
    }
    found_splits = set(mapping.values())
    if not expected_splits.issubset(found_splits):
        raise RuntimeError(
            f"Split manifest does not contain all required splits: "
            f"{expected_splits - found_splits}"
        )

    return mapping


def load_competition_gender_map(
    path: Path,
) -> dict[tuple[str, str], str]:
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )
    required = {
        "competition_id",
        "season_id",
        "competition_gender",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"Competitions file missing columns: {missing}"
        )

    mapping = {}
    for _, row in frame.iterrows():
        key = (
            normalize_id(row["competition_id"]),
            normalize_id(row["season_id"]),
        )
        gender = (
            str(row["competition_gender"])
            .strip()
            .lower()
        )
        mapping[key] = gender if gender else "unknown"
    return mapping


def load_tuning_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if data.get("status") != "PASS":
        raise RuntimeError(
            f"Tuning file is not PASS: {path}"
        )
    if data.get("test_set_read") is not False:
        raise RuntimeError(
            f"Tuning file does not certify an untouched test set: {path}"
        )
    return data


def make_encoder(
    category_map: dict[str, list[str]],
    categorical_features: list[str],
) -> OneHotEncoder:
    categories = [
        category_map[column]
        for column in categorical_features
    ]

    try:
        encoder = OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32,
        )
    except TypeError:
        encoder = OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            sparse=True,
            dtype=np.float32,
        )

    dummy = pd.DataFrame(
        {
            column: [category_map[column][0]]
            for column in categorical_features
        }
    )
    encoder.fit(dummy)
    return encoder


def make_classifier(
    config: dict[str, float],
    seed: int,
) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=float(config["alpha"]),
        l1_ratio=float(config["l1_ratio"]),
        fit_intercept=True,
        learning_rate="optimal",
        average=True,
        random_state=seed,
        shuffle=False,
        tol=None,
        max_iter=1,
        warm_start=True,
    )


def build_matrix(
    frame: pd.DataFrame,
    model_kind: str,
    encoder: OneHotEncoder | None,
    categorical_features: list[str],
) -> sparse.csr_matrix:
    full = model_kind == "full"
    numeric = sparse.csr_matrix(
        derive_numeric(frame, full=full)
    )

    if not full:
        return numeric

    if encoder is None:
        raise RuntimeError(
            "Full model requires a fitted encoder."
        )

    categorical = normalize_category_frame(
        frame,
        categorical_features,
    )
    encoded = encoder.transform(categorical)

    return sparse.hstack(
        [numeric, encoded],
        format="csr",
        dtype=np.float32,
    )


def is_true_flag(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "1.0"])
    )


def read_filtered_chunks(
    input_csv: Path,
    split_map: dict[str, str],
    allowed_splits: set[str],
    filter_column: str,
    chunksize: int,
    shuffle_seed: int | None = None,
) -> Iterable[pd.DataFrame]:
    header = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    missing = sorted(set(RAW_COLUMNS) - set(header))
    if missing:
        raise RuntimeError(
            f"Input CSV missing required columns: {missing}"
        )

    reader = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        usecols=RAW_COLUMNS,
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        match_ids = chunk["match_id"].map(normalize_id)
        split = match_ids.map(split_map)

        if split.isna().any():
            missing_ids = sorted(
                set(match_ids.loc[split.isna()])
            )
            raise RuntimeError(
                "Pass rows contain match IDs absent from split manifest: "
                + ", ".join(missing_ids[:20])
            )

        filter_mask = is_true_flag(chunk[filter_column])
        selected_mask = (
            filter_mask & split.isin(allowed_splits)
        )
        selected = chunk.loc[selected_mask].copy()

        if selected.empty:
            continue

        selected["_split"] = (
            split.loc[selected_mask].to_numpy()
        )

        if shuffle_seed is not None:
            selected = selected.sample(
                frac=1.0,
                random_state=shuffle_seed + chunk_number,
            )

        yield selected.reset_index(drop=True)


def fit_primary_models_joint(
    input_csv: Path,
    split_map: dict[str, str],
    chunksize: int,
    model_specs: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, SGDClassifier]:
    models = {}
    for index, (name, spec) in enumerate(
        model_specs.items(),
        start=1,
    ):
        models[name] = make_classifier(
            spec["config"],
            seed + index,
        )

    max_epochs = max(
        int(spec["epochs"])
        for spec in model_specs.values()
    )
    classes = np.array([0, 1], dtype=np.int8)

    for epoch in range(1, max_epochs + 1):
        rows = 0

        for chunk in read_filtered_chunks(
            input_csv,
            split_map,
            {"training", "tuning"},
            "primary_open_play",
            chunksize,
            shuffle_seed=seed + epoch * 10_000,
        ):
            y = target_from_frame(chunk)
            rows += len(y)

            for name, spec in model_specs.items():
                if epoch > int(spec["epochs"]):
                    continue

                matrix = build_matrix(
                    chunk,
                    spec["model_kind"],
                    spec["encoder"],
                    spec["categorical_features"],
                )
                models[name].partial_fit(
                    matrix,
                    y,
                    classes=classes,
                )

        print(
            f"[primary models] epoch={epoch}/{max_epochs}, "
            f"rows={rows:,}"
        )

    return models


def fit_strict_model(
    input_csv: Path,
    split_map: dict[str, str],
    chunksize: int,
    spec: dict[str, Any],
    seed: int,
) -> SGDClassifier:
    model = make_classifier(
        spec["config"],
        seed,
    )
    classes = np.array([0, 1], dtype=np.int8)
    epochs = int(spec["epochs"])

    for epoch in range(1, epochs + 1):
        rows = 0

        for chunk in read_filtered_chunks(
            input_csv,
            split_map,
            {"training", "tuning"},
            "strict_open_play",
            chunksize,
            shuffle_seed=seed + epoch * 20_000,
        ):
            matrix = build_matrix(
                chunk,
                spec["model_kind"],
                spec["encoder"],
                spec["categorical_features"],
            )
            y = target_from_frame(chunk)

            model.partial_fit(
                matrix,
                y,
                classes=classes,
            )
            rows += len(y)

        print(
            f"[strict full] epoch={epoch}/{epochs}, "
            f"rows={rows:,}"
        )

    return model


def safe_predict(
    model: SGDClassifier,
    matrix: sparse.csr_matrix,
) -> np.ndarray:
    probability = model.predict_proba(matrix)[:, 1]
    return np.clip(
        probability,
        1e-7,
        1.0 - 1e-7,
    )


def predict_models(
    input_csv: Path,
    split_map: dict[str, str],
    allowed_splits: set[str],
    filter_column: str,
    chunksize: int,
    model_specs: dict[str, dict[str, Any]],
    models: dict[str, SGDClassifier],
    gender_map: dict[tuple[str, str], str],
) -> tuple[np.ndarray, dict[str, np.ndarray], pd.DataFrame]:
    y_parts = []
    prediction_parts = {
        name: []
        for name in model_specs
    }
    metadata_parts = []

    for chunk in read_filtered_chunks(
        input_csv,
        split_map,
        allowed_splits,
        filter_column,
        chunksize,
    ):
        y = target_from_frame(chunk)
        y_parts.append(y)

        metadata = chunk[
            [
                "id",
                "match_id",
                "competition_id",
                "season_id",
                "strict_open_play",
            ]
        ].copy()

        metadata["match_id"] = (
            metadata["match_id"].map(normalize_id)
        )
        metadata["competition_gender"] = [
            gender_map.get(
                (
                    normalize_id(comp),
                    normalize_id(season),
                ),
                "unknown",
            )
            for comp, season in zip(
                metadata["competition_id"],
                metadata["season_id"],
            )
        ]
        metadata_parts.append(metadata)

        for name, spec in model_specs.items():
            matrix = build_matrix(
                chunk,
                spec["model_kind"],
                spec["encoder"],
                spec["categorical_features"],
            )
            prediction_parts[name].append(
                safe_predict(models[name], matrix)
            )

    if not y_parts:
        raise RuntimeError(
            f"No rows for splits={allowed_splits}, "
            f"filter={filter_column}."
        )

    y_all = np.concatenate(y_parts)
    predictions = {
        name: np.concatenate(parts)
        for name, parts in prediction_parts.items()
    }
    metadata_all = pd.concat(
        metadata_parts,
        ignore_index=True,
    )

    return y_all, predictions, metadata_all


def collect_prevalence(
    input_csv: Path,
    split_map: dict[str, str],
    filter_column: str,
    chunksize: int,
) -> tuple[int, int, float]:
    rows = 0
    successes = 0

    for chunk in read_filtered_chunks(
        input_csv,
        split_map,
        {"training", "tuning"},
        filter_column,
        chunksize,
    ):
        y = target_from_frame(chunk)
        rows += len(y)
        successes += int(y.sum())

    if rows == 0:
        raise RuntimeError(
            f"No training+tuning rows for {filter_column}."
        )

    return rows, successes, successes / rows


def fit_platt(
    y_success: np.ndarray,
    p_success: np.ndarray,
) -> LogisticRegression:
    p = np.clip(
        p_success,
        1e-7,
        1.0 - 1e-7,
    )
    logit_values = np.log(
        p / (1.0 - p)
    ).reshape(-1, 1)

    try:
        calibrator = LogisticRegression(
            solver="lbfgs",
            penalty=None,
            max_iter=1000,
            random_state=0,
        )
        calibrator.fit(logit_values, y_success)
    except Exception:
        calibrator = LogisticRegression(
            solver="lbfgs",
            C=1e12,
            max_iter=1000,
            random_state=0,
        )
        calibrator.fit(logit_values, y_success)

    return calibrator


def apply_platt(
    calibrator: LogisticRegression,
    p_success: np.ndarray,
) -> np.ndarray:
    p = np.clip(
        p_success,
        1e-7,
        1.0 - 1e-7,
    )
    logit_values = np.log(
        p / (1.0 - p)
    ).reshape(-1, 1)

    return np.clip(
        calibrator.predict_proba(logit_values)[:, 1],
        1e-7,
        1.0 - 1e-7,
    )


def weighted_calibration_slope(
    y_success: np.ndarray,
    p_success: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, float]:
    p = np.clip(
        np.asarray(p_success, dtype=np.float64),
        1e-7,
        1.0 - 1e-7,
    )
    y = np.asarray(y_success, dtype=np.int8)
    logit_values = np.log(
        p / (1.0 - p)
    ).reshape(-1, 1)

    if np.std(logit_values) < 1e-12:
        prevalence = np.average(
            y,
            weights=sample_weight,
        )
        prevalence = np.clip(
            prevalence,
            1e-7,
            1.0 - 1e-7,
        )
        intercept = float(
            np.log(prevalence / (1.0 - prevalence))
        )
        return intercept, 0.0

    try:
        model = LogisticRegression(
            solver="lbfgs",
            penalty=None,
            max_iter=1000,
        )
        model.fit(
            logit_values,
            y,
            sample_weight=sample_weight,
        )
    except Exception:
        model = LogisticRegression(
            solver="lbfgs",
            C=1e12,
            max_iter=1000,
        )
        model.fit(
            logit_values,
            y,
            sample_weight=sample_weight,
        )

    return (
        float(model.intercept_[0]),
        float(model.coef_[0][0]),
    )


def compute_metrics(
    y_success: np.ndarray,
    p_success: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(y_success, dtype=np.int8)
    p = np.clip(
        np.asarray(p_success, dtype=np.float64),
        1e-7,
        1.0 - 1e-7,
    )

    y_failure = 1 - y
    p_failure = 1.0 - p
    predicted_success = (p >= 0.5).astype(np.int8)
    predicted_failure = 1 - predicted_success

    intercept, slope = weighted_calibration_slope(
        y,
        p,
    )

    return {
        "rows": int(len(y)),
        "success_prevalence": float(y.mean()),
        "failure_prevalence": float(y_failure.mean()),
        "roc_auc_success": float(
            roc_auc_score(y, p)
        ),
        "pr_auc_failure": float(
            average_precision_score(
                y_failure,
                p_failure,
            )
        ),
        "log_loss": float(
            log_loss(y, p, labels=[0, 1])
        ),
        "brier_score": float(
            brier_score_loss(y, p)
        ),
        "accuracy": float(
            accuracy_score(y, predicted_success)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_failure,
                predicted_failure,
            )
        ),
        "failure_precision": float(
            precision_score(
                y_failure,
                predicted_failure,
                zero_division=0,
            )
        ),
        "failure_recall": float(
            recall_score(
                y_failure,
                predicted_failure,
                zero_division=0,
            )
        ),
        "failure_f1": float(
            f1_score(
                y_failure,
                predicted_failure,
                zero_division=0,
            )
        ),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def reliability_table(
    y_success: np.ndarray,
    p_success: np.ndarray,
    bins: int = 20,
) -> tuple[pd.DataFrame, float]:
    y = np.asarray(y_success, dtype=np.int8)
    p = np.clip(
        np.asarray(p_success, dtype=np.float64),
        1e-7,
        1.0 - 1e-7,
    )

    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_id = np.clip(
        np.digitize(p, edges, right=True) - 1,
        0,
        bins - 1,
    )

    rows = []
    ece = 0.0
    n = len(y)

    for index in range(bins):
        mask = bin_id == index
        count = int(mask.sum())

        if count == 0:
            rows.append(
                {
                    "bin": index + 1,
                    "lower": edges[index],
                    "upper": edges[index + 1],
                    "count": 0,
                    "mean_predicted_success": np.nan,
                    "observed_success_rate": np.nan,
                    "absolute_gap": np.nan,
                }
            )
            continue

        mean_probability = float(p[mask].mean())
        observed = float(y[mask].mean())
        gap = abs(mean_probability - observed)
        ece += count / n * gap

        rows.append(
            {
                "bin": index + 1,
                "lower": edges[index],
                "upper": edges[index + 1],
                "count": count,
                "mean_predicted_success": mean_probability,
                "observed_success_rate": observed,
                "absolute_gap": gap,
            }
        )

    return pd.DataFrame(rows), float(ece)


def evaluate_prediction_set(
    cohort: str,
    y_success: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    reliability_rows = []

    for model_name, probability in predictions.items():
        metrics = compute_metrics(
            y_success,
            probability,
        )
        reliability, ece = reliability_table(
            y_success,
            probability,
            bins=20,
        )
        metrics["ece_20_bins"] = ece

        metric_rows.append(
            {
                "cohort": cohort,
                "model": model_name,
                **metrics,
            }
        )

        reliability.insert(0, "model", model_name)
        reliability.insert(0, "cohort", cohort)
        reliability_rows.append(reliability)

    return (
        pd.DataFrame(metric_rows),
        pd.concat(reliability_rows, ignore_index=True),
    )


def subgroup_metrics_by_gender(
    y_success: np.ndarray,
    predictions: dict[str, np.ndarray],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    genders = sorted(
        metadata["competition_gender"]
        .astype("string")
        .fillna("unknown")
        .unique()
        .tolist()
    )

    for gender in genders:
        mask = (
            metadata["competition_gender"]
            .astype("string")
            .eq(gender)
            .to_numpy()
        )

        if mask.sum() == 0:
            continue

        for model_name, probability in predictions.items():
            metrics = compute_metrics(
                y_success[mask],
                probability[mask],
            )
            _, ece = reliability_table(
                y_success[mask],
                probability[mask],
                bins=20,
            )
            metrics["ece_20_bins"] = ece

            rows.append(
                {
                    "competition_gender": gender,
                    "model": model_name,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def bootstrap_metric_set(
    y_success: np.ndarray,
    p_success: np.ndarray,
    match_ids: np.ndarray,
    reps: int,
    seed: int,
) -> dict[str, list[float]]:
    y = np.asarray(y_success, dtype=np.int8)
    p = np.clip(
        np.asarray(p_success, dtype=np.float64),
        1e-7,
        1.0 - 1e-7,
    )
    match_ids = np.asarray(match_ids, dtype=str)

    unique_matches, match_codes = np.unique(
        match_ids,
        return_inverse=True,
    )
    match_count = len(unique_matches)
    rng = np.random.default_rng(seed)

    output = {
        metric: []
        for metric in BOOTSTRAP_METRICS
    }

    for replicate in range(1, reps + 1):
        sampled = rng.integers(
            0,
            match_count,
            size=match_count,
        )
        match_weights = np.bincount(
            sampled,
            minlength=match_count,
        ).astype(np.float64)
        row_weights = match_weights[match_codes]

        y_failure = 1 - y
        p_failure = 1.0 - p

        try:
            output["roc_auc_success"].append(
                float(
                    roc_auc_score(
                        y,
                        p,
                        sample_weight=row_weights,
                    )
                )
            )
        except Exception:
            output["roc_auc_success"].append(np.nan)

        try:
            output["pr_auc_failure"].append(
                float(
                    average_precision_score(
                        y_failure,
                        p_failure,
                        sample_weight=row_weights,
                    )
                )
            )
        except Exception:
            output["pr_auc_failure"].append(np.nan)

        try:
            output["log_loss"].append(
                float(
                    log_loss(
                        y,
                        p,
                        sample_weight=row_weights,
                        labels=[0, 1],
                    )
                )
            )
        except Exception:
            output["log_loss"].append(np.nan)

        try:
            output["brier_score"].append(
                float(
                    brier_score_loss(
                        y,
                        p,
                        sample_weight=row_weights,
                    )
                )
            )
        except Exception:
            output["brier_score"].append(np.nan)

        try:
            _, slope = weighted_calibration_slope(
                y,
                p,
                sample_weight=row_weights,
            )
            output["calibration_slope"].append(slope)
        except Exception:
            output["calibration_slope"].append(np.nan)

        if replicate % 50 == 0 or replicate == reps:
            print(
                f"[bootstrap] replicate={replicate}/{reps}"
            )

    return output


def bootstrap_ci_table(
    cohort: str,
    model_name: str,
    y_success: np.ndarray,
    p_success: np.ndarray,
    match_ids: np.ndarray,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    estimates = compute_metrics(
        y_success,
        p_success,
    )
    bootstrap_values = bootstrap_metric_set(
        y_success,
        p_success,
        match_ids,
        reps,
        seed,
    )

    rows = []
    for metric in BOOTSTRAP_METRICS:
        values = np.asarray(
            bootstrap_values[metric],
            dtype=np.float64,
        )
        valid = values[np.isfinite(values)]

        if len(valid) == 0:
            lower = np.nan
            upper = np.nan
        else:
            lower, upper = np.quantile(
                valid,
                [0.025, 0.975],
            )

        rows.append(
            {
                "cohort": cohort,
                "model": model_name,
                "metric": metric,
                "estimate": estimates[metric],
                "ci_lower_2_5": lower,
                "ci_upper_97_5": upper,
                "bootstrap_reps_requested": reps,
                "bootstrap_reps_valid": len(valid),
                "bootstrap_unit": "match_id",
            }
        )

    return pd.DataFrame(rows)


def feature_names(
    categorical_features: list[str],
    encoder: OneHotEncoder | None,
    model_kind: str,
) -> list[str]:
    if model_kind == "geometry":
        return list(GEOMETRY_NUMERIC_FEATURES)

    numeric = (
        GEOMETRY_NUMERIC_FEATURES
        + FULL_EXTRA_NUMERIC_FEATURES
    )

    if encoder is None:
        raise RuntimeError(
            "Full model requires encoder for feature names."
        )

    try:
        encoded = encoder.get_feature_names_out(
            categorical_features
        ).tolist()
    except Exception:
        encoded = []
        for column, categories in zip(
            categorical_features,
            encoder.categories_,
        ):
            encoded.extend(
                [
                    f"{column}_{category}"
                    for category in categories
                ]
            )

    return numeric + encoded


def save_coefficients(
    model: SGDClassifier,
    names: list[str],
    output_csv: Path,
) -> None:
    coefficients = model.coef_.ravel()
    if len(coefficients) != len(names):
        raise RuntimeError(
            "Coefficient count does not match feature names."
        )

    table = pd.DataFrame(
        {
            "feature": names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values(
        "absolute_coefficient",
        ascending=False,
    )
    table.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )


def run_evaluate(args: argparse.Namespace) -> None:
    if not args.confirm_test_evaluation:
        raise RuntimeError(
            "The evaluate command reads the locked independent test set. "
            "Re-run with --confirm_test_evaluation only for documented reproduction, "
            "not for additional model selection."
        )


    input_csv = args.input_csv.resolve()
    split_manifest = args.split_manifest.resolve()
    main_tuning_json = args.main_tuning_json.resolve()
    no_team_tuning_json = (
        args.no_team_tuning_json.resolve()
    )
    competitions_csv = args.competitions_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        input_csv,
        split_manifest,
        main_tuning_json,
        no_team_tuning_json,
        competitions_csv,
    ]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)

    final_record = (
        output_dir / "final_test_evaluation_record.json"
    )

    if final_record.exists():
        if not args.unlock_test:
            raise FileExistsError(
                "A final test evaluation already exists. Re-running would "
                "inspect the locked test set again. Use --unlock_test only "
                "with a documented methodological reason."
            )
        if not args.overwrite:
            raise FileExistsError(
                "Use both --unlock_test and --overwrite to replace "
                "an existing final test evaluation."
            )

    if args.overwrite:
        for path in output_dir.iterdir():
            if path.is_file():
                path.unlink()

    split_map = load_split_map(split_manifest)
    gender_map = load_competition_gender_map(
        competitions_csv
    )

    main_selection = load_tuning_json(
        main_tuning_json
    )
    no_team_selection = load_tuning_json(
        no_team_tuning_json
    )

    main_categories = list(
        main_selection["categorical_features"]
    )
    no_team_categories = list(
        no_team_selection["categorical_features"]
    )

    expected_main_identity = {
        "team",
        "possession_team",
    }
    if not expected_main_identity.issubset(
        set(main_categories)
    ):
        raise RuntimeError(
            "Main tuning result does not contain team identity features."
        )

    if expected_main_identity & set(
        no_team_categories
    ):
        raise RuntimeError(
            "No-team tuning result still contains team identity features."
        )

    main_encoder = make_encoder(
        main_selection["category_map"],
        main_categories,
    )
    no_team_encoder = make_encoder(
        no_team_selection["category_map"],
        no_team_categories,
    )

    geometry_spec = {
        "model_kind": "geometry",
        "encoder": None,
        "categorical_features": [],
        "config": main_selection["geometry_best"]["config"],
        "epochs": int(
            main_selection["geometry_best"]["epoch"]
        ),
    }
    full_spec = {
        "model_kind": "full",
        "encoder": main_encoder,
        "categorical_features": main_categories,
        "config": main_selection["full_best"]["config"],
        "epochs": int(
            main_selection["full_best"]["epoch"]
        ),
    }
    no_team_spec = {
        "model_kind": "full",
        "encoder": no_team_encoder,
        "categorical_features": no_team_categories,
        "config": no_team_selection["full_best"]["config"],
        "epochs": int(
            no_team_selection["full_best"]["epoch"]
        ),
    }

    primary_specs = {
        "geometry": geometry_spec,
        "full": full_spec,
        "no_team": no_team_spec,
    }

    print("=" * 100)
    print("Training locked models using training and tuning matches only")
    print("=" * 100)

    primary_models = fit_primary_models_joint(
        input_csv,
        split_map,
        args.chunksize,
        primary_specs,
        args.seed,
    )

    strict_full_model = fit_strict_model(
        input_csv,
        split_map,
        args.chunksize,
        full_spec,
        args.seed + 10_000,
    )

    print("=" * 100)
    print("Fitting calibrators using calibration matches only")
    print("=" * 100)

    y_cal_primary, cal_primary_raw, cal_primary_metadata = (
        predict_models(
            input_csv,
            split_map,
            {"calibration"},
            "primary_open_play",
            args.chunksize,
            primary_specs,
            primary_models,
            gender_map,
        )
    )

    primary_calibrators = {
        name: fit_platt(
            y_cal_primary,
            probability,
        )
        for name, probability in cal_primary_raw.items()
    }

    strict_specs = {
        "strict_full": full_spec,
    }
    strict_models = {
        "strict_full": strict_full_model,
    }

    y_cal_strict, cal_strict_raw, _ = predict_models(
        input_csv,
        split_map,
        {"calibration"},
        "strict_open_play",
        args.chunksize,
        strict_specs,
        strict_models,
        gender_map,
    )
    strict_calibrator = fit_platt(
        y_cal_strict,
        cal_strict_raw["strict_full"],
    )

    primary_train_rows, primary_train_successes, primary_prevalence = (
        collect_prevalence(
            input_csv,
            split_map,
            "primary_open_play",
            args.chunksize,
        )
    )
    strict_train_rows, strict_train_successes, strict_prevalence = (
        collect_prevalence(
            input_csv,
            split_map,
            "strict_open_play",
            args.chunksize,
        )
    )

    print("=" * 100)
    print("All model and calibration choices are locked; evaluating the independent test set")
    print("=" * 100)

    y_test_primary, test_primary_raw, test_primary_metadata = (
        predict_models(
            input_csv,
            split_map,
            {"test"},
            "primary_open_play",
            args.chunksize,
            primary_specs,
            primary_models,
            gender_map,
        )
    )

    y_test_strict, test_strict_raw, test_strict_metadata = (
        predict_models(
            input_csv,
            split_map,
            {"test"},
            "strict_open_play",
            args.chunksize,
            strict_specs,
            strict_models,
            gender_map,
        )
    )

    primary_predictions = {
        "majority_success": np.full(
            len(y_test_primary),
            1.0 - 1e-7,
            dtype=np.float64,
        ),
        "constant_probability": np.full(
            len(y_test_primary),
            primary_prevalence,
            dtype=np.float64,
        ),
    }

    for name, probability in test_primary_raw.items():
        primary_predictions[
            f"{name}_uncalibrated"
        ] = probability
        primary_predictions[
            f"{name}_calibrated"
        ] = apply_platt(
            primary_calibrators[name],
            probability,
        )

    strict_predictions = {
        "strict_majority_success": np.full(
            len(y_test_strict),
            1.0 - 1e-7,
            dtype=np.float64,
        ),
        "strict_constant_probability": np.full(
            len(y_test_strict),
            strict_prevalence,
            dtype=np.float64,
        ),
        "strict_full_uncalibrated":
            test_strict_raw["strict_full"],
        "strict_full_calibrated": apply_platt(
            strict_calibrator,
            test_strict_raw["strict_full"],
        ),
    }

    primary_metrics, primary_reliability = (
        evaluate_prediction_set(
            "primary_open_play_test",
            y_test_primary,
            primary_predictions,
        )
    )
    strict_metrics, strict_reliability = (
        evaluate_prediction_set(
            "strict_open_play_test",
            y_test_strict,
            strict_predictions,
        )
    )

    all_metrics = pd.concat(
        [primary_metrics, strict_metrics],
        ignore_index=True,
    )
    all_metrics.to_csv(
        output_dir / "final_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        [primary_reliability, strict_reliability],
        ignore_index=True,
    ).to_csv(
        output_dir / "final_test_reliability_bins.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gender_metrics = subgroup_metrics_by_gender(
        y_test_primary,
        primary_predictions,
        test_primary_metadata,
    )
    gender_metrics.to_csv(
        output_dir / "final_test_gender_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Event-level prediction file for reproducibility and downstream analysis.
    primary_event_predictions = (
        test_primary_metadata.copy()
    )
    primary_event_predictions[
        "xpass_success"
    ] = y_test_primary

    for name, probability in primary_predictions.items():
        primary_event_predictions[
            f"p_success_{name}"
        ] = probability
        primary_event_predictions[
            f"p_failure_{name}"
        ] = 1.0 - probability

    primary_event_predictions.to_csv(
        output_dir / "final_test_event_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    strict_event_predictions = (
        test_strict_metadata.copy()
    )
    strict_event_predictions[
        "xpass_success"
    ] = y_test_strict

    for name, probability in strict_predictions.items():
        strict_event_predictions[
            f"p_success_{name}"
        ] = probability
        strict_event_predictions[
            f"p_failure_{name}"
        ] = 1.0 - probability

    strict_event_predictions.to_csv(
        output_dir
        / "strict_open_play_test_event_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Direct model-comparison deltas.
    metric_lookup = all_metrics.set_index(
        ["cohort", "model"]
    )

    comparison_rows = []
    comparisons = [
        (
            "full_calibrated_vs_geometry_calibrated",
            "primary_open_play_test",
            "full_calibrated",
            "geometry_calibrated",
        ),
        (
            "full_calibrated_vs_no_team_calibrated",
            "primary_open_play_test",
            "full_calibrated",
            "no_team_calibrated",
        ),
        (
            "full_calibrated_vs_constant_probability",
            "primary_open_play_test",
            "full_calibrated",
            "constant_probability",
        ),
    ]

    comparison_metrics = [
        "roc_auc_success",
        "pr_auc_failure",
        "log_loss",
        "brier_score",
        "calibration_slope",
        "ece_20_bins",
    ]

    for label, cohort, model_a, model_b in comparisons:
        for metric in comparison_metrics:
            value_a = float(
                metric_lookup.loc[
                    (cohort, model_a),
                    metric,
                ]
            )
            value_b = float(
                metric_lookup.loc[
                    (cohort, model_b),
                    metric,
                ]
            )
            comparison_rows.append(
                {
                    "comparison": label,
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    "model_a_value": value_a,
                    "model_b_value": value_b,
                    "difference_a_minus_b":
                        value_a - value_b,
                }
            )

    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "locked_model_comparisons.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Match-cluster bootstrap CIs for the pre-specified key models.
    bootstrap_tables = []

    bootstrap_targets = [
        (
            "primary_open_play_test",
            "geometry_calibrated",
            y_test_primary,
            primary_predictions["geometry_calibrated"],
            test_primary_metadata["match_id"].to_numpy(),
        ),
        (
            "primary_open_play_test",
            "full_calibrated",
            y_test_primary,
            primary_predictions["full_calibrated"],
            test_primary_metadata["match_id"].to_numpy(),
        ),
        (
            "primary_open_play_test",
            "no_team_calibrated",
            y_test_primary,
            primary_predictions["no_team_calibrated"],
            test_primary_metadata["match_id"].to_numpy(),
        ),
        (
            "strict_open_play_test",
            "strict_full_calibrated",
            y_test_strict,
            strict_predictions["strict_full_calibrated"],
            test_strict_metadata["match_id"].to_numpy(),
        ),
    ]

    for index, (
        cohort,
        model_name,
        y_values,
        p_values,
        match_ids,
    ) in enumerate(bootstrap_targets):
        print(
            f"Starting match-cluster bootstrap: {cohort} / {model_name}"
        )
        table = bootstrap_ci_table(
            cohort,
            model_name,
            y_values,
            p_values,
            match_ids,
            args.bootstrap_reps,
            args.bootstrap_seed + index * 1000,
        )
        bootstrap_tables.append(table)

    pd.concat(
        bootstrap_tables,
        ignore_index=True,
    ).to_csv(
        output_dir / "match_cluster_bootstrap_95ci.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Coefficients.
    save_coefficients(
        primary_models["geometry"],
        feature_names([], None, "geometry"),
        output_dir / "geometry_model_coefficients.csv",
    )
    save_coefficients(
        primary_models["full"],
        feature_names(
            main_categories,
            main_encoder,
            "full",
        ),
        output_dir / "full_model_coefficients.csv",
    )
    save_coefficients(
        primary_models["no_team"],
        feature_names(
            no_team_categories,
            no_team_encoder,
            "full",
        ),
        output_dir / "no_team_model_coefficients.csv",
    )
    save_coefficients(
        strict_full_model,
        feature_names(
            main_categories,
            main_encoder,
            "full",
        ),
        output_dir / "strict_full_model_coefficients.csv",
    )

    bundle = {
        "primary_models": primary_models,
        "strict_full_model": strict_full_model,
        "primary_calibrators": primary_calibrators,
        "strict_calibrator": strict_calibrator,
        "main_encoder": main_encoder,
        "no_team_encoder": no_team_encoder,
        "main_categories": main_categories,
        "no_team_categories": no_team_categories,
        "main_selection": main_selection,
        "no_team_selection": no_team_selection,
        "split_manifest": str(split_manifest),
        "target": "success=1; failure=0",
        "main_filter": "primary_open_play == True",
        "strict_filter": "strict_open_play == True",
    }
    joblib.dump(
        bundle,
        output_dir / "xpass_final_model_bundle.joblib",
        compress=3,
    )

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    with (
        output_dir / "software_environment.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            environment,
            file,
            ensure_ascii=False,
            indent=2,
        )

    main_full_metrics = (
        primary_metrics.loc[
            primary_metrics["model"].eq(
                "full_calibrated"
            )
        ]
        .iloc[0]
        .to_dict()
    )
    no_team_metrics = (
        primary_metrics.loc[
            primary_metrics["model"].eq(
                "no_team_calibrated"
            )
        ]
        .iloc[0]
        .to_dict()
    )
    strict_full_metrics = (
        strict_metrics.loc[
            strict_metrics["model"].eq(
                "strict_full_calibrated"
            )
        ]
        .iloc[0]
        .to_dict()
    )

    methodological_audit = {
        "locked_split_unit": "match_id",
        "training_and_tuning_combined_for_refit": True,
        "calibration_data_used_only_for_platt_calibration": True,
        "test_data_used_only_after_all_model_choices_locked": True,
        "test_set_evaluated_in_this_run": True,
        "majority_baseline_included": True,
        "constant_probability_baseline_included": True,
        "geometry_only_model_included": True,
        "no_team_identity_sensitivity_included": True,
        "strict_open_play_retrained_sensitivity_included": True,
        "male_female_subgroup_metrics_included": True,
        "match_cluster_bootstrap_95ci_included": True,
        "post_outcome_predictors_excluded": [
            "pass_outcome",
            "pass_aerial_won",
            "pass_miscommunication",
            "pass_recovery",
            "pass_recipient",
            "related_events",
            "pass_shot_assist",
            "pass_goal_assist",
            "pass_deflected",
        ],
    }
    with (
        output_dir / "methodological_audit.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            methodological_audit,
            file,
            ensure_ascii=False,
            indent=2,
        )

    record = {
        "status": "PASS",
        "test_set_evaluated_once": True,
        "input_csv": str(input_csv),
        "split_manifest": str(split_manifest),
        "main_tuning_json": str(main_tuning_json),
        "no_team_tuning_json": str(
            no_team_tuning_json
        ),
        "input_csv_sha256": sha256_file(input_csv),
        "split_manifest_sha256": sha256_file(
            split_manifest
        ),
        "main_tuning_sha256": sha256_file(
            main_tuning_json
        ),
        "no_team_tuning_sha256": sha256_file(
            no_team_tuning_json
        ),
        "primary_training_plus_tuning": {
            "rows": primary_train_rows,
            "successes": primary_train_successes,
            "success_prevalence": primary_prevalence,
        },
        "strict_training_plus_tuning": {
            "rows": strict_train_rows,
            "successes": strict_train_successes,
            "success_prevalence": strict_prevalence,
        },
        "primary_calibration_rows": len(
            y_cal_primary
        ),
        "strict_calibration_rows": len(
            y_cal_strict
        ),
        "primary_test_rows": len(
            y_test_primary
        ),
        "strict_test_rows": len(
            y_test_strict
        ),
        "main_full_calibrated_metrics":
            json_native(main_full_metrics),
        "no_team_calibrated_metrics":
            json_native(no_team_metrics),
        "strict_full_calibrated_metrics":
            json_native(strict_full_metrics),
        "bootstrap_reps": args.bootstrap_reps,
        "model_bundle": str(
            output_dir / "xpass_final_model_bundle.joblib"
        ),
        "primary_test_predictions": str(
            output_dir / "final_test_event_predictions.csv"
        ),
        "strict_test_predictions": str(
            output_dir
            / "strict_open_play_test_event_predictions.csv"
        ),
    }

    with final_record.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_native(record),
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 100)
    print("Locked xPass evaluation, calibration, and sensitivity analyses completed")
    print("=" * 100)
    print(f"Primary test rows: {len(y_test_primary):,}")
    print(f"Strict test rows: {len(y_test_strict):,}")
    print(
        "Full calibrated ROC-AUC: "
        f"{main_full_metrics['roc_auc_success']:.6f}"
    )
    print(
        "Full calibrated failure PR-AUC: "
        f"{main_full_metrics['pr_auc_failure']:.6f}"
    )
    print(
        "Full calibrated log loss: "
        f"{main_full_metrics['log_loss']:.6f}"
    )
    print(
        "Full calibrated Brier score: "
        f"{main_full_metrics['brier_score']:.6f}"
    )
    print(
        "Full calibrated calibration slope: "
        f"{main_full_metrics['calibration_slope']:.6f}"
    )
    print(
        "No-team calibrated log loss: "
        f"{no_team_metrics['log_loss']:.6f}"
    )
    print(
        "Strict full calibrated log loss: "
        f"{strict_full_metrics['log_loss']:.6f}"
    )
    print("Test set read: Yes, only after all choices were locked")
    print("Status: PASS")
    print(f"Output directory: {output_dir}")

# -----------------------------------------------------------------------------
# Match-level OOF cross-fitting
# -----------------------------------------------------------------------------
def stable_hash(seed: int, label: str, match_id: str) -> str:
    payload = f"{seed}|{label}|{match_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def create_fold_roles(
    manifest_path: Path,
    outer_folds: int,
    calibration_fraction: float,
    seed: int,
    expected_matches: int,
) -> pd.DataFrame:
    frame = pd.read_csv(
        manifest_path,
        encoding="utf-8-sig",
        dtype={"match_id": "string"},
        low_memory=False,
    )

    required = {"match_id", "stratum", "stable_hash"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"Split manifest missing columns required for cross-fitting: {missing}"
        )

    frame["match_id"] = frame["match_id"].map(normalize_id)
    if len(frame) != expected_matches:
        raise RuntimeError(
            f"Expected {expected_matches:,} matches, found {len(frame):,}."
        )
    if frame["match_id"].duplicated().any():
        raise RuntimeError("Duplicate match_id values in split manifest.")

    frame["outer_fold"] = -1
    for _, group in frame.groupby("stratum", sort=True):
        ordered = group.sort_values(
            ["stable_hash", "match_id"],
            kind="stable",
        )
        frame.loc[ordered.index, "outer_fold"] = (
            np.arange(len(ordered), dtype=int) % outer_folds
        )

    if (frame["outer_fold"] < 0).any():
        raise RuntimeError("Some matches lack an outer-fold assignment.")

    role_rows: list[dict[str, Any]] = []
    for outer_fold in range(outer_folds):
        holdout_ids = set(
            frame.loc[frame["outer_fold"].eq(outer_fold), "match_id"]
        )
        development = frame.loc[
            ~frame["match_id"].isin(holdout_ids)
        ].copy()
        calibration_ids: set[str] = set()

        for stratum, group in development.groupby("stratum", sort=True):
            ordered = group.copy()
            ordered["_calibration_hash"] = ordered["match_id"].map(
                lambda match_id: stable_hash(
                    seed,
                    f"calibration-fold-{outer_fold}",
                    match_id,
                )
            )
            ordered = ordered.sort_values(
                ["_calibration_hash", "match_id"],
                kind="stable",
            )
            n_calibration = int(round(len(ordered) * calibration_fraction))
            if len(ordered) >= 2:
                n_calibration = max(1, n_calibration)
                n_calibration = min(n_calibration, len(ordered) - 1)
            else:
                n_calibration = 0
            calibration_ids.update(
                ordered.head(n_calibration)["match_id"]
            )

        for _, row in frame.iterrows():
            match_id = row["match_id"]
            if match_id in holdout_ids:
                role = "holdout"
            elif match_id in calibration_ids:
                role = "calibration"
            else:
                role = "training"
            role_rows.append(
                {
                    "outer_fold": outer_fold,
                    "match_id": match_id,
                    "role": role,
                    "stratum": row["stratum"],
                    "original_locked_split": row.get("split", ""),
                }
            )

    roles = pd.DataFrame(role_rows)
    exactly_one = (
        roles.groupby(["outer_fold", "match_id"]).size().eq(1).all()
    )
    if not bool(exactly_one):
        raise RuntimeError(
            "Fold-role manifest does not assign exactly one role per match per fold."
        )

    holdout_counts = (
        roles.loc[roles["role"].eq("holdout")]
        .groupby("match_id")
        .size()
    )
    if not (
        len(holdout_counts) == expected_matches
        and holdout_counts.eq(1).all()
    ):
        raise RuntimeError("Each match must serve as holdout exactly once.")

    return roles


def role_map_for_fold(
    roles: pd.DataFrame,
    outer_fold: int,
) -> dict[str, str]:
    current = roles.loc[roles["outer_fold"].eq(outer_fold)]
    return dict(zip(current["match_id"], current["role"]))


def read_role_chunks(
    input_csv: Path,
    role_map: dict[str, str],
    allowed_roles: set[str],
    chunksize: int,
    shuffle_seed: int | None = None,
) -> Iterable[pd.DataFrame]:
    header = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()
    missing = sorted(set(RAW_COLUMNS) - set(header))
    if missing:
        raise RuntimeError(f"Input CSV missing required columns: {missing}")

    reader = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        usecols=RAW_COLUMNS,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        match_ids = chunk["match_id"].map(normalize_id)
        role = match_ids.map(role_map)
        if role.isna().any():
            missing_ids = sorted(set(match_ids.loc[role.isna()]))
            raise RuntimeError(
                "Input contains match IDs missing from fold-role map: "
                + ", ".join(missing_ids[:20])
            )

        mask = (
            is_true_flag(chunk["primary_open_play"])
            & role.isin(allowed_roles)
        )
        selected = chunk.loc[mask].copy()
        if selected.empty:
            continue

        selected["_role"] = role.loc[mask].to_numpy()
        selected["match_id"] = selected["match_id"].map(normalize_id)
        if shuffle_seed is not None:
            selected = selected.sample(
                frac=1.0,
                random_state=shuffle_seed + chunk_number,
            )
        yield selected.reset_index(drop=True)


def collect_fold_categories(
    input_csv: Path,
    role_map: dict[str, str],
    chunksize: int,
    main_features: list[str],
    no_team_features: list[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]], int]:
    all_features = sorted(set(main_features) | set(no_team_features))
    categories = {feature: set() for feature in all_features}
    rows = 0

    for chunk in read_role_chunks(
        input_csv,
        role_map,
        {"training"},
        chunksize,
    ):
        rows += len(chunk)
        normalized = normalize_category_frame(chunk, all_features)
        for feature in all_features:
            categories[feature].update(
                normalized[feature].unique().tolist()
            )

    if rows == 0:
        raise RuntimeError("No training rows found in outer fold.")

    main_map = {
        feature: sorted(categories[feature])
        for feature in main_features
    }
    no_team_map = {
        feature: sorted(categories[feature])
        for feature in no_team_features
    }
    return main_map, no_team_map, rows


def fit_fold_models(
    input_csv: Path,
    role_map: dict[str, str],
    chunksize: int,
    main_spec: dict[str, Any],
    no_team_spec: dict[str, Any],
    main_encoder: OneHotEncoder,
    no_team_encoder: OneHotEncoder,
    seed: int,
) -> tuple[SGDClassifier, SGDClassifier]:
    main_model = make_classifier(main_spec["config"], seed + 1)
    no_team_model = make_classifier(no_team_spec["config"], seed + 2)
    max_epochs = max(
        int(main_spec["epochs"]),
        int(no_team_spec["epochs"]),
    )
    classes = np.array([0, 1], dtype=np.int8)

    for epoch in range(1, max_epochs + 1):
        rows = 0
        for chunk in read_role_chunks(
            input_csv,
            role_map,
            {"training"},
            chunksize,
            shuffle_seed=seed + epoch * 10_000,
        ):
            y = target_from_frame(chunk)
            rows += len(y)
            if epoch <= int(main_spec["epochs"]):
                main_model.partial_fit(
                    build_matrix(
                        chunk,
                        "full",
                        main_encoder,
                        main_spec["categorical_features"],
                    ),
                    y,
                    classes=classes,
                )
            if epoch <= int(no_team_spec["epochs"]):
                no_team_model.partial_fit(
                    build_matrix(
                        chunk,
                        "full",
                        no_team_encoder,
                        no_team_spec["categorical_features"],
                    ),
                    y,
                    classes=classes,
                )
        print(f"[OOF training] epoch={epoch}/{max_epochs}, rows={rows:,}")

    return main_model, no_team_model


def collect_calibration_predictions(
    input_csv: Path,
    role_map: dict[str, str],
    chunksize: int,
    main_model: SGDClassifier,
    no_team_model: SGDClassifier,
    main_encoder: OneHotEncoder,
    no_team_encoder: OneHotEncoder,
    main_features: list[str],
    no_team_features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_parts: list[np.ndarray] = []
    main_parts: list[np.ndarray] = []
    no_team_parts: list[np.ndarray] = []

    for chunk in read_role_chunks(
        input_csv,
        role_map,
        {"calibration"},
        chunksize,
    ):
        y_parts.append(target_from_frame(chunk))
        main_parts.append(
            safe_predict(
                main_model,
                build_matrix(chunk, "full", main_encoder, main_features),
            )
        )
        no_team_parts.append(
            safe_predict(
                no_team_model,
                build_matrix(chunk, "full", no_team_encoder, no_team_features),
            )
        )

    if not y_parts:
        raise RuntimeError("No calibration rows found.")

    return (
        np.concatenate(y_parts),
        np.concatenate(main_parts),
        np.concatenate(no_team_parts),
    )


def write_holdout_predictions(
    input_csv: Path,
    role_map: dict[str, str],
    chunksize: int,
    outer_fold: int,
    main_model: SGDClassifier,
    no_team_model: SGDClassifier,
    main_encoder: OneHotEncoder,
    no_team_encoder: OneHotEncoder,
    main_features: list[str],
    no_team_features: list[str],
    main_calibrator: LogisticRegression,
    no_team_calibrator: LogisticRegression,
    fold_output_csv: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[str], int]:
    header_written = False
    y_parts: list[np.ndarray] = []
    main_parts: list[np.ndarray] = []
    no_team_parts: list[np.ndarray] = []
    matches: set[str] = set()
    rows = 0

    for chunk in read_role_chunks(
        input_csv,
        role_map,
        {"holdout"},
        chunksize,
    ):
        y = target_from_frame(chunk)
        main_probability = apply_platt(
            main_calibrator,
            safe_predict(
                main_model,
                build_matrix(chunk, "full", main_encoder, main_features),
            ),
        )
        no_team_probability = apply_platt(
            no_team_calibrator,
            safe_predict(
                no_team_model,
                build_matrix(chunk, "full", no_team_encoder, no_team_features),
            ),
        )

        output = pd.DataFrame(
            {
                "id": chunk["id"].astype("string"),
                "match_id": chunk["match_id"].astype("string"),
                "outer_fold": outer_fold,
                "xpass_success": y,
                "strict_open_play": is_true_flag(
                    chunk["strict_open_play"]
                ).to_numpy(),
                "p_success_main_oof": main_probability,
                "p_failure_main_oof": 1.0 - main_probability,
                "p_success_no_team_oof": no_team_probability,
                "p_failure_no_team_oof": 1.0 - no_team_probability,
            }
        )
        output.to_csv(
            fold_output_csv,
            mode="a",
            header=not header_written,
            index=False,
            encoding="utf-8-sig",
        )
        header_written = True
        y_parts.append(y)
        main_parts.append(main_probability)
        no_team_parts.append(no_team_probability)
        matches.update(output["match_id"].astype(str).unique())
        rows += len(output)

    if not header_written:
        raise RuntimeError(
            f"No holdout predictions generated for fold {outer_fold}."
        )

    return (
        np.concatenate(y_parts),
        np.concatenate(main_parts),
        np.concatenate(no_team_parts),
        matches,
        rows,
    )


def oof_metric_row(
    cohort: str,
    model: str,
    y_success: np.ndarray,
    p_success: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(y_success, dtype=np.int8)
    p = np.clip(np.asarray(p_success, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    y_failure = 1 - y
    return {
        "cohort": cohort,
        "model": model,
        "rows": int(len(y)),
        "success_prevalence": float(y.mean()),
        "failure_prevalence": float(y_failure.mean()),
        "roc_auc_success": float(roc_auc_score(y, p)),
        "pr_auc_failure": float(
            average_precision_score(y_failure, 1.0 - p)
        ),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
    }


def run_crossfit(args: argparse.Namespace) -> None:
    input_csv = args.input_csv.resolve()
    split_manifest = args.split_manifest.resolve()
    main_tuning_json = args.main_tuning_json.resolve()
    no_team_tuning_json = args.no_team_tuning_json.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in [
        input_csv,
        split_manifest,
        main_tuning_json,
        no_team_tuning_json,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    if args.outer_folds < 2:
        raise ValueError("--outer_folds must be at least 2")
    if not 0.02 <= args.calibration_fraction_of_development <= 0.30:
        raise ValueError(
            "--calibration_fraction_of_development must be between 0.02 and 0.30"
        )

    final_oof_csv = output_dir / "oof_xpass_predictions.csv"
    summary_json = output_dir / "oof_generation_summary.json"
    if (final_oof_csv.exists() or summary_json.exists()) and not args.overwrite:
        raise FileExistsError(
            "OOF outputs already exist. Use --overwrite only after reviewing "
            "the existing results."
        )

    if args.overwrite:
        for path in output_dir.iterdir():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

    fold_dir = output_dir / "fold_predictions"
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "fold_models"
    if args.save_fold_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    main_selection = load_tuning_json(main_tuning_json)
    no_team_selection = load_tuning_json(no_team_tuning_json)
    main_features = list(main_selection["categorical_features"])
    no_team_features = list(no_team_selection["categorical_features"])

    if not {"team", "possession_team"}.issubset(set(main_features)):
        raise RuntimeError(
            "Main tuning file does not include team and possession_team."
        )
    if {"team", "possession_team"} & set(no_team_features):
        raise RuntimeError(
            "No-team tuning file still includes team identity features."
        )

    main_spec = {
        "config": main_selection["full_best"]["config"],
        "epochs": int(main_selection["full_best"]["epoch"]),
        "categorical_features": main_features,
    }
    no_team_spec = {
        "config": no_team_selection["full_best"]["config"],
        "epochs": int(no_team_selection["full_best"]["epoch"]),
        "categorical_features": no_team_features,
    }

    roles = create_fold_roles(
        split_manifest,
        args.outer_folds,
        args.calibration_fraction_of_development,
        args.seed,
        args.expected_matches,
    )
    roles.to_csv(
        output_dir / "oof_fold_role_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    role_summary = (
        roles.groupby(["outer_fold", "role"])
        .agg(matches=("match_id", "nunique"))
        .reset_index()
    )
    role_summary.to_csv(
        output_dir / "oof_fold_role_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_metric_rows: list[dict[str, Any]] = []
    overall_y_parts: list[np.ndarray] = []
    overall_main_parts: list[np.ndarray] = []
    overall_no_team_parts: list[np.ndarray] = []
    predicted_matches: set[str] = set()
    total_predicted_rows = 0

    for outer_fold in range(args.outer_folds):
        print("=" * 100)
        print(f"Starting xPass OOF fold {outer_fold + 1}/{args.outer_folds}")
        print("=" * 100)
        role_map = role_map_for_fold(roles, outer_fold)
        main_category_map, no_team_category_map, training_rows = (
            collect_fold_categories(
                input_csv,
                role_map,
                args.chunksize,
                main_features,
                no_team_features,
            )
        )
        main_encoder = make_encoder(main_category_map, main_features)
        no_team_encoder = make_encoder(no_team_category_map, no_team_features)
        main_model, no_team_model = fit_fold_models(
            input_csv,
            role_map,
            args.chunksize,
            main_spec,
            no_team_spec,
            main_encoder,
            no_team_encoder,
            args.seed + outer_fold * 100_000,
        )
        y_cal, main_cal_raw, no_team_cal_raw = (
            collect_calibration_predictions(
                input_csv,
                role_map,
                args.chunksize,
                main_model,
                no_team_model,
                main_encoder,
                no_team_encoder,
                main_features,
                no_team_features,
            )
        )
        main_calibrator = fit_platt(y_cal, main_cal_raw)
        no_team_calibrator = fit_platt(y_cal, no_team_cal_raw)

        fold_output_csv = fold_dir / f"oof_predictions_fold_{outer_fold}.csv"
        if fold_output_csv.exists():
            fold_output_csv.unlink()
        y_holdout, main_holdout, no_team_holdout, holdout_matches, holdout_rows = (
            write_holdout_predictions(
                input_csv,
                role_map,
                args.chunksize,
                outer_fold,
                main_model,
                no_team_model,
                main_encoder,
                no_team_encoder,
                main_features,
                no_team_features,
                main_calibrator,
                no_team_calibrator,
                fold_output_csv,
            )
        )

        main_metrics = oof_metric_row(
            f"fold_{outer_fold}",
            "main_oof",
            y_holdout,
            main_holdout,
        )
        no_team_metrics = oof_metric_row(
            f"fold_{outer_fold}",
            "no_team_oof",
            y_holdout,
            no_team_holdout,
        )
        fold_metric_rows.append(
            {
                "outer_fold": outer_fold,
                "training_rows": training_rows,
                "calibration_rows": int(len(y_cal)),
                "holdout_rows": int(holdout_rows),
                "holdout_matches": int(len(holdout_matches)),
                **{
                    f"main_{key}": value
                    for key, value in main_metrics.items()
                    if key not in {"cohort", "model", "rows"}
                },
                **{
                    f"no_team_{key}": value
                    for key, value in no_team_metrics.items()
                    if key not in {"cohort", "model", "rows"}
                },
            }
        )

        overlap = predicted_matches & holdout_matches
        if overlap:
            raise RuntimeError(
                "Some matches received OOF predictions in multiple folds: "
                + ", ".join(sorted(overlap)[:20])
            )
        predicted_matches.update(holdout_matches)
        total_predicted_rows += holdout_rows
        overall_y_parts.append(y_holdout)
        overall_main_parts.append(main_holdout)
        overall_no_team_parts.append(no_team_holdout)

        if args.save_fold_models:
            joblib.dump(
                {
                    "outer_fold": outer_fold,
                    "main_model": main_model,
                    "no_team_model": no_team_model,
                    "main_encoder": main_encoder,
                    "no_team_encoder": no_team_encoder,
                    "main_calibrator": main_calibrator,
                    "no_team_calibrator": no_team_calibrator,
                    "main_category_map": main_category_map,
                    "no_team_category_map": no_team_category_map,
                    "main_spec": main_spec,
                    "no_team_spec": no_team_spec,
                },
                model_dir / f"oof_model_bundle_fold_{outer_fold}.joblib",
                compress=3,
            )
        print(
            f"Fold {outer_fold}: holdout matches={len(holdout_matches):,}, "
            f"rows={holdout_rows:,}"
        )

    with final_oof_csv.open("wb") as output_file:
        for outer_fold in range(args.outer_folds):
            fold_file = fold_dir / f"oof_predictions_fold_{outer_fold}.csv"
            with fold_file.open("rb") as input_file:
                if outer_fold == 0:
                    shutil.copyfileobj(input_file, output_file)
                else:
                    start = input_file.read(3)
                    if start != b"\xef\xbb\xbf":
                        input_file.seek(0)
                    input_file.readline()
                    shutil.copyfileobj(input_file, output_file)

    y_all = np.concatenate(overall_y_parts)
    main_all = np.concatenate(overall_main_parts)
    no_team_all = np.concatenate(overall_no_team_parts)
    overall_metrics = pd.DataFrame(
        [
            oof_metric_row(
                "all_primary_open_play_oof",
                "main_oof_calibrated",
                y_all,
                main_all,
            ),
            oof_metric_row(
                "all_primary_open_play_oof",
                "no_team_oof_calibrated",
                y_all,
                no_team_all,
            ),
        ]
    )
    overall_metrics.to_csv(
        output_dir / "oof_overall_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(fold_metric_rows).to_csv(
        output_dir / "oof_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    checks = {
        "predicted_match_count_matches_expected": (
            len(predicted_matches) == args.expected_matches
        ),
        "predicted_row_count_matches_expected": (
            total_predicted_rows == args.expected_primary_rows
        ),
        "concatenated_array_rows_match_expected": (
            len(y_all) == args.expected_primary_rows
        ),
        "main_probability_rows_match": len(main_all) == len(y_all),
        "no_team_probability_rows_match": len(no_team_all) == len(y_all),
        "main_probabilities_in_unit_interval": bool(
            np.isfinite(main_all).all()
            and (main_all > 0.0).all()
            and (main_all < 1.0).all()
        ),
        "no_team_probabilities_in_unit_interval": bool(
            np.isfinite(no_team_all).all()
            and (no_team_all > 0.0).all()
            and (no_team_all < 1.0).all()
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    with (output_dir / "software_environment.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(environment, file, ensure_ascii=False, indent=2)

    summary = {
        "status": status,
        "purpose": (
            "Match-level cross-fitted xPass probabilities for downstream "
            "PCTV, player, team, and risk-reward analyses."
        ),
        "performance_reporting_policy": (
            "The locked independent test analysis remains the primary "
            "model-performance result. OOF metrics are supplementary."
        ),
        "input_csv": str(input_csv),
        "split_manifest": str(split_manifest),
        "main_tuning_json": str(main_tuning_json),
        "no_team_tuning_json": str(no_team_tuning_json),
        "outer_folds": args.outer_folds,
        "calibration_fraction_of_development": (
            args.calibration_fraction_of_development
        ),
        "expected_matches": args.expected_matches,
        "predicted_matches": len(predicted_matches),
        "expected_primary_rows": args.expected_primary_rows,
        "predicted_primary_rows": total_predicted_rows,
        "main_locked_specification": main_spec,
        "no_team_locked_specification": no_team_spec,
        "checks": checks,
        "hashes": {
            "input_csv_sha256": sha256_file(input_csv),
            "split_manifest_sha256": sha256_file(split_manifest),
            "main_tuning_sha256": sha256_file(main_tuning_json),
            "no_team_tuning_sha256": sha256_file(no_team_tuning_json),
            "oof_prediction_csv_sha256": sha256_file(final_oof_csv),
            "fold_role_manifest_sha256": sha256_file(
                output_dir / "oof_fold_role_manifest.csv"
            ),
        },
        "output_files": {
            "predictions": str(final_oof_csv),
            "fold_role_manifest": str(
                output_dir / "oof_fold_role_manifest.csv"
            ),
            "fold_metrics": str(output_dir / "oof_fold_metrics.csv"),
            "overall_metrics": str(output_dir / "oof_overall_metrics.csv"),
        },
    }
    with summary_json.open("w", encoding="utf-8") as file:
        json.dump(json_native(summary), file, ensure_ascii=False, indent=2)

    main_metric = overall_metrics.loc[
        overall_metrics["model"].eq("main_oof_calibrated")
    ].iloc[0]
    no_team_metric = overall_metrics.loc[
        overall_metrics["model"].eq("no_team_oof_calibrated")
    ].iloc[0]
    print("=" * 100)
    print("Full-sample match-level OOF xPass generation completed")
    print("=" * 100)
    print(
        f"Predicted matches: {len(predicted_matches):,}/"
        f"{args.expected_matches:,}"
    )
    print(
        f"Predicted primary passes: {total_predicted_rows:,}/"
        f"{args.expected_primary_rows:,}"
    )
    print(f"Main OOF ROC-AUC: {main_metric['roc_auc_success']:.6f}")
    print(
        "Main OOF failure PR-AUC: "
        f"{main_metric['pr_auc_failure']:.6f}"
    )
    print(f"Main OOF log loss: {main_metric['log_loss']:.6f}")
    print(
        "No-team OOF log loss: "
        f"{no_team_metric['log_loss']:.6f}"
    )
    print(f"Status: {status}")
    print(f"OOF predictions: {final_oof_csv}")
    print(f"Audit summary: {summary_json}")
    if status != "PASS":
        raise SystemExit(2)


def run_verify(args: argparse.Namespace) -> None:
    oof_dir = args.oof_dir.resolve()
    summary_path = oof_dir / "oof_generation_summary.json"
    prediction_path = oof_dir / "oof_xpass_predictions.csv"
    overall_path = oof_dir / "oof_overall_metrics.csv"
    fold_path = oof_dir / "oof_fold_metrics.csv"
    role_path = oof_dir / "oof_fold_role_manifest.csv"

    for path in [
        summary_path,
        prediction_path,
        overall_path,
        fold_path,
        role_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    roles = pd.read_csv(
        role_path,
        encoding="utf-8-sig",
        dtype={"match_id": "string"},
        low_memory=False,
    )
    header = pd.read_csv(
        prediction_path,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()
    required_columns = {
        "id",
        "match_id",
        "outer_fold",
        "xpass_success",
        "strict_open_play",
        "p_success_main_oof",
        "p_failure_main_oof",
        "p_success_no_team_oof",
        "p_failure_no_team_oof",
    }
    missing_columns = sorted(required_columns - set(header))

    checks = {
        "summary_status_pass": summary.get("status") == "PASS",
        "summary_match_count": (
            int(summary.get("predicted_matches", -1))
            == args.expected_matches
        ),
        "summary_row_count": (
            int(summary.get("predicted_primary_rows", -1))
            == args.expected_primary_rows
        ),
        "prediction_columns_complete": not missing_columns,
        "each_match_holdout_once": bool(
            roles.loc[roles["role"].eq("holdout")]
            .groupby("match_id")
            .size()
            .eq(1)
            .all()
        ),
        "prediction_hash_matches_summary": (
            summary.get("hashes", {}).get("oof_prediction_csv_sha256")
            == sha256_file(prediction_path)
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "oof_dir": str(oof_dir),
        "checks": checks,
        "missing_prediction_columns": missing_columns,
        "prediction_sha256": sha256_file(prediction_path),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status != "PASS":
        raise SystemExit(2)


def main() -> None:
    args = parse_args()
    if args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "crossfit":
        run_crossfit(args)
    elif args.command == "verify":
        run_verify(args)
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
