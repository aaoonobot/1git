#!/usr/bin/env python
"""Tune the main and no-team xPass model specifications.

The script uses training matches for fitting and tuning matches for model
selection. Calibration and test matches are never read. Run the script once
for each variant using separate output directories.

Examples
--------
Main model::

    python scripts/05_tune_xpass.py \
        --variant main \
        --input_csv data/processed/passes_preprocessed_with_open_play_flags.csv \
        --split_manifest data/manifests/xpass_match_split_manifest.csv \
        --output_dir outputs/xpass_tuning/main

No-team sensitivity model::

    python scripts/05_tune_xpass.py \
        --variant no-team \
        --input_csv data/processed/passes_preprocessed_with_open_play_flags.csv \
        --split_manifest data/manifests/xpass_match_split_manifest.csv \
        --output_dir outputs/xpass_tuning/no_team
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier
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

BASE_CATEGORICAL_FEATURES = [
    "pass_height",
    "pass_body_part",
    "pass_type",
    "play_pattern",
    "position",
    "pass_technique",
]

IDENTITY_FEATURES = ["team", "possession_team"]

RAW_COLUMNS = [
    "id",
    "match_id",
    "pass_outcome",
    "primary_open_play",
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

DEFAULT_GRID = [
    {"alpha": 1e-5, "l1_ratio": 0.00},
    {"alpha": 1e-5, "l1_ratio": 0.15},
    {"alpha": 1e-5, "l1_ratio": 0.30},
    {"alpha": 3e-5, "l1_ratio": 0.00},
    {"alpha": 3e-5, "l1_ratio": 0.15},
    {"alpha": 3e-5, "l1_ratio": 0.30},
    {"alpha": 1e-4, "l1_ratio": 0.00},
    {"alpha": 1e-4, "l1_ratio": 0.15},
    {"alpha": 1e-4, "l1_ratio": 0.30},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune main and no-team xPass specifications without reading calibration or test matches."
    )
    parser.add_argument("--variant", choices=["main", "no-team"], required=True)
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--split_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=75_000)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--grid_json",
        type=Path,
        default=None,
        help="Optional JSON list containing alpha and l1_ratio values.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_native(item) for key, item in value.items()}
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


def bool_to_float(series: pd.Series) -> np.ndarray:
    text = series.astype("string").fillna("").str.strip().str.lower()
    result = np.zeros(len(series), dtype=np.float32)
    result[text.isin(["true", "1", "1.0", "yes"])] = 1.0
    return result


def categorical_features_for_variant(variant: str) -> list[str]:
    if variant == "main":
        return BASE_CATEGORICAL_FEATURES + IDENTITY_FEATURES
    return list(BASE_CATEGORICAL_FEATURES)


def normalize_category_frame(
    frame: pd.DataFrame,
    categorical_features: list[str],
) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in categorical_features:
        output[column] = (
            frame[column]
            .astype("string")
            .fillna("<MISSING>")
            .str.strip()
            .replace("", "<MISSING>")
        )
    return output


def derive_numeric(frame: pd.DataFrame, full: bool) -> np.ndarray:
    sx = pd.to_numeric(frame["start_x_model"], errors="coerce").to_numpy(np.float32)
    sy = pd.to_numeric(frame["start_y_model"], errors="coerce").to_numpy(np.float32)
    ex = pd.to_numeric(frame["end_x_model"], errors="coerce").to_numpy(np.float32)
    ey = pd.to_numeric(frame["end_y_model"], errors="coerce").to_numpy(np.float32)
    length = (
        pd.to_numeric(frame["pass_length"], errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float32)
    )
    angle = (
        pd.to_numeric(frame["pass_angle"], errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float32)
    )

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
        np.sqrt((120.0 - sx) ** 2 + (40.0 - sy) ** 2) / goal_norm,
        np.sqrt((120.0 - ex) ** 2 + (40.0 - ey) ** 2) / goal_norm,
    ]

    if full:
        minute = (
            pd.to_numeric(frame["minute"], errors="coerce")
            .fillna(0.0)
            .to_numpy(np.float32)
        )
        second = (
            pd.to_numeric(frame["second"], errors="coerce")
            .fillna(0.0)
            .to_numpy(np.float32)
        )
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
    outcome = frame["pass_outcome"].astype("string").fillna("").str.strip()
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
        raise RuntimeError(f"Split manifest missing columns: {missing}")

    mapping: dict[str, str] = {}
    for match_id, split in zip(frame["match_id"], frame["split"]):
        match_text = normalize_id(match_id)
        split_text = str(split).strip()
        if not match_text:
            raise RuntimeError("Empty match_id in split manifest.")
        if match_text in mapping:
            raise RuntimeError(f"Duplicate match_id in split manifest: {match_text}")
        mapping[match_text] = split_text

    required_splits = {"training", "tuning", "calibration", "test"}
    absent = required_splits - set(mapping.values())
    if absent:
        raise RuntimeError(f"Split manifest is missing required splits: {sorted(absent)}")
    return mapping


def read_filtered_chunks(
    input_csv: Path,
    split_map: dict[str, str],
    allowed_splits: set[str],
    chunksize: int,
    shuffle_seed: int | None = None,
) -> Iterable[pd.DataFrame]:
    header = pd.read_csv(input_csv, encoding="utf-8-sig", nrows=0).columns.tolist()
    missing = sorted(set(RAW_COLUMNS) - set(header))
    if missing:
        raise RuntimeError(f"Input CSV is missing required columns: {missing}")

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
            missing_ids = sorted(set(match_ids.loc[split.isna()]))
            raise RuntimeError(
                "Pass rows contain match IDs absent from the split manifest: "
                + ", ".join(missing_ids[:20])
            )

        open_play = (
            chunk["primary_open_play"]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.lower()
            .isin(["true", "1", "1.0"])
        )
        mask = open_play & split.isin(allowed_splits)
        selected = chunk.loc[mask].copy()
        if selected.empty:
            continue
        selected["_split"] = split.loc[mask].to_numpy()
        if shuffle_seed is not None:
            selected = selected.sample(
                frac=1.0,
                random_state=shuffle_seed + chunk_number,
            )
        yield selected.reset_index(drop=True)


def collect_training_categories(
    input_csv: Path,
    split_map: dict[str, str],
    chunksize: int,
    categorical_features: list[str],
) -> dict[str, list[str]]:
    categories = {column: set() for column in categorical_features}
    rows = 0
    for chunk in read_filtered_chunks(
        input_csv,
        split_map,
        {"training"},
        chunksize,
    ):
        normalized = normalize_category_frame(chunk, categorical_features)
        rows += len(normalized)
        for column in categorical_features:
            categories[column].update(normalized[column].unique().tolist())
    if rows == 0:
        raise RuntimeError("No training rows were found.")
    return {column: sorted(values) for column, values in categories.items()}


def make_encoder(
    category_map: dict[str, list[str]],
    categorical_features: list[str],
) -> OneHotEncoder:
    categories = [category_map[column] for column in categorical_features]
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
        {column: [category_map[column][0]] for column in categorical_features}
    )
    encoder.fit(dummy)
    return encoder


def build_matrix(
    frame: pd.DataFrame,
    model_kind: str,
    encoder: OneHotEncoder | None,
    categorical_features: list[str],
) -> sparse.csr_matrix:
    full = model_kind == "full"
    numeric = sparse.csr_matrix(derive_numeric(frame, full=full))
    if not full:
        return numeric
    if encoder is None:
        raise RuntimeError("Full model requires a fitted encoder.")
    categorical = normalize_category_frame(frame, categorical_features)
    encoded = encoder.transform(categorical)
    return sparse.hstack([numeric, encoded], format="csr", dtype=np.float32)


def make_classifier(config: dict[str, float], seed: int) -> SGDClassifier:
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


def safe_probabilities(
    classifier: SGDClassifier,
    matrix: sparse.csr_matrix,
) -> np.ndarray:
    probability = classifier.predict_proba(matrix)[:, 1]
    return np.clip(probability, 1e-7, 1.0 - 1e-7)


def compute_metrics(y_success: np.ndarray, p_success: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_success, dtype=np.int8)
    p = np.clip(np.asarray(p_success, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    y_failure = 1 - y
    p_failure = 1.0 - p
    predicted_success = (p >= 0.5).astype(np.int8)
    predicted_failure = 1 - predicted_success
    return {
        "rows": int(len(y)),
        "success_prevalence": float(y.mean()),
        "failure_prevalence": float(y_failure.mean()),
        "roc_auc_success": float(roc_auc_score(y, p)),
        "pr_auc_failure": float(average_precision_score(y_failure, p_failure)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, predicted_success)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_failure, predicted_failure)
        ),
        "failure_precision": float(
            precision_score(y_failure, predicted_failure, zero_division=0)
        ),
        "failure_recall": float(
            recall_score(y_failure, predicted_failure, zero_division=0)
        ),
        "failure_f1": float(f1_score(y_failure, predicted_failure, zero_division=0)),
    }


def collect_predictions(
    input_csv: Path,
    split_map: dict[str, str],
    allowed_splits: set[str],
    chunksize: int,
    models: dict[str, SGDClassifier],
    model_kind: str,
    encoder: OneHotEncoder | None,
    categorical_features: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    labels: list[np.ndarray] = []
    prediction_parts = {name: [] for name in models}
    for chunk in read_filtered_chunks(
        input_csv,
        split_map,
        allowed_splits,
        chunksize,
    ):
        matrix = build_matrix(
            chunk,
            model_kind,
            encoder,
            categorical_features,
        )
        labels.append(target_from_frame(chunk))
        for name, model in models.items():
            prediction_parts[name].append(safe_probabilities(model, matrix))
    if not labels:
        raise RuntimeError(f"No rows found for splits: {sorted(allowed_splits)}")
    return (
        np.concatenate(labels),
        {name: np.concatenate(parts) for name, parts in prediction_parts.items()},
    )


def train_candidates(
    input_csv: Path,
    split_map: dict[str, str],
    chunksize: int,
    grid: list[dict[str, float]],
    max_epochs: int,
    seed: int,
    model_kind: str,
    encoder: OneHotEncoder | None,
    categorical_features: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidates: dict[str, dict[str, Any]] = {}
    for index, config in enumerate(grid, start=1):
        name = f"{model_kind}_a{config['alpha']}_l1{config['l1_ratio']}"
        candidates[name] = {
            "config": config,
            "model": make_classifier(config, seed + index),
        }

    classes = np.array([0, 1], dtype=np.int8)
    metric_rows: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None

    for epoch in range(1, max_epochs + 1):
        training_rows = 0
        for chunk in read_filtered_chunks(
            input_csv,
            split_map,
            {"training"},
            chunksize,
            shuffle_seed=seed + epoch * 10_000,
        ):
            matrix = build_matrix(
                chunk,
                model_kind,
                encoder,
                categorical_features,
            )
            labels = target_from_frame(chunk)
            training_rows += len(labels)
            for candidate in candidates.values():
                candidate["model"].partial_fit(matrix, labels, classes=classes)

        models = {name: item["model"] for name, item in candidates.items()}
        tuning_labels, predictions = collect_predictions(
            input_csv,
            split_map,
            {"tuning"},
            chunksize,
            models,
            model_kind,
            encoder,
            categorical_features,
        )

        for name, probability in predictions.items():
            metrics = compute_metrics(tuning_labels, probability)
            config = candidates[name]["config"]
            record = {
                "model_kind": model_kind,
                "candidate_name": name,
                "epoch": epoch,
                "training_rows_seen_this_epoch": training_rows,
                "alpha": config["alpha"],
                "l1_ratio": config["l1_ratio"],
                **metrics,
            }
            metric_rows.append(record)
            ranking_key = (
                metrics["log_loss"],
                metrics["brier_score"],
                -metrics["pr_auc_failure"],
            )
            if best_record is None or ranking_key < best_record["ranking_key"]:
                best_record = {
                    "ranking_key": ranking_key,
                    "candidate_name": name,
                    "epoch": epoch,
                    "config": config,
                    "metrics": metrics,
                }

        assert best_record is not None
        print(
            f"[{model_kind}] epoch={epoch}/{max_epochs}, "
            f"training_rows={training_rows:,}, "
            f"best_log_loss={best_record['metrics']['log_loss']:.6f}"
        )

    if best_record is None:
        raise RuntimeError(f"No candidate selected for {model_kind}.")
    best_record.pop("ranking_key")
    return best_record, pd.DataFrame(metric_rows)


def load_grid(path: Path | None) -> list[dict[str, float]]:
    if path is None:
        return list(DEFAULT_GRID)
    with path.resolve().open("r", encoding="utf-8") as file:
        grid = json.load(file)
    if not isinstance(grid, list) or not grid:
        raise RuntimeError("grid_json must contain a non-empty list.")
    return [
        {"alpha": float(item["alpha"]), "l1_ratio": float(item["l1_ratio"])}
        for item in grid
    ]


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    split_manifest = args.split_manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in [input_csv, split_manifest]:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.max_epochs < 1:
        raise ValueError("--max_epochs must be at least 1")

    tuning_json = output_dir / "tuning_selection.json"
    tuning_csv = output_dir / "tuning_candidate_metrics.csv"
    existing = [path for path in [tuning_json, tuning_csv] if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Tuning outputs already exist. Use --overwrite after review:\n"
            + "\n".join(str(path) for path in existing)
        )
    for path in existing:
        path.unlink()

    split_map = load_split_map(split_manifest)
    grid = load_grid(args.grid_json)
    categorical_features = categorical_features_for_variant(args.variant)
    category_map = collect_training_categories(
        input_csv,
        split_map,
        args.chunksize,
        categorical_features,
    )
    encoder = make_encoder(category_map, categorical_features)

    geometry_best, geometry_metrics = train_candidates(
        input_csv,
        split_map,
        args.chunksize,
        grid,
        args.max_epochs,
        args.seed,
        "geometry",
        None,
        categorical_features,
    )
    full_best, full_metrics = train_candidates(
        input_csv,
        split_map,
        args.chunksize,
        grid,
        args.max_epochs,
        args.seed + 1000,
        "full",
        encoder,
        categorical_features,
    )

    pd.concat([geometry_metrics, full_metrics], ignore_index=True).to_csv(
        tuning_csv,
        index=False,
        encoding="utf-8-sig",
    )

    selection: dict[str, Any] = {
        "status": "PASS",
        "model_variant": "main_with_team_identity"
        if args.variant == "main"
        else "no_team_identity",
        "excluded_identity_features": []
        if args.variant == "main"
        else IDENTITY_FEATURES,
        "input_csv": str(args.input_csv),
        "split_manifest": str(args.split_manifest),
        "seed": args.seed,
        "grid": grid,
        "max_epochs": args.max_epochs,
        "geometry_best": geometry_best,
        "full_best": full_best,
        "category_map": category_map,
        "categorical_features": categorical_features,
        "geometry_numeric_features": GEOMETRY_NUMERIC_FEATURES,
        "full_numeric_features": (
            GEOMETRY_NUMERIC_FEATURES + FULL_EXTRA_NUMERIC_FEATURES
        ),
        "selection_metric": (
            "lowest tuning log loss; ties resolved by Brier score and then "
            "failure-class PR-AUC"
        ),
        "test_set_read": False,
    }
    with tuning_json.open("w", encoding="utf-8") as file:
        json.dump(json_native(selection), file, ensure_ascii=False, indent=2)

    print("=" * 100)
    print(f"xPass tuning completed for variant: {args.variant}")
    print("=" * 100)
    print(
        "Geometry best: "
        f"alpha={geometry_best['config']['alpha']}, "
        f"l1_ratio={geometry_best['config']['l1_ratio']}, "
        f"epoch={geometry_best['epoch']}, "
        f"log_loss={geometry_best['metrics']['log_loss']:.6f}"
    )
    print(
        "Full best: "
        f"alpha={full_best['config']['alpha']}, "
        f"l1_ratio={full_best['config']['l1_ratio']}, "
        f"epoch={full_best['epoch']}, "
        f"log_loss={full_best['metrics']['log_loss']:.6f}"
    )
    print("Calibration set read: No")
    print("Test set read: No")
    print("Status: PASS")
    print(f"Selection file: {tuning_json}")


if __name__ == "__main__":
    main()
