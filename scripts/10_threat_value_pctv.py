#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Match-level cross-fitted pass-chain threat and pass-value pipeline.

Workflow
---------------
inventory
    Confirm whether the supplied directory contains pass-only tables or full
    StatsBomb event streams. This is an optional provenance check.

build
    Join the finalized match-level OOF xPass predictions to the pass table,
    construct a pass-chain shot-creation target, fit a match-level cross-fitted
    threat model, and calculate pass-level success value, failure cost,
    expected net value, realized value, and execution residual.

Outputs from ``build``
-----------------------------
- pass_value_oof.csv.gz by default; Parquet can be requested explicitly
- pass_value_summary.json

The main threat target is pass-based rather than full-event xT: whether the
current or next H pass events in the same match-possession contain a StatsBomb
shot-assist or goal-assist pass. Team and player identity are not model inputs.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


THREAT_FEATURE_NAMES = [
    "x",
    "y",
    "x_squared",
    "x_cubed",
    "lateral_distance",
    "lateral_distance_squared",
    "centrality",
    "x_by_centrality",
    "distance_to_goal",
    "goal_opening_angle",
    "minute_normalized",
    "period_2",
    "period_3_plus",
    "log_pass_stage",
]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-fitted pass-chain threat and pass-value pipeline."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser(
        "inventory",
        help="Inventory local JSON/CSV/Parquet data sources.",
    )
    inventory.add_argument("--data_root", type=Path, required=True)
    inventory.add_argument("--output_dir", type=Path, required=True)
    inventory.add_argument("--sample_files_per_type", type=int, default=20)
    inventory.add_argument("--overwrite", action="store_true")

    build = commands.add_parser(
        "build",
        help="Build OOF pass-chain threat and risk-reward outputs.",
    )
    build.add_argument("--pass_csv", type=Path, required=True)
    build.add_argument("--oof_csv", type=Path, required=True)
    build.add_argument("--output_dir", type=Path, required=True)
    build.add_argument(
        "--horizon",
        type=int,
        default=3,
        help=(
            "Number of pass events, including the current pass, used to define "
            "near-term shot creation."
        ),
    )
    build.add_argument("--calibration_fraction", type=float, default=0.10)
    build.add_argument("--sgd_epochs", type=int, default=5)
    build.add_argument("--alpha", type=float, default=1e-5)
    build.add_argument("--batch_size", type=int, default=200_000)
    build.add_argument("--random_state", type=int, default=42)
    build.add_argument(
        "--output_format",
        choices=("csv.gz", "parquet"),
        default="csv.gz",
        help=(
            "Pass-value output format. CSV.GZ is the default because all "
            "downstream reporting scripts read it directly."
        ),
    )
    build.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Common utilities
# -----------------------------------------------------------------------------
def normalize_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except Exception:
            pass
    return text


def to_bool_array(series: pd.Series) -> np.ndarray:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "1.0", "yes", "y", "t"])
        .to_numpy(dtype=bool)
    )


def infer_match_id_from_path(path: Path) -> str:
    name = path.name
    for suffix in (".json.gz", ".csv.gz", ".parquet", ".json", ".csv"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    if re.fullmatch(r"\d+", name):
        return name
    matches = re.findall(r"(?<!\d)(\d{4,})(?!\d)", name)
    return matches[-1] if matches else ""


def read_json(path: Path) -> Any:
    if path.name.lower().endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8-sig") as file:
            return json.load(file)
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def iter_index_batches(indices: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


# -----------------------------------------------------------------------------
# Optional source inventory
# -----------------------------------------------------------------------------
def classify_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "classification": "unreadable_or_unknown",
        "top_level_type": "",
        "sample_keys": [],
        "match_id_from_path": infer_match_id_from_path(path),
    }
    try:
        obj = read_json(path)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["top_level_type"] = type(obj).__name__
    if isinstance(obj, dict) and isinstance(obj.get("events"), list):
        obj = obj["events"]

    if isinstance(obj, list):
        sample = next((item for item in obj if isinstance(item, dict)), None)
        if sample is None:
            result["classification"] = "json_list"
            return result
        result["sample_keys"] = sorted(sample.keys())[:50]
        if {"id", "index", "type"}.issubset(sample.keys()):
            result["classification"] = "statsbomb_event_list"
        else:
            result["classification"] = "json_list_of_dicts"
    elif isinstance(obj, dict):
        result["sample_keys"] = sorted(obj.keys())[:50]
        result["classification"] = "json_object"
    else:
        result["classification"] = "json_scalar"
    return result


def classify_table(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "classification": "unreadable_or_unknown",
        "columns": [],
        "match_id_from_path": infer_match_id_from_path(path),
    }
    try:
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path).head(0)
        else:
            compression = "gzip" if path.name.lower().endswith(".csv.gz") else None
            frame = pd.read_csv(
                path,
                encoding="utf-8-sig",
                compression=compression,
                nrows=0,
            )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    columns = [str(column) for column in frame.columns]
    lower = {column.strip().lower() for column in columns}
    result["columns"] = columns[:120]

    event_like = (
        ("id" in lower or "event_id" in lower)
        and ("index" in lower or "event_index" in lower)
        and ("type" in lower or "type_name" in lower or "event_type" in lower)
        and ("match_id" in lower or bool(infer_match_id_from_path(path)))
    )
    pass_like = (
        ("pass_end_location" in lower or "pass.end_location" in lower)
        and not any(column.startswith("carry") for column in lower)
        and "shot_outcome" not in lower
        and "shot.outcome" not in lower
    )

    if event_like:
        result["classification"] = "event_table"
    elif pass_like:
        result["classification"] = "pass_only_table"
    else:
        result["classification"] = "other_table"
    return result


def run_inventory(args: argparse.Namespace) -> None:
    root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        raise FileNotFoundError(root)

    output_path = output_dir / "source_inventory.json"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    if output_path.exists():
        output_path.unlink()

    all_files = [path for path in root.rglob("*") if path.is_file()]
    candidates = [
        path
        for path in all_files
        if path.name.lower().endswith(
            (".json", ".json.gz", ".csv", ".csv.gz", ".parquet")
        )
    ]

    extension_counts: Counter[str] = Counter()
    classification_counts: Counter[str] = Counter()
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for path in all_files:
        lower = path.name.lower()
        if lower.endswith(".json.gz"):
            extension = ".json.gz"
        elif lower.endswith(".csv.gz"):
            extension = ".csv.gz"
        else:
            extension = path.suffix.lower() or "<no_extension>"
        extension_counts[extension] += 1

    for path in candidates:
        lower = path.name.lower()
        info = classify_json(path) if lower.endswith((".json", ".json.gz")) else classify_table(path)
        kind = str(info["classification"])
        classification_counts[kind] += 1
        if len(samples[kind]) < args.sample_files_per_type:
            samples[kind].append({"path": str(path), "size_bytes": path.stat().st_size, **info})

    event_files = (
        classification_counts.get("statsbomb_event_list", 0)
        + classification_counts.get("event_table", 0)
    )
    pass_files = classification_counts.get("pass_only_table", 0)
    diagnosis = (
        "FULL_EVENT_FILES_FOUND"
        if event_files
        else "PASS_ONLY_DATA_FOUND"
        if pass_files
        else "NO_RECOGNIZABLE_STATSBOMB_DATA"
    )

    summary = {
        "status": "PASS",
        "diagnosis": diagnosis,
        "data_root": str(root),
        "counts": {
            "all_files": len(all_files),
            "candidate_files": len(candidates),
            "event_like_files": event_files,
            "pass_only_files": pass_files,
        },
        "extension_counts": dict(extension_counts.most_common()),
        "classification_counts": dict(classification_counts.most_common()),
        "sample_files_by_classification": dict(samples),
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print("Local data-source inventory completed")
    print("=" * 100)
    print(f"All files: {len(all_files):,}")
    print(f"Full event files: {event_files:,}")
    print(f"Pass-only tables: {pass_files:,}")
    print(f"Diagnosis: {diagnosis}")
    print(f"Inventory summary: {output_path}")


# -----------------------------------------------------------------------------
# Pass-chain labels and state features
# -----------------------------------------------------------------------------
def parse_xy_series(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = np.full(len(series), np.nan, dtype=np.float32)
    y = np.full(len(series), np.nan, dtype=np.float32)
    values = series.to_numpy(dtype=object)

    for i, value in enumerate(values):
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                x[i], y[i] = float(value[0]), float(value[1])
            except Exception:
                pass
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                parts = stripped[1:-1].split(",")
                if len(parts) >= 2:
                    try:
                        x[i], y[i] = float(parts[0]), float(parts[1])
                    except Exception:
                        pass
    return x, y


def forward_horizon_label(
    event: np.ndarray,
    group_code: np.ndarray,
    horizon: int,
) -> np.ndarray:
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    event = event.astype(np.int8, copy=False)
    label = event.copy()
    n = len(event)
    for offset in range(1, horizon):
        if offset >= n:
            break
        same_group = group_code[:-offset] == group_code[offset:]
        label[:-offset] = np.maximum(
            label[:-offset],
            event[offset:] * same_group.astype(np.int8),
        )
    return label.astype(np.int8)


def make_state_features(
    x: np.ndarray,
    y: np.ndarray,
    minute_norm: np.ndarray,
    period: np.ndarray,
    pass_stage: np.ndarray,
) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    y = np.clip(np.asarray(y, dtype=np.float32), 0.0, 1.0)
    minute_norm = np.clip(np.asarray(minute_norm, dtype=np.float32), 0.0, 1.5)
    period = np.asarray(period, dtype=np.float32)
    pass_stage = np.asarray(pass_stage, dtype=np.float32)

    lateral = np.abs(y - 0.5) * 2.0
    centrality = 1.0 - lateral
    distance_to_goal = np.sqrt(
        (1.0 - x) ** 2 + ((y - 0.5) * (80.0 / 120.0)) ** 2
    )

    goal_low = 0.5 - (7.32 / 80.0) / 2.0
    goal_high = 0.5 + (7.32 / 80.0) / 2.0
    dx = np.maximum(1.0 - x, 1e-5)
    goal_angle = np.abs(
        np.arctan2(goal_high - y, dx) - np.arctan2(goal_low - y, dx)
    )
    stage_log = np.log1p(np.clip(pass_stage, 1.0, 100.0)) / np.log(11.0)

    return np.column_stack(
        [
            x,
            y,
            x * x,
            x * x * x,
            lateral,
            lateral * lateral,
            centrality,
            x * centrality,
            distance_to_goal,
            goal_angle,
            minute_norm,
            (period == 2).astype(np.float32),
            (period >= 3).astype(np.float32),
            stage_log,
        ]
    ).astype(np.float32)


def calibrated_probability(intercept: float, slope: float, score: np.ndarray) -> np.ndarray:
    z = np.clip(intercept + slope * score, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


# -----------------------------------------------------------------------------
# Match-level cross-fitting
# -----------------------------------------------------------------------------
def crossfit_threat(
    *,
    label: np.ndarray,
    start_x: np.ndarray,
    start_y: np.ndarray,
    end_x: np.ndarray,
    end_y: np.ndarray,
    minute_norm: np.ndarray,
    period: np.ndarray,
    pass_stage: np.ndarray,
    folds: np.ndarray,
    matches: np.ndarray,
    calibration_fraction: float,
    sgd_epochs: int,
    alpha: float,
    batch_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    n = len(label)
    threat_start = np.full(n, np.nan, dtype=np.float32)
    threat_end = np.full(n, np.nan, dtype=np.float32)
    threat_opp_end = np.full(n, np.nan, dtype=np.float32)

    valid_start = np.isfinite(start_x) & np.isfinite(start_y)
    valid_end = np.isfinite(end_x) & np.isfinite(end_y)
    all_idx = np.arange(n, dtype=np.int64)
    fold_summaries: list[dict[str, Any]] = []

    for fold_position, fold in enumerate(sorted(pd.unique(folds)), start=1):
        held_idx = all_idx[(folds == fold) & valid_start & valid_end]
        train_idx = all_idx[(folds != fold) & valid_start]
        train_groups = matches[train_idx]

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=calibration_fraction,
            random_state=random_state + fold_position,
        )
        fit_pos, cal_pos = next(
            splitter.split(train_idx.reshape(-1, 1), groups=train_groups)
        )
        fit_idx = train_idx[fit_pos]
        cal_idx = train_idx[cal_pos]

        scaler = StandardScaler()
        for idx in iter_index_batches(fit_idx, batch_size):
            scaler.partial_fit(
                make_state_features(
                    start_x[idx],
                    start_y[idx],
                    minute_norm[idx],
                    period[idx],
                    pass_stage[idx],
                )
            )

        classifier = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            fit_intercept=True,
            learning_rate="optimal",
            average=True,
            random_state=random_state + fold_position,
        )
        rng = np.random.default_rng(random_state + fold_position)
        first_update = True

        for _ in range(sgd_epochs):
            epoch_idx = rng.permutation(fit_idx)
            for idx in iter_index_batches(epoch_idx, batch_size):
                features = scaler.transform(
                    make_state_features(
                        start_x[idx],
                        start_y[idx],
                        minute_norm[idx],
                        period[idx],
                        pass_stage[idx],
                    )
                )
                if first_update:
                    classifier.partial_fit(
                        features,
                        label[idx],
                        classes=np.array([0, 1], dtype=np.int8),
                    )
                    first_update = False
                else:
                    classifier.partial_fit(features, label[idx])

        calibration_scores: list[np.ndarray] = []
        calibration_labels: list[np.ndarray] = []
        for idx in iter_index_batches(cal_idx, batch_size):
            features = scaler.transform(
                make_state_features(
                    start_x[idx],
                    start_y[idx],
                    minute_norm[idx],
                    period[idx],
                    pass_stage[idx],
                )
            )
            calibration_scores.append(
                np.asarray(classifier.decision_function(features), dtype=np.float64)
            )
            calibration_labels.append(label[idx])

        cal_score = np.concatenate(calibration_scores)
        cal_label = np.concatenate(calibration_labels)
        if np.unique(cal_label).size != 2:
            raise RuntimeError(f"Calibration subset for fold {fold} has one class.")

        calibrator = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=200,
            random_state=random_state + fold_position,
        )
        calibrator.fit(cal_score.reshape(-1, 1), cal_label)
        cal_intercept = float(calibrator.intercept_[0])
        cal_slope = float(calibrator.coef_[0, 0])

        for idx in iter_index_batches(held_idx, batch_size):
            start_features = scaler.transform(
                make_state_features(
                    start_x[idx],
                    start_y[idx],
                    minute_norm[idx],
                    period[idx],
                    pass_stage[idx],
                )
            )
            threat_start[idx] = calibrated_probability(
                cal_intercept,
                cal_slope,
                classifier.decision_function(start_features),
            ).astype(np.float32)

            end_features = scaler.transform(
                make_state_features(
                    end_x[idx],
                    end_y[idx],
                    minute_norm[idx],
                    period[idx],
                    pass_stage[idx] + 1.0,
                )
            )
            threat_end[idx] = calibrated_probability(
                cal_intercept,
                cal_slope,
                classifier.decision_function(end_features),
            ).astype(np.float32)

            opponent_features = scaler.transform(
                make_state_features(
                    1.0 - end_x[idx],
                    1.0 - end_y[idx],
                    minute_norm[idx],
                    period[idx],
                    np.ones(len(idx), dtype=np.float32),
                )
            )
            threat_opp_end[idx] = calibrated_probability(
                cal_intercept,
                cal_slope,
                classifier.decision_function(opponent_features),
            ).astype(np.float32)

        held_label = label[held_idx]
        held_pred = threat_start[held_idx]
        metrics = {
            "fold": int(fold) if isinstance(fold, (int, np.integer)) else str(fold),
            "fit_rows": int(len(fit_idx)),
            "calibration_rows": int(len(cal_idx)),
            "held_out_rows": int(len(held_idx)),
            "positive_rate": float(held_label.mean()),
            "roc_auc": float(roc_auc_score(held_label, held_pred)),
            "pr_auc": float(average_precision_score(held_label, held_pred)),
            "log_loss": float(log_loss(held_label, held_pred, labels=[0, 1])),
            "brier_score": float(brier_score_loss(held_label, held_pred)),
            "platt_map_intercept": cal_intercept,
            "platt_map_slope": cal_slope,
            "feature_names": THREAT_FEATURE_NAMES,
            "scaler_mean": scaler.mean_.astype(float).tolist(),
            "scaler_scale": scaler.scale_.astype(float).tolist(),
            "model_intercept": float(np.ravel(classifier.intercept_)[0]),
            "model_coefficients": np.ravel(classifier.coef_).astype(float).tolist(),
        }
        fold_summaries.append(metrics)
        print(
            f"[PCTV] fold={fold} fit={len(fit_idx):,} cal={len(cal_idx):,} "
            f"held={len(held_idx):,} logloss={metrics['log_loss']:.6f} "
            f"brier={metrics['brier_score']:.6f}"
        )

    return threat_start, threat_end, threat_opp_end, fold_summaries


# -----------------------------------------------------------------------------
# Build command
# -----------------------------------------------------------------------------
def run_build(args: argparse.Namespace) -> None:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    pass_csv = args.pass_csv.resolve()
    oof_csv = args.oof_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (pass_csv, oof_csv):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.horizon < 1:
        raise ValueError("--horizon must be at least 1")
    if not 0.02 <= args.calibration_fraction <= 0.30:
        raise ValueError("--calibration_fraction must be between 0.02 and 0.30")
    if args.sgd_epochs < 1:
        raise ValueError("--sgd_epochs must be at least 1")

    summary_path = output_dir / "pass_value_summary.json"
    parquet_path = output_dir / "pass_value_oof.parquet"
    csv_path = output_dir / "pass_value_oof.csv.gz"
    existing = [path for path in (summary_path, parquet_path, csv_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Formal outputs already exist. Use --overwrite after review:\n"
            + "\n".join(str(path) for path in existing)
        )
    for path in existing:
        path.unlink()

    pass_header = pd.read_csv(pass_csv, encoding="utf-8-sig", nrows=0).columns.tolist()
    oof_header = pd.read_csv(oof_csv, encoding="utf-8-sig", nrows=0).columns.tolist()

    required_pass = {
        "id",
        "match_id",
        "possession",
        "index",
        "period",
        "minute",
        "second",
        "location",
        "pass_end_location",
        "pass_outcome",
        "pass_shot_assist",
        "pass_goal_assist",
        "primary_open_play",
        "strict_open_play",
    }
    required_oof = {
        "id",
        "match_id",
        "outer_fold",
        "xpass_success",
        "p_success_main_oof",
        "p_success_no_team_oof",
    }
    missing_pass = sorted(required_pass - set(pass_header))
    missing_oof = sorted(required_oof - set(oof_header))
    if missing_pass:
        raise RuntimeError("Pass file is missing required fields: " + ", ".join(missing_pass))
    if missing_oof:
        raise RuntimeError("OOF xPass file is missing required fields: " + ", ".join(missing_oof))

    pass_columns = [
        column
        for column in [
            "id",
            "match_id",
            "competition_id",
            "season_id",
            "competition_name",
            "season_name",
            "competition_gender",
            "competition_country",
            "match_date",
            "home_team",
            "away_team",
            "possession",
            "index",
            "period",
            "minute",
            "second",
            "team",
            "player",
            "location",
            "pass_end_location",
            "pass_outcome",
            "pass_shot_assist",
            "pass_goal_assist",
            "primary_open_play",
            "strict_open_play",
        ]
        if column in pass_header
    ]
    oof_columns = [
        "id",
        "match_id",
        "outer_fold",
        "xpass_success",
        "p_success_main_oof",
        "p_success_no_team_oof",
    ]

    print(f"[build] loading pass metadata: {pass_csv}")
    passes = pd.read_csv(
        pass_csv,
        encoding="utf-8-sig",
        usecols=pass_columns,
        low_memory=False,
    )
    print(f"[build] loading finalized OOF xPass: {oof_csv}")
    oof = pd.read_csv(
        oof_csv,
        encoding="utf-8-sig",
        usecols=oof_columns,
        low_memory=False,
    )

    for frame in (passes, oof):
        frame["id"] = frame["id"].astype("string").str.strip()
        frame["match_id"] = frame["match_id"].map(normalize_id)

    duplicate_pass = int(passes.duplicated(["id", "match_id"], keep=False).sum())
    duplicate_oof = int(oof.duplicated(["id", "match_id"], keep=False).sum())
    if duplicate_pass or duplicate_oof:
        raise RuntimeError(
            "Event-match keys must be unique. "
            f"pass duplicates={duplicate_pass}, OOF duplicates={duplicate_oof}"
        )

    frame = passes.merge(
        oof,
        on=["id", "match_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(oof):
        raise RuntimeError(
            "Not every OOF row matched pass metadata. "
            f"OOF={len(oof):,}, matched={len(frame):,}"
        )

    for column, default in {
        "team": "",
        "player": "",
    }.items():
        if column not in frame.columns:
            frame[column] = default

    frame = frame.sort_values(
        ["match_id", "possession", "index", "minute", "second"],
        kind="stable",
    ).reset_index(drop=True)

    start_x_raw, start_y_raw = parse_xy_series(frame["location"])
    end_x_raw, end_y_raw = parse_xy_series(frame["pass_end_location"])
    start_x = np.clip(start_x_raw / 120.0, 0.0, 1.0).astype(np.float32)
    start_y = np.clip(start_y_raw / 80.0, 0.0, 1.0).astype(np.float32)
    end_x = np.clip(end_x_raw / 120.0, 0.0, 1.0).astype(np.float32)
    end_y = np.clip(end_y_raw / 80.0, 0.0, 1.0).astype(np.float32)

    minute = pd.to_numeric(frame["minute"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    second = pd.to_numeric(frame["second"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    minute_norm = np.clip((minute + second / 60.0) / 90.0, 0.0, 1.5).astype(np.float32)
    period = pd.to_numeric(frame["period"], errors="coerce").fillna(1).to_numpy(np.float32)

    possession_key = (
        frame["match_id"].astype("string")
        + "|"
        + frame["possession"].astype("string").fillna("UNK")
    )
    group_code, _ = pd.factorize(possession_key, sort=False)
    pass_stage = (
        frame.groupby(["match_id", "possession"], sort=False, dropna=False)
        .cumcount()
        .to_numpy(np.float32)
        + 1.0
    )

    shot_creation_event = to_bool_array(
        frame["pass_shot_assist"]
    ).copy()

    shot_creation_event = np.logical_or(
        shot_creation_event,
        to_bool_array(frame["pass_goal_assist"]),
    )
    chain_label = forward_horizon_label(
        shot_creation_event.astype(np.int8),
        group_code.astype(np.int64),
        args.horizon,
    )

    folds = pd.to_numeric(frame["outer_fold"], errors="raise").to_numpy()
    matches = frame["match_id"].astype("string").to_numpy()
    fold_per_match = frame.groupby("match_id")["outer_fold"].nunique()
    if (fold_per_match != 1).any():
        raise RuntimeError("outer_fold is not constant within every match")

    threat_start, threat_end, threat_opp_end, fold_summaries = crossfit_threat(
        label=chain_label,
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y,
        minute_norm=minute_norm,
        period=period,
        pass_stage=pass_stage,
        folds=folds,
        matches=matches,
        calibration_fraction=args.calibration_fraction,
        sgd_epochs=args.sgd_epochs,
        alpha=args.alpha,
        batch_size=args.batch_size,
        random_state=args.random_state,
    )

    valid = np.isfinite(threat_start) & np.isfinite(threat_end) & np.isfinite(threat_opp_end)
    if not valid.all():
        raise RuntimeError(f"Threat predictions are missing for {(~valid).sum():,} rows")

    xpass = pd.to_numeric(frame["p_success_main_oof"], errors="coerce").to_numpy(np.float64)
    xpass_no_team = pd.to_numeric(frame["p_success_no_team_oof"], errors="coerce").to_numpy(np.float64)
    actual = pd.to_numeric(frame["xpass_success"], errors="coerce").to_numpy(np.float64)
    if np.isnan(xpass).any() or np.isnan(xpass_no_team).any() or np.isnan(actual).any():
        raise RuntimeError("OOF xPass probabilities or labels contain missing values")

    success_value = (threat_end - threat_start).astype(np.float32)
    failure_cost_start = threat_start.astype(np.float32)
    failure_cost_opponent = threat_opp_end.astype(np.float32)
    failure_cost = (failure_cost_start + failure_cost_opponent).astype(np.float32)

    expected_success_value = (xpass * success_value).astype(np.float32)
    expected_failure_cost = ((1.0 - xpass) * failure_cost).astype(np.float32)
    expected_net_value = (expected_success_value - expected_failure_cost).astype(np.float32)

    expected_success_value_no_team = (xpass_no_team * success_value).astype(np.float32)
    expected_failure_cost_no_team = ((1.0 - xpass_no_team) * failure_cost).astype(np.float32)
    expected_net_value_no_team = (
        expected_success_value_no_team - expected_failure_cost_no_team
    ).astype(np.float32)

    realized_value = (
        actual * success_value - (1.0 - actual) * failure_cost
    ).astype(np.float32)
    execution_residual = (actual - xpass).astype(np.float32)
    execution_residual_no_team = (actual - xpass_no_team).astype(np.float32)

    output_data: dict[str, Any] = {}
    passthrough_columns = [
        "id",
        "match_id",
        "competition_id",
        "season_id",
        "competition_name",
        "season_name",
        "competition_gender",
        "competition_country",
        "match_date",
        "home_team",
        "away_team",
        "outer_fold",
        "team",
        "player",
        "possession",
        "index",
    ]
    for column in passthrough_columns:
        if column in frame.columns:
            output_data[column] = frame[column]

    output_data.update(
        {
            "period": period.astype(np.int8),
            "minute": minute,
            "second": second,
            "x_start": start_x,
            "y_start": start_y,
            "x_end": end_x,
            "y_end": end_y,
            "pass_stage_in_possession": pass_stage.astype(np.int16),
            f"shot_creation_within_{args.horizon}_passes": chain_label,
            "xpass_success": actual.astype(np.int8),
            "p_success_main_oof": xpass.astype(np.float32),
            "p_success_no_team_oof": xpass_no_team.astype(np.float32),
            "threat_start_oof": threat_start,
            "threat_end_oof": threat_end,
            "opponent_threat_end_oof": threat_opp_end,
            "success_value": success_value,
            "failure_cost_start": failure_cost_start,
            "failure_cost_opponent": failure_cost_opponent,
            "failure_cost": failure_cost,
            "expected_success_value": expected_success_value,
            "expected_failure_cost": expected_failure_cost,
            "expected_net_value": expected_net_value,
            "expected_success_value_no_team": expected_success_value_no_team,
            "expected_failure_cost_no_team": expected_failure_cost_no_team,
            "expected_net_value_no_team": expected_net_value_no_team,
            "realized_value": realized_value,
            "execution_residual": execution_residual,
            "execution_residual_no_team": execution_residual_no_team,
            "primary_open_play": to_bool_array(frame["primary_open_play"]),
            "strict_open_play": to_bool_array(frame["strict_open_play"]),
        }
    )
    output = pd.DataFrame(output_data)

    output_file: Path
    output_format = args.output_format
    if output_format == "parquet":
        try:
            output.to_parquet(parquet_path, index=False)
        except Exception as exc:
            raise RuntimeError(
                "Parquet output was requested but could not be written. "
                "Install a compatible Parquet engine or use "
                "--output_format csv.gz."
            ) from exc
        output_file = parquet_path
    else:
        output.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
            compression={
                "method": "gzip",
                "compresslevel": 6,
                "mtime": 0,
            },
        )
        output_file = csv_path

    global_metrics = {
        "rows": int(len(output)),
        "matches": int(output["match_id"].nunique()),
        "positive_rate": float(chain_label.mean()),
        "roc_auc": float(roc_auc_score(chain_label, threat_start)),
        "pr_auc": float(average_precision_score(chain_label, threat_start)),
        "log_loss": float(log_loss(chain_label, threat_start, labels=[0, 1])),
        "brier_score": float(brier_score_loss(chain_label, threat_start)),
    }

    checks = {
        "all_OOF_rows_matched": len(frame) == len(oof),
        "event_match_keys_unique": duplicate_pass == 0 and duplicate_oof == 0,
        "outer_fold_constant_within_match": bool((fold_per_match == 1).all()),
        "all_threat_predictions_present": bool(valid.all()),
        "xpass_probabilities_valid": bool(
            ((xpass > 0) & (xpass < 1) & (xpass_no_team > 0) & (xpass_no_team < 1)).all()
        ),
        "output_written": output_file.exists(),
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    summary = {
        "status": status,
        "module": "02_threat_value",
        "method_name": "match-level cross-fitted pass-chain threat value",
        "primary_definition": {
            "horizon_passes_including_current": args.horizon,
            "threat_target": (
                "Whether the current pass or the next H - 1 pass events within the same "
                "match-possession contain a StatsBomb shot-assist or goal-assist pass."
            ),
            "success_value": "threat_end_oof - threat_start_oof",
            "failure_cost": "threat_start_oof + opponent_threat_end_oof",
            "expected_success_value": "p_success_main_oof * success_value",
            "expected_failure_cost": "(1 - p_success_main_oof) * failure_cost",
            "expected_net_value": "expected_success_value - expected_failure_cost",
            "realized_value": (
                "observed_success * success_value - (1 - observed_success) * failure_cost"
            ),
            "execution_residual": "observed_success - p_success_main_oof",
        },
        "model": {
            "family": "SGD logistic regression with nonlinear spatial state features",
            "cross_fitting_unit": "match",
            "fold_source": "outer_fold from finalized OOF xPass",
            "calibration_fraction_within_training_matches": args.calibration_fraction,
            "sgd_epochs": args.sgd_epochs,
            "alpha": args.alpha,
            "random_state": args.random_state,
            "feature_names": THREAT_FEATURE_NAMES,
            "team_or_player_identity_used": False,
            "pass_destination_used_for_model_training": False,
            "counterfactual_evaluation": (
                "The same held-out state-value model is evaluated at the pass start, "
                "the intended successful endpoint, and the mirrored opponent endpoint."
            ),
            "pass_stage_treatment": {
                "start_state": "observed pass stage within the possession",
                "successful_endpoint": "observed pass stage plus one",
                "opponent_endpoint": "reset to the first stage of a potential opponent possession",
            },
        },
        "global_performance": global_metrics,
        "fold_performance_and_parameters": fold_summaries,
        "counts": {
            "rows": int(len(output)),
            "matches": int(output["match_id"].nunique()),
            "positive_rows": int(chain_label.sum()),
        },
        "checks": checks,
        "outputs": {
            "pass_value_file": str(output_file),
            "pass_value_format": output_format,
            "summary_file": str(summary_path),
        },
        "sensitivity_columns_already_exported": [
            "p_success_no_team_oof",
            "expected_net_value_no_team",
            "failure_cost_start",
            "failure_cost_opponent",
            "strict_open_play",
        ],
        "limitations": [
            (
                "This is a pass-chain shot-creation model rather than full-event xT, "
                "because the source used for this analysis contains pass events only."
            ),
            (
                "The opponent transition component mirrors the intended endpoint; "
                "alternative failure-cost definitions should be evaluated in the robustness analysis."
            ),
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print("OOF pass-chain threat and pass-value calculation completed")
    print("=" * 100)
    print(f"Pass records: {len(output):,}")
    print(f"Matches: {output['match_id'].nunique():,}")
    print(f"Shot-creation target prevalence: {global_metrics['positive_rate']:.6f}")
    print(f"PCTV ROC-AUC: {global_metrics['roc_auc']:.6f}")
    print(f"PCTV PR-AUC: {global_metrics['pr_auc']:.6f}")
    print(f"PCTV log loss: {global_metrics['log_loss']:.6f}")
    print(f"PCTV Brier score: {global_metrics['brier_score']:.6f}")
    print(f"Status: {status}")
    print(f"Pass-value output: {output_file}")
    print(f"Model and definition summary: {summary_path}")

    if status != "PASS":
        raise SystemExit(2)


def main() -> None:
    args = parse_args()
    if args.command == "inventory":
        run_inventory(args)
    elif args.command == "build":
        run_build(args)
    else:
        raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
