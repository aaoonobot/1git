#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
17_full_sample_analysis.py

Revised Module 05: full-sample application of the locked H=3 pass
risk–reward framework.

Narrative role
--------------
This module is the principal empirical application of the model. It asks:

1. How do expected success value, expected failure cost, and expected net
   value vary across the full distribution of pass failure risk?
2. How does StatsBomb's event-level pressure indicator alter the same
   risk–reward components?
3. How do Ground, Low, and High passes differ in their risk–reward structure?

The pressure × pass-height combinations are retained as supplementary
evidence rather than treated as another independent story.

No model is fitted or modified.

Required files
--------------
--pass_value_csv:
    result2/02_pctv_h3/pass_value_oof.csv.gz

--pass_metadata_csv:
    passes_preprocessed_with_open_play_flags.csv
    This is used only when pressure or pass height are not already present
    in the pass-value file.

--style_file:
    publication_style.py must be in the same folder as this script.

Main outputs
------------
Figure_2_full_sample_application.png/.pdf/.svg
Figure_S3_pressure_height.png/.pdf/.svg
full_sample_framework_descriptives.csv
full_sample_risk_bin_summary.csv
full_sample_pressure_summary.csv
full_sample_pass_height_summary.csv
full_sample_pressure_height_summary.csv
full_sample_context_contrasts.csv
module05_results_text.txt
module05_figure_captions.txt
module05_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from publication_style import (
        STYLE,
        add_panel_label,
        annotate_horizontal_estimate,
        clean_axis,
        configure_matplotlib,
        height_color_map,
        save_figure_all_formats,
    )
except ImportError as exc:
    raise ImportError(
        "publication_style.py must be saved in the same folder as "
        "17_full_sample_analysis.py."
    ) from exc


SCRIPT_VERSION = "2026-07-31-final-audit-1"

EXPECTED_MATCHES = 3_926
EXPECTED_ROWS = 3_396_912

PRESSURE_ORDER = ["Not under pressure", "Under pressure"]
HEIGHT_ORDER = ["Ground Pass", "Low Pass", "High Pass"]

COLUMN_CANDIDATES = {
    "match_id": ["match_id", "matchid"],
    "event_id": ["id", "event_id", "event_uuid"],
    "under_pressure": ["under_pressure", "is_under_pressure"],
    "pass_height": ["pass_height", "pass_height_name"],
    "xpass_success": ["xpass_success", "observed_success", "pass_success"],
    "p_success_main_oof": [
        "p_success_main_oof",
        "p_success_full_oof",
        "xpass_probability",
    ],
    "execution_residual": ["execution_residual"],
    "expected_success_value": ["expected_success_value"],
    "expected_failure_cost": ["expected_failure_cost"],
    "expected_net_value": ["expected_net_value"],
    "realized_value": ["realized_value"],
}

FRAMEWORK_METRICS = {
    "completion_rate": {
        "source": "xpass_success",
        "scale": 1.0,
        "label": "Pass-completion rate",
    },
    "mean_xpass": {
        "source": "p_success_main_oof",
        "scale": 1.0,
        "label": "Mean xPass",
    },
    "failure_risk": {
        "source": "failure_risk",
        "scale": 1.0,
        "label": "Failure probability",
    },
    "execution_residual": {
        "source": "execution_residual",
        "scale": 1.0,
        "label": "Execution residual",
    },
    "expected_success_value_per100": {
        "source": "expected_success_value",
        "scale": 100.0,
        "label": "Expected success value",
    },
    "expected_failure_cost_per100": {
        "source": "expected_failure_cost",
        "scale": 100.0,
        "label": "Expected failure cost",
    },
    "expected_net_value_per100": {
        "source": "expected_net_value",
        "scale": 100.0,
        "label": "Expected net value",
    },
    "centered_expected_net_value_per100": {
        "source": "expected_net_value_centered",
        "scale": 100.0,
        "label": "Centered expected net value",
    },
    "realized_value_per100": {
        "source": "realized_value",
        "scale": 100.0,
        "label": "Realized value",
    },
    "positive_net_value_rate": {
        "source": "positive_net_value",
        "scale": 1.0,
        "label": "Positive expected-net-value rate",
    },
}

MAIN_VALUE_METRICS = [
    "expected_success_value_per100",
    "expected_failure_cost_per100",
    "centered_expected_net_value_per100",
]



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
            "Create the revised full-sample risk–reward application and "
            "publication-quality Figure 2."
        )
    )
    parser.add_argument("--pass_value_csv", type=Path, required=True)
    parser.add_argument("--pass_metadata_csv", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--risk_bins", type=int, default=20)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


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


def detect_column(
    columns: Iterable[str],
    candidates: list[str],
    logical_name: str,
    required: bool = True,
) -> str | None:
    columns = list(columns)
    normalized = {normalize_text(column): column for column in columns}

    for candidate in candidates:
        if candidate in columns:
            return candidate
        candidate_key = normalize_text(candidate)
        if candidate_key in normalized:
            return normalized[candidate_key]

    if required:
        raise RuntimeError(
            f"Could not identify column '{logical_name}'. "
            f"Candidates: {candidates}. Available columns: {columns}"
        )
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_file_hashes(
    directory: Path,
    *,
    excluded_names: set[str] | None = None,
) -> dict[str, str]:
    """Return SHA-256 hashes for files already present in a directory."""
    excluded = excluded_names or set()
    return {
        path.name: sha256_file(path)
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name not in excluded
    }


def write_sha256sums(
    directory: Path,
    filename: str = "SHA256SUMS.txt",
) -> Path:
    """
    Write a standard hash manifest for all output files except the manifest
    itself. The audit JSON is therefore included without creating a circular
    self-hash dependency.
    """
    manifest_path = directory / filename
    lines = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != filename:
            lines.append(f"{sha256_file(path)}  {path.name}")
    manifest_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return manifest_path


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


def normalize_pressure(value: object) -> str:
    if pd.isna(value):
        return "Not under pressure"

    if isinstance(value, bool):
        return "Under pressure" if value else "Not under pressure"

    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y", "under pressure"}:
        return "Under pressure"
    if text in {"0", "0.0", "false", "no", "n", "", "nan", "none"}:
        return "Not under pressure"

    raise ValueError(f"Unrecognized under_pressure value: {value!r}")


def normalize_pass_height(value: object) -> str:
    mapping = {
        "groundpass": "Ground Pass",
        "ground": "Ground Pass",
        "lowpass": "Low Pass",
        "low": "Low Pass",
        "highpass": "High Pass",
        "high": "High Pass",
    }
    return mapping.get(normalize_text(value), "Unknown")


def read_csv_in_chunks(
    path: Path,
    usecols: list[str],
    chunksize: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        parts.append(chunk)

    if not parts:
        raise RuntimeError(f"No rows were read from {path}.")

    return pd.concat(parts, ignore_index=True)



def load_analysis_frame(
    pass_value_csv: Path,
    pass_metadata_csv: Path | None,
    chunksize: int,
) -> tuple[
    pd.DataFrame,
    dict[str, str | None],
    dict[str, object],
]:
    header = pd.read_csv(
        pass_value_csv,
        encoding="utf-8-sig",
        nrows=0,
    )
    columns = header.columns.tolist()

    detected: dict[str, str | None] = {}
    for logical_name, candidates in COLUMN_CANDIDATES.items():
        detected[logical_name] = detect_column(
            columns,
            candidates,
            logical_name,
            required=logical_name
            not in {"under_pressure", "pass_height"},
        )

    value_usecols = sorted(
        {
            column
            for column in detected.values()
            if column is not None
        }
    )
    frame = read_csv_in_chunks(
        pass_value_csv,
        value_usecols,
        chunksize,
    )

    merge_audit: dict[str, object] = {
        "pass_value_rows_before_context_merge": int(len(frame)),
        "context_source": "pass_value_file",
        "pass_value_duplicate_event_id_rows": 0,
        "pass_value_duplicate_event_id_count": 0,
        "metadata_rows_loaded": 0,
        "metadata_duplicate_event_id_rows": 0,
        "metadata_duplicate_event_id_count": 0,
        "matched_event_rows": int(len(frame)),
        "unmatched_event_rows": 0,
    }

    missing_context = [
        logical_name
        for logical_name in ["under_pressure", "pass_height"]
        if detected[logical_name] is None
    ]

    if missing_context:
        if pass_metadata_csv is None:
            raise RuntimeError(
                "The pass-value file does not include "
                f"{missing_context}. Supply --pass_metadata_csv."
            )

        metadata_header = pd.read_csv(
            pass_metadata_csv,
            encoding="utf-8-sig",
            nrows=0,
        )
        metadata_columns = metadata_header.columns.tolist()

        metadata_event = detect_column(
            metadata_columns,
            COLUMN_CANDIDATES["event_id"],
            "event_id",
        )
        metadata_pressure = detect_column(
            metadata_columns,
            COLUMN_CANDIDATES["under_pressure"],
            "under_pressure",
        )
        metadata_height = detect_column(
            metadata_columns,
            COLUMN_CANDIDATES["pass_height"],
            "pass_height",
        )

        metadata = read_csv_in_chunks(
            pass_metadata_csv,
            [
                metadata_event,
                metadata_pressure,
                metadata_height,
            ],
            chunksize,
        )

        value_event = detected["event_id"]
        frame["_event_id_norm"] = frame[value_event].map(normalize_id)
        metadata["_event_id_norm"] = metadata[
            metadata_event
        ].map(normalize_id)

        value_duplicate_mask = frame["_event_id_norm"].duplicated(
            keep=False
        )
        metadata_duplicate_mask = metadata[
            "_event_id_norm"
        ].duplicated(keep=False)

        value_duplicate_rows = int(value_duplicate_mask.sum())
        value_duplicate_ids = int(
            frame.loc[
                value_duplicate_mask,
                "_event_id_norm",
            ].nunique()
        )
        metadata_duplicate_rows = int(
            metadata_duplicate_mask.sum()
        )
        metadata_duplicate_ids = int(
            metadata.loc[
                metadata_duplicate_mask,
                "_event_id_norm",
            ].nunique()
        )

        merge_audit.update(
            {
                "context_source": "event_id_metadata_merge",
                "metadata_rows_loaded": int(len(metadata)),
                "pass_value_duplicate_event_id_rows": (
                    value_duplicate_rows
                ),
                "pass_value_duplicate_event_id_count": (
                    value_duplicate_ids
                ),
                "metadata_duplicate_event_id_rows": (
                    metadata_duplicate_rows
                ),
                "metadata_duplicate_event_id_count": (
                    metadata_duplicate_ids
                ),
            }
        )

        if value_duplicate_ids > 0:
            raise RuntimeError(
                "Duplicate event IDs were found in the pass-value input: "
                f"{value_duplicate_ids} duplicated IDs across "
                f"{value_duplicate_rows} rows."
            )
        if metadata_duplicate_ids > 0:
            raise RuntimeError(
                "Duplicate event IDs were found in the metadata input: "
                f"{metadata_duplicate_ids} duplicated IDs across "
                f"{metadata_duplicate_rows} rows."
            )

        metadata = metadata[
            [
                "_event_id_norm",
                metadata_pressure,
                metadata_height,
            ]
        ]

        frame = frame.merge(
            metadata,
            on="_event_id_norm",
            how="left",
            validate="one_to_one",
            indicator=True,
            suffixes=("", "_metadata"),
        )

        matched_rows = int((frame["_merge"] == "both").sum())
        unmatched_rows = int(
            (frame["_merge"] == "left_only").sum()
        )
        merge_audit.update(
            {
                "matched_event_rows": matched_rows,
                "unmatched_event_rows": unmatched_rows,
            }
        )

        if unmatched_rows > 0:
            raise RuntimeError(
                "Some pass-value rows did not match the metadata input: "
                f"{unmatched_rows} unmatched event IDs."
            )

        frame = frame.drop(columns=["_merge"])

        if detected["under_pressure"] is None:
            detected["under_pressure"] = metadata_pressure
        if detected["pass_height"] is None:
            detected["pass_height"] = metadata_height

    rename_map = {
        detected["match_id"]: "match_id",
        detected["event_id"]: "event_id",
        detected["under_pressure"]: "under_pressure_source",
        detected["pass_height"]: "pass_height_source",
        detected["xpass_success"]: "xpass_success",
        detected["p_success_main_oof"]: "p_success_main_oof",
        detected["execution_residual"]: "execution_residual",
        detected["expected_success_value"]: "expected_success_value",
        detected["expected_failure_cost"]: "expected_failure_cost",
        detected["expected_net_value"]: "expected_net_value",
        detected["realized_value"]: "realized_value",
    }
    frame = frame.rename(columns=rename_map)

    numeric_columns = [
        "xpass_success",
        "p_success_main_oof",
        "execution_residual",
        "expected_success_value",
        "expected_failure_cost",
        "expected_net_value",
        "realized_value",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    if frame[numeric_columns].isna().any().any():
        missing = frame[numeric_columns].isna().sum()
        raise RuntimeError(
            "Missing values remain in analysis columns:\n"
            + missing[missing > 0].to_string()
        )

    frame["match_id"] = frame["match_id"].map(normalize_id)
    frame["event_id"] = frame["event_id"].map(normalize_id)
    frame["pressure_status"] = frame[
        "under_pressure_source"
    ].map(normalize_pressure)
    frame["pass_height_group"] = frame[
        "pass_height_source"
    ].map(normalize_pass_height)

    frame["failure_risk"] = (
        1.0 - frame["p_success_main_oof"]
    )
    global_net_mean = float(frame["expected_net_value"].mean())
    frame["expected_net_value_centered"] = (
        frame["expected_net_value"] - global_net_mean
    )
    frame["positive_net_value"] = (
        frame["expected_net_value"] > 0.0
    ).astype(np.int8)

    keep_columns = [
        "event_id",
        "match_id",
        "pressure_status",
        "pass_height_group",
        "xpass_success",
        "p_success_main_oof",
        "failure_risk",
        "execution_residual",
        "expected_success_value",
        "expected_failure_cost",
        "expected_net_value",
        "expected_net_value_centered",
        "realized_value",
        "positive_net_value",
    ]

    merge_audit["rows_after_context_processing"] = int(
        len(frame)
    )
    merge_audit["unknown_pass_height_rows"] = int(
        np.sum(frame["pass_height_group"] == "Unknown")
    )

    return (
        frame[keep_columns].copy(),
        detected,
        merge_audit,
    )

def full_sample_descriptives(frame: pd.DataFrame) -> pd.DataFrame:
    specifications = [
        ("Failure probability", "failure_risk", 1.0),
        (
            "Expected success value per 100 passes",
            "expected_success_value",
            100.0,
        ),
        (
            "Expected failure cost per 100 passes",
            "expected_failure_cost",
            100.0,
        ),
        (
            "Expected net value per 100 passes",
            "expected_net_value",
            100.0,
        ),
        (
            "Realized value per 100 passes",
            "realized_value",
            100.0,
        ),
        ("Execution residual", "execution_residual", 1.0),
    ]

    rows: list[dict[str, object]] = []
    for label, column, scale in specifications:
        values = (
            frame[column].to_numpy(dtype=np.float64)
            * scale
        )
        rows.append(
            {
                "metric": label,
                "n": len(values),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "standard_deviation": float(
                    np.std(values, ddof=1)
                ),
                "p25": float(np.quantile(values, 0.25)),
                "p75": float(np.quantile(values, 0.75)),
                "p95": float(np.quantile(values, 0.95)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        )

    return pd.DataFrame(rows)


def build_match_category_arrays(
    frame: pd.DataFrame,
    category_column: str,
    categories: list[str],
    metric_names: list[str],
) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    match_ids = sorted(frame["match_id"].unique().tolist())
    match_index = {
        match_id: index
        for index, match_id in enumerate(match_ids)
    }
    n_matches = len(match_ids)

    arrays: dict[str, dict[str, np.ndarray]] = {}

    for category in categories:
        selected = frame.loc[
            frame[category_column].eq(category)
        ]
        grouped = selected.groupby("match_id", sort=False)

        count = np.zeros(n_matches, dtype=np.float64)
        for match_id, value in grouped.size().items():
            count[match_index[match_id]] = float(value)

        category_arrays: dict[str, np.ndarray] = {
            "count": count,
        }

        for metric_name in metric_names:
            source = FRAMEWORK_METRICS[metric_name]["source"]
            values = np.zeros(n_matches, dtype=np.float64)
            sums = grouped[source].sum()
            for match_id, value in sums.items():
                values[match_index[match_id]] = float(value)
            category_arrays[metric_name] = values

        arrays[category] = category_arrays

    return match_ids, arrays


def mean_from_arrays(
    category_arrays: dict[str, np.ndarray],
    metric_name: str,
    sampled_matches: np.ndarray | None,
) -> float:
    counts = category_arrays["count"]
    sums = category_arrays[metric_name]

    if sampled_matches is None:
        denominator = counts.sum()
        numerator = sums.sum()
    else:
        denominator = counts[sampled_matches].sum()
        numerator = sums[sampled_matches].sum()

    if denominator <= 0:
        return float("nan")

    scale = float(FRAMEWORK_METRICS[metric_name]["scale"])
    return float(numerator / denominator * scale)


def summarize_categories(
    frame: pd.DataFrame,
    category_column: str,
    categories: list[str],
    metric_names: list[str],
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    match_ids, arrays = build_match_category_arrays(
        frame,
        category_column,
        categories,
        metric_names,
    )
    n_matches = len(match_ids)
    rng = np.random.default_rng(seed)

    bootstrap_store: dict[str, dict[str, list[float]]] = {
        category: {
            "share": [],
            **{metric_name: [] for metric_name in metric_names},
        }
        for category in categories
    }

    for _ in range(bootstrap_reps):
        sampled = rng.integers(
            0,
            n_matches,
            size=n_matches,
        )

        total_sampled_count = sum(
            arrays[category]["count"][sampled].sum()
            for category in categories
        )

        for category in categories:
            sampled_count = arrays[category]["count"][
                sampled
            ].sum()
            bootstrap_store[category]["share"].append(
                sampled_count / total_sampled_count
                if total_sampled_count > 0
                else float("nan")
            )

            for metric_name in metric_names:
                bootstrap_store[category][metric_name].append(
                    mean_from_arrays(
                        arrays[category],
                        metric_name,
                        sampled,
                    )
                )

    total_count = sum(
        arrays[category]["count"].sum()
        for category in categories
    )

    rows: list[dict[str, object]] = []
    for category in categories:
        category_count = int(
            arrays[category]["count"].sum()
        )

        row: dict[str, object] = {
            "category_variable": category_column,
            "category": category,
            "passes": category_count,
            "matches_with_category": int(
                np.sum(arrays[category]["count"] > 0)
            ),
            "total_matches": n_matches,
            "share": category_count / total_count,
            "share_ci_lower": float(
                np.nanquantile(
                    bootstrap_store[category]["share"],
                    0.025,
                )
            ),
            "share_ci_upper": float(
                np.nanquantile(
                    bootstrap_store[category]["share"],
                    0.975,
                )
            ),
        }

        for metric_name in metric_names:
            bootstrap_values = np.asarray(
                bootstrap_store[category][metric_name],
                dtype=np.float64,
            )

            row[metric_name] = mean_from_arrays(
                arrays[category],
                metric_name,
                None,
            )
            row[f"{metric_name}_ci_lower"] = float(
                np.nanquantile(bootstrap_values, 0.025)
            )
            row[f"{metric_name}_ci_upper"] = float(
                np.nanquantile(bootstrap_values, 0.975)
            )

        rows.append(row)

    return pd.DataFrame(rows)


def paired_context_contrasts(
    frame: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    contrast_definitions = [
        {
            "family": "pressure",
            "category_column": "pressure_status",
            "left": "Under pressure",
            "right": "Not under pressure",
            "label": "Under pressure minus not under pressure",
        },
        {
            "family": "pass_height",
            "category_column": "pass_height_group",
            "left": "Low Pass",
            "right": "Ground Pass",
            "label": "Low Pass minus Ground Pass",
        },
        {
            "family": "pass_height",
            "category_column": "pass_height_group",
            "left": "High Pass",
            "right": "Ground Pass",
            "label": "High Pass minus Ground Pass",
        },
    ]

    rows: list[dict[str, object]] = []

    for definition_index, definition in enumerate(
        contrast_definitions
    ):
        match_ids, arrays = build_match_category_arrays(
            frame,
            definition["category_column"],
            [definition["left"], definition["right"]],
            list(FRAMEWORK_METRICS),
        )
        n_matches = len(match_ids)
        rng = np.random.default_rng(
            seed + 1000 * definition_index
        )

        for metric_name in FRAMEWORK_METRICS:
            left_point = mean_from_arrays(
                arrays[definition["left"]],
                metric_name,
                None,
            )
            right_point = mean_from_arrays(
                arrays[definition["right"]],
                metric_name,
                None,
            )
            estimate = left_point - right_point

            bootstrap_values = np.empty(
                bootstrap_reps,
                dtype=np.float64,
            )

            for repetition in range(bootstrap_reps):
                sampled = rng.integers(
                    0,
                    n_matches,
                    size=n_matches,
                )
                left_value = mean_from_arrays(
                    arrays[definition["left"]],
                    metric_name,
                    sampled,
                )
                right_value = mean_from_arrays(
                    arrays[definition["right"]],
                    metric_name,
                    sampled,
                )
                bootstrap_values[repetition] = (
                    left_value - right_value
                )

            lower, upper = np.nanquantile(
                bootstrap_values,
                [0.025, 0.975],
            )

            rows.append(
                {
                    "contrast_family": definition["family"],
                    "contrast": definition["label"],
                    "left_category": definition["left"],
                    "right_category": definition["right"],
                    "metric": metric_name,
                    "metric_label": FRAMEWORK_METRICS[
                        metric_name
                    ]["label"],
                    "estimate": estimate,
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "bootstrap_reps": bootstrap_reps,
                    "bootstrap_unit": "match_id",
                    "ci_excludes_zero": bool(
                        lower > 0.0 or upper < 0.0
                    ),
                }
            )

    return pd.DataFrame(rows)


def risk_bin_summary_with_bootstrap(
    frame: pd.DataFrame,
    n_bins: int,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["risk_bin"] = pd.qcut(
        frame["failure_risk"],
        q=n_bins,
        labels=False,
        duplicates="drop",
    )

    actual_bins = sorted(
        frame["risk_bin"].dropna().astype(int).unique()
    )
    match_ids = sorted(frame["match_id"].unique().tolist())
    match_index = {
        match_id: index
        for index, match_id in enumerate(match_ids)
    }
    bin_index = {
        risk_bin: index
        for index, risk_bin in enumerate(actual_bins)
    }

    n_matches = len(match_ids)
    n_actual_bins = len(actual_bins)

    count_matrix = np.zeros(
        (n_matches, n_actual_bins),
        dtype=np.float64,
    )
    risk_sum_matrix = np.zeros_like(count_matrix)

    metric_names = [
        "expected_success_value_per100",
        "expected_failure_cost_per100",
        "expected_net_value_per100",
    ]
    sum_matrices = {
        metric_name: np.zeros_like(count_matrix)
        for metric_name in metric_names
    }

    grouped = frame.groupby(
        ["match_id", "risk_bin"],
        observed=True,
        sort=False,
    )

    for (match_id, risk_bin), group in grouped:
        match_position = match_index[match_id]
        bin_position = bin_index[int(risk_bin)]

        count_matrix[
            match_position,
            bin_position,
        ] = len(group)
        risk_sum_matrix[
            match_position,
            bin_position,
        ] = float(group["failure_risk"].sum())

        for metric_name in metric_names:
            source = FRAMEWORK_METRICS[
                metric_name
            ]["source"]
            sum_matrices[metric_name][
                match_position,
                bin_position,
            ] = float(group[source].sum())

    total_counts = count_matrix.sum(axis=0)
    mean_risk = (
        risk_sum_matrix.sum(axis=0)
        / total_counts
    )

    point_values: dict[str, np.ndarray] = {}
    for metric_name in metric_names:
        scale = float(
            FRAMEWORK_METRICS[metric_name]["scale"]
        )
        point_values[metric_name] = (
            sum_matrices[metric_name].sum(axis=0)
            / total_counts
            * scale
        )

    rng = np.random.default_rng(seed)
    bootstrap_values = {
        metric_name: np.full(
            (bootstrap_reps, n_actual_bins),
            np.nan,
            dtype=np.float64,
        )
        for metric_name in metric_names
    }

    for repetition in range(bootstrap_reps):
        sampled = rng.integers(
            0,
            n_matches,
            size=n_matches,
        )

        sampled_counts = count_matrix[
            sampled
        ].sum(axis=0)

        for metric_name in metric_names:
            scale = float(
                FRAMEWORK_METRICS[metric_name]["scale"]
            )
            sampled_sums = sum_matrices[
                metric_name
            ][sampled].sum(axis=0)

            bootstrap_values[metric_name][
                repetition
            ] = np.divide(
                sampled_sums,
                sampled_counts,
                out=np.full(
                    n_actual_bins,
                    np.nan,
                    dtype=np.float64,
                ),
                where=sampled_counts > 0,
            ) * scale

    rows: list[dict[str, object]] = []
    for bin_position, risk_bin in enumerate(actual_bins):
        row: dict[str, object] = {
            "risk_bin": int(risk_bin) + 1,
            "passes": int(total_counts[bin_position]),
            "matches": int(
                np.sum(
                    count_matrix[:, bin_position] > 0
                )
            ),
            "mean_failure_risk": float(
                mean_risk[bin_position]
            ),
        }

        for metric_name in metric_names:
            values = bootstrap_values[
                metric_name
            ][:, bin_position]

            row[metric_name] = float(
                point_values[metric_name][bin_position]
            )
            row[f"{metric_name}_ci_lower"] = float(
                np.nanquantile(values, 0.025)
            )
            row[f"{metric_name}_ci_upper"] = float(
                np.nanquantile(values, 0.975)
            )

        rows.append(row)

    return pd.DataFrame(rows)


def pressure_height_summary(
    frame: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    selected = frame.loc[
        frame["pass_height_group"].isin(HEIGHT_ORDER)
    ].copy()

    selected["pressure_height"] = (
        selected["pressure_status"]
        + " | "
        + selected["pass_height_group"]
    )

    categories = [
        f"{pressure} | {height}"
        for pressure in PRESSURE_ORDER
        for height in HEIGHT_ORDER
    ]

    return summarize_categories(
        selected,
        "pressure_height",
        categories,
        list(FRAMEWORK_METRICS),
        bootstrap_reps,
        seed,
    )


def plot_figure_2(
    risk_bins: pd.DataFrame,
    contrasts: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Main-text Figure 2.

    A: Full-sample risk–reward structure.
    B: Under-pressure minus not-under-pressure contrasts.
    C: Low/High minus Ground Pass contrasts.
    """
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(
            STYLE.double_column_width_in,
            STYLE.figure2_height_in,
        ),
        gridspec_kw={
            "width_ratios": [1.48, 0.90, 1.02],
            "wspace": 0.47,
        },
    )

    # ------------------------------------------------------------------
    # Panel A: full-sample risk–reward structure.
    # ------------------------------------------------------------------
    ax = axes[0]

    line_specs = [
        (
            "expected_success_value_per100",
            "Expected success value",
            STYLE.deep_blue,
            "o",
            STYLE.secondary_line_width,
            2.45,
            0.13,
        ),
        (
            "expected_failure_cost_per100",
            "Expected failure cost",
            STYLE.muted_orange,
            "s",
            STYLE.secondary_line_width,
            2.45,
            0.13,
        ),
        (
            "expected_net_value_per100",
            "Expected net value",
            STYLE.dark_slate,
            "^",
            STYLE.main_line_width,
            2.75,
            0.16,
        ),
    ]

    x_values = risk_bins["mean_failure_risk"].to_numpy(dtype=float)

    for (
        metric_name,
        label,
        color,
        marker,
        line_width,
        marker_size,
        ribbon_alpha,
    ) in line_specs:
        estimate = risk_bins[metric_name].to_numpy(dtype=float)
        lower = risk_bins[
            f"{metric_name}_ci_lower"
        ].to_numpy(dtype=float)
        upper = risk_bins[
            f"{metric_name}_ci_upper"
        ].to_numpy(dtype=float)

        ax.fill_between(
            x_values,
            lower,
            upper,
            color=color,
            alpha=ribbon_alpha,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x_values,
            estimate,
            color=color,
            marker=marker,
            markersize=marker_size,
            markeredgewidth=0,
            linewidth=line_width,
            label=label,
            zorder=3,
        )

    ax.axhline(
        0.0,
        color=STYLE.light_gray,
        linewidth=STYLE.reference_line_width,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax.set_xlabel("Failure probability")
    ax.set_ylabel("Value per 100 passes")
    ax.set_title(
        "Full-sample risk–reward structure",
        loc="left",
        pad=5,
    )
    clean_axis(ax, grid_axis="y")
    ax.legend(
        frameon=False,
        loc="upper left",
        handlelength=2.25,
        handletextpad=0.55,
        borderaxespad=0.25,
    )
    add_panel_label(ax, "A", x=-0.17)

    # ------------------------------------------------------------------
    # Shared order for panels B and C.
    # ------------------------------------------------------------------
    metric_order = [
        "expected_success_value_per100",
        "expected_failure_cost_per100",
        "centered_expected_net_value_per100",
    ]
    metric_labels = [
        "Expected\nsuccess value",
        "Expected\nfailure cost",
        "Centered expected\nnet value",
    ]
    y_positions = np.arange(len(metric_order))

    # ------------------------------------------------------------------
    # Panel B: pressure contrast.
    # ------------------------------------------------------------------
    ax = axes[1]

    pressure = (
        contrasts.loc[
            contrasts["contrast_family"].eq("pressure")
            & contrasts["metric"].isin(metric_order)
        ]
        .set_index("metric")
        .loc[metric_order]
        .reset_index()
    )

    estimates = pressure["estimate"].to_numpy(dtype=float)
    lower = pressure["ci_lower"].to_numpy(dtype=float)
    upper = pressure["ci_upper"].to_numpy(dtype=float)

    ax.errorbar(
        estimates,
        y_positions,
        xerr=np.vstack(
            [estimates - lower, upper - estimates]
        ),
        fmt="o",
        color=STYLE.deep_blue,
        ecolor=STYLE.deep_blue,
        markersize=STYLE.marker_size - 0.35,
        markeredgecolor="white",
        markeredgewidth=0.65,
        elinewidth=STYLE.error_line_width,
        capsize=STYLE.cap_size,
        zorder=3,
    )
    ax.axvline(
        0.0,
        color=STYLE.light_gray,
        linewidth=STYLE.reference_line_width,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(metric_labels)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Difference under pressure vs\n"
        "not under pressure (per 100 passes)"
    )
    ax.set_title(
        "Pressure-related change",
        loc="left",
        pad=5,
    )
    clean_axis(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Set limits before adding numeric labels.
    pressure_min = min(float(lower.min()), 0.0)
    pressure_max = max(float(upper.max()), 0.0)
    pressure_span = max(pressure_max - pressure_min, 0.5)
    ax.set_xlim(
        pressure_min - 0.16 * pressure_span,
        pressure_max + 0.28 * pressure_span,
    )

    for estimate, y_position in zip(estimates, y_positions):
        annotate_horizontal_estimate(
            ax,
            float(estimate),
            float(y_position),
            text=f"{estimate:.2f}",
        )

    add_panel_label(ax, "B", x=-0.31)

    # ------------------------------------------------------------------
    # Panel C: pass-height contrasts.
    # ------------------------------------------------------------------
    ax = axes[2]

    height = contrasts.loc[
        contrasts["contrast_family"].eq("pass_height")
        & contrasts["metric"].isin(metric_order)
    ].copy()

    contrast_specs = [
        (
            "Low Pass minus Ground Pass",
            "Low − Ground",
            STYLE.medium_gray,
            "s",
            -0.11,
        ),
        (
            "High Pass minus Ground Pass",
            "High − Ground",
            STYLE.muted_orange,
            "o",
            0.11,
        ),
    ]

    all_height_bounds: list[float] = [0.0]
    annotation_rows: list[
        tuple[np.ndarray, np.ndarray, str]
    ] = []

    for (
        contrast_name,
        label,
        color,
        marker,
        offset,
    ) in contrast_specs:
        selected = (
            height.loc[
                height["contrast"].eq(contrast_name)
            ]
            .set_index("metric")
            .loc[metric_order]
            .reset_index()
        )

        estimates = selected["estimate"].to_numpy(dtype=float)
        lower = selected["ci_lower"].to_numpy(dtype=float)
        upper = selected["ci_upper"].to_numpy(dtype=float)

        all_height_bounds.extend(lower.tolist())
        all_height_bounds.extend(upper.tolist())

        ax.errorbar(
            estimates,
            y_positions + offset,
            xerr=np.vstack(
                [estimates - lower, upper - estimates]
            ),
            fmt=marker,
            color=color,
            ecolor=color,
            markersize=STYLE.marker_size - 0.45,
            markeredgecolor="white",
            markeredgewidth=0.65,
            elinewidth=STYLE.error_line_width,
            capsize=STYLE.cap_size,
            label=label,
            zorder=3,
        )
        annotation_rows.append(
            (
                estimates,
                y_positions + offset,
                color,
            )
        )

    ax.axvline(
        0.0,
        color=STYLE.light_gray,
        linewidth=STYLE.reference_line_width,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(metric_labels)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Difference from Ground Pass\n(per 100 passes)"
    )
    ax.set_title(
        "Pass-height contrasts",
        loc="left",
        pad=27,
    )
    clean_axis(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    height_min = min(all_height_bounds)
    height_max = max(all_height_bounds)
    height_span = max(height_max - height_min, 1.0)
    ax.set_xlim(
        height_min - 0.12 * height_span,
        height_max + 0.26 * height_span,
    )

    ax.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=2,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )

    for estimates, positions, _ in annotation_rows:
        for estimate, y_position in zip(estimates, positions):
            annotate_horizontal_estimate(
                ax,
                float(estimate),
                float(y_position),
                text=f"{estimate:.2f}",
                pad_fraction=0.018,
            )

    add_panel_label(ax, "C", x=-0.29, y=1.16)

    save_figure_all_formats(
        figure,
        output_dir,
        "Figure_2_full_sample_application",
    )
    plt.close(figure)


def plot_pressure_height_supplement(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Supplementary three-row dumbbell plot.

    Each pass-height row directly compares:
    - Not under pressure: open blue point
    - Under pressure: filled orange point
    """
    selected = summary.set_index("category")

    figure, ax = plt.subplots(
        figsize=(
            STYLE.supplementary_width_in,
            STYLE.supplementary_height_in,
        ),
        constrained_layout=True,
    )

    y_positions = np.arange(len(HEIGHT_ORDER))

    plotted_values: list[float] = [0.0]

    for y_position, height in zip(
        y_positions,
        HEIGHT_ORDER,
    ):
        not_key = f"Not under pressure | {height}"
        under_key = f"Under pressure | {height}"

        not_row = selected.loc[not_key]
        under_row = selected.loc[under_key]

        not_estimate = float(
            not_row["centered_expected_net_value_per100"]
        )
        not_lower = float(
            not_row[
                "centered_expected_net_value_per100_ci_lower"
            ]
        )
        not_upper = float(
            not_row[
                "centered_expected_net_value_per100_ci_upper"
            ]
        )

        under_estimate = float(
            under_row["centered_expected_net_value_per100"]
        )
        under_lower = float(
            under_row[
                "centered_expected_net_value_per100_ci_lower"
            ]
        )
        under_upper = float(
            under_row[
                "centered_expected_net_value_per100_ci_upper"
            ]
        )

        plotted_values.extend(
            [
                not_lower,
                not_upper,
                under_lower,
                under_upper,
            ]
        )

        ax.plot(
            [not_estimate, under_estimate],
            [y_position, y_position],
            color=STYLE.light_gray,
            linewidth=STYLE.connector_line_width,
            zorder=1,
        )

        ax.errorbar(
            not_estimate,
            y_position,
            xerr=np.array(
                [
                    [not_estimate - not_lower],
                    [not_upper - not_estimate],
                ]
            ),
            fmt="o",
            color=STYLE.deep_blue,
            ecolor=STYLE.deep_blue,
            markerfacecolor="white",
            markeredgecolor=STYLE.deep_blue,
            markeredgewidth=1.0,
            markersize=STYLE.marker_size + 0.15,
            elinewidth=STYLE.error_line_width,
            capsize=STYLE.cap_size,
            label=(
                "Not under pressure"
                if y_position == 0
                else None
            ),
            zorder=3,
        )

        ax.errorbar(
            under_estimate,
            y_position,
            xerr=np.array(
                [
                    [under_estimate - under_lower],
                    [under_upper - under_estimate],
                ]
            ),
            fmt="o",
            color=STYLE.muted_orange,
            ecolor=STYLE.muted_orange,
            markerfacecolor=STYLE.muted_orange,
            markeredgecolor="white",
            markeredgewidth=0.65,
            markersize=STYLE.marker_size + 0.15,
            elinewidth=STYLE.error_line_width,
            capsize=STYLE.cap_size,
            label=(
                "Under pressure"
                if y_position == 0
                else None
            ),
            zorder=4,
        )

        delta = under_estimate - not_estimate
        midpoint = 0.5 * (not_estimate + under_estimate)
        ax.text(
            midpoint,
            y_position - 0.17,
            f"Δ {delta:.2f}",
            ha="center",
            va="top",
            fontsize=STYLE.annotation_size,
            color=STYLE.dark_slate,
        )

    ax.axvline(
        0.0,
        color=STYLE.light_gray,
        linewidth=STYLE.reference_line_width,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [
            "Ground Pass",
            "Low Pass",
            "High Pass",
        ]
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        "Centered expected net value per 100 passes"
    )
    ax.set_title(
        "Pressure-related change by pass height",
        loc="left",
        pad=20,
    )
    clean_axis(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    x_min = min(plotted_values)
    x_max = max(plotted_values)
    x_span = max(x_max - x_min, 1.0)
    ax.set_xlim(
        x_min - 0.08 * x_span,
        x_max + 0.08 * x_span,
    )

    ax.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    save_figure_all_formats(
        figure,
        output_dir,
        "Figure_S3_pressure_height",
    )
    plt.close(figure)



def build_results_text(
    risk_bins: pd.DataFrame,
    pressure_summary: pd.DataFrame,
    height_summary: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> str:
    pressure_contrast = contrasts.loc[
        contrasts["contrast_family"].eq("pressure")
        & contrasts["metric"].eq(
            "centered_expected_net_value_per100"
        )
    ].iloc[0]

    pressure_success = contrasts.loc[
        contrasts["contrast_family"].eq("pressure")
        & contrasts["metric"].eq(
            "expected_success_value_per100"
        )
    ].iloc[0]
    pressure_failure = contrasts.loc[
        contrasts["contrast_family"].eq("pressure")
        & contrasts["metric"].eq(
            "expected_failure_cost_per100"
        )
    ].iloc[0]

    height_net = height_summary.set_index("category")[
        "centered_expected_net_value_per100"
    ]

    minimum_net_bin = risk_bins.loc[
        risk_bins["expected_net_value_per100"].idxmin()
    ]
    maximum_net_bin = risk_bins.loc[
        risk_bins["expected_net_value_per100"].idxmax()
    ]
    positive_bin_count = int(
        np.sum(risk_bins["expected_net_value_per100"] > 0.0)
    )

    return f"""FULL-SAMPLE APPLICATION

The locked H=3 framework was applied to {int(pressure_summary['passes'].sum()):,} primary open-play passes. Across 20 equal-frequency failure-risk bins, mean expected net value was closest to zero at a mean failure probability of {float(maximum_net_bin['mean_failure_risk']):.3f} ({float(maximum_net_bin['expected_net_value_per100']):.3f} per 100 passes; 95% CI {float(maximum_net_bin['expected_net_value_per100_ci_lower']):.3f} to {float(maximum_net_bin['expected_net_value_per100_ci_upper']):.3f}) and was lowest at a mean failure probability of {float(minimum_net_bin['mean_failure_risk']):.3f} ({float(minimum_net_bin['expected_net_value_per100']):.3f} per 100 passes; 95% CI {float(minimum_net_bin['expected_net_value_per100_ci_lower']):.3f} to {float(minimum_net_bin['expected_net_value_per100_ci_upper']):.3f}). The number of bins with a positive mean expected net value was {positive_bin_count}. This descriptive pattern should not be interpreted as a prespecified causal or optimal-risk threshold.

Passes recorded as under pressure had lower expected success value (difference {float(pressure_success['estimate']):.3f} per 100 passes) and higher expected failure cost (difference {float(pressure_failure['estimate']):+.3f} per 100 passes) than passes not recorded as under pressure. The resulting difference in centered expected net value was {float(pressure_contrast['estimate']):.3f} per 100 passes (95% CI {float(pressure_contrast['ci_lower']):.3f} to {float(pressure_contrast['ci_upper']):.3f}).

Centered expected net value per 100 passes was {float(height_net['Ground Pass']):.3f} for Ground Passes, {float(height_net['Low Pass']):.3f} for Low Passes, and {float(height_net['High Pass']):.3f} for High Passes. These estimates are descriptive contextual comparisons; the corresponding pass-height contrasts in expected success value, expected failure cost, and centered expected net value are reported in the main figure and numerical tables.
"""


def main() -> None:
    print(f"Module 05 script version: {SCRIPT_VERSION}")
    args = parse_args()

    script_path = Path(__file__).resolve()
    style_path = script_path.with_name(
        "publication_style.py"
    )

    if not args.pass_value_csv.exists():
        raise FileNotFoundError(args.pass_value_csv)
    if (
        args.pass_metadata_csv is not None
        and not args.pass_metadata_csv.exists()
    ):
        raise FileNotFoundError(args.pass_metadata_csv)
    if not style_path.exists():
        raise FileNotFoundError(style_path)

    if args.bootstrap_reps < 200:
        raise ValueError(
            "Use at least 200 match-cluster bootstrap repetitions."
        )
    if args.risk_bins < 5:
        raise ValueError("Use at least five failure-risk bins.")

    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, args.overwrite)
    configure_matplotlib()

    (
        frame,
        detected_columns,
        merge_audit,
    ) = load_analysis_frame(
        args.pass_value_csv.resolve(),
        args.pass_metadata_csv.resolve()
        if args.pass_metadata_csv
        else None,
        args.chunksize,
    )

    known_height = frame.loc[
        frame["pass_height_group"].isin(HEIGHT_ORDER)
    ].copy()

    descriptives = full_sample_descriptives(frame)
    risk_bins = risk_bin_summary_with_bootstrap(
        frame,
        args.risk_bins,
        args.bootstrap_reps,
        args.seed,
    )
    pressure_summary = summarize_categories(
        frame,
        "pressure_status",
        PRESSURE_ORDER,
        list(FRAMEWORK_METRICS),
        args.bootstrap_reps,
        args.seed + 1,
    )
    height_summary = summarize_categories(
        known_height,
        "pass_height_group",
        HEIGHT_ORDER,
        list(FRAMEWORK_METRICS),
        args.bootstrap_reps,
        args.seed + 2,
    )
    pressure_height = pressure_height_summary(
        frame,
        args.bootstrap_reps,
        args.seed + 3,
    )
    contrasts = paired_context_contrasts(
        known_height,
        args.bootstrap_reps,
        args.seed + 4,
    )

    descriptives.to_csv(
        output_dir / "full_sample_framework_descriptives.csv",
        index=False,
        encoding="utf-8-sig",
    )
    risk_bins.to_csv(
        output_dir / "full_sample_risk_bin_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pressure_summary.to_csv(
        output_dir / "full_sample_pressure_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    height_summary.to_csv(
        output_dir / "full_sample_pass_height_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pressure_height.to_csv(
        output_dir / "full_sample_pressure_height_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    contrasts.to_csv(
        output_dir / "full_sample_context_contrasts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_figure_2(
        risk_bins,
        contrasts,
        output_dir,
    )
    plot_pressure_height_supplement(
        pressure_height,
        output_dir,
    )

    results_text = build_results_text(
        risk_bins,
        pressure_summary,
        height_summary,
        contrasts,
    )
    (
        output_dir / "module05_results_text.txt"
    ).write_text(
        results_text,
        encoding="utf-8",
    )

    captions = """Figure 2. Full-sample application of the locked H=3 pass risk–reward framework. (A) Mean expected success value, expected failure cost, and expected net value across equal-frequency bins of pass failure probability; shaded areas show percentile 95% confidence intervals from match-cluster bootstrap resampling. (B) Differences between passes recorded as under pressure and passes not recorded as under pressure. (C) Differences between Low and Ground Passes and between High and Ground Passes. Points in panels B and C show paired differences and horizontal intervals show percentile 95% confidence intervals from match-cluster bootstrap resampling.

Supplementary Figure S3. Pressure-related changes in centered expected net value across Ground, Low, and High passes. Open blue points denote passes not recorded as under pressure, filled orange points denote passes recorded as under pressure, and connecting lines show descriptive within-height differences. Horizontal intervals show percentile 95% confidence intervals for each category mean from match-cluster bootstrap resampling. The figure does not constitute a formal pressure-by-height interaction test. The pressure variable is the StatsBomb event-level under-pressure indicator.
"""
    (
        output_dir / "module05_figure_captions.txt"
    ).write_text(
        captions,
        encoding="utf-8",
    )

    checks = {
        "expected_rows": len(frame) == EXPECTED_ROWS,
        "expected_matches": (
            frame["match_id"].nunique() == EXPECTED_MATCHES
        ),
        "binary_pressure_after_normalization": (
            set(frame["pressure_status"].unique())
            <= set(PRESSURE_ORDER)
        ),
        "all_three_height_categories_present": (
            set(HEIGHT_ORDER)
            <= set(frame["pass_height_group"].unique())
        ),
        "all_framework_metrics_complete": (
            not frame[
                [
                    "xpass_success",
                    "p_success_main_oof",
                    "execution_residual",
                    "expected_success_value",
                    "expected_failure_cost",
                    "expected_net_value",
                    "realized_value",
                ]
            ].isna().any().any()
        ),
        "requested_risk_bins_created": (
            len(risk_bins) == args.risk_bins
        ),
        "no_duplicate_pass_value_event_ids": (
            merge_audit[
                "pass_value_duplicate_event_id_count"
            ]
            == 0
        ),
        "no_duplicate_metadata_event_ids": (
            merge_audit[
                "metadata_duplicate_event_id_count"
            ]
            == 0
        ),
        "no_unmatched_context_rows": (
            merge_audit["unmatched_event_rows"] == 0
        ),
    }

    output_hashes = collect_file_hashes(
        output_dir,
        excluded_names={
            "module05_audit.json",
            "SHA256SUMS.txt",
        },
    )

    audit = {
        "status": (
            "PASS"
            if all(checks.values())
            else "REVIEW_REQUIRED"
        ),
        "script_version": SCRIPT_VERSION,
        "purpose": (
            "Full-sample application of the locked H=3 pass "
            "risk–reward framework."
        ),
        "rows": int(len(frame)),
        "matches": int(frame["match_id"].nunique()),
        "unknown_pass_height_rows": int(
            np.sum(frame["pass_height_group"] == "Unknown")
        ),
        "full_sample_expected_net_value_per100": float(
            frame["expected_net_value"].mean() * 100.0
        ),
        "bootstrap": {
            "unit": "match_id",
            "repetitions": args.bootstrap_reps,
            "seed": args.seed,
        },
        "risk_bins": {
            "method": "equal-frequency",
            "requested": args.risk_bins,
            "created": len(risk_bins),
        },
        "merge_audit": merge_audit,
        "checks": checks,
        "detected_columns": detected_columns,
        "input_hashes": {
            "pass_value_csv_sha256": sha256_file(
                args.pass_value_csv.resolve()
            ),
            "pass_metadata_csv_sha256": (
                sha256_file(
                    args.pass_metadata_csv.resolve()
                )
                if args.pass_metadata_csv
                else None
            ),
        },
        "code_hashes": {
            "script_sha256": sha256_file(script_path),
            "style_sha256": sha256_file(style_path),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "output_hashes_before_audit": output_hashes,
        "hash_manifest": "SHA256SUMS.txt",
        "outputs": sorted(
            list(output_hashes)
            + [
                "module05_audit.json",
                "SHA256SUMS.txt",
            ]
        ),
    }

    audit_path = output_dir / "module05_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_sha256sums(output_dir)

    print("=" * 100)
    print("Final-audit Module 05 completed")
    print("=" * 100)
    print(f"Matches: {frame['match_id'].nunique():,}")
    print(f"Passes: {len(frame):,}")
    print(f"Risk bins: {len(risk_bins)}")
    print(f"Status: {audit['status']}")
    print(f"Outputs: {output_dir}")

if __name__ == "__main__":
    main()
