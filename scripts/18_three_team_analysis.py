#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
18_three_team_analysis.py

Revised Module 06: compressed illustrative application of the locked H=3
pass risk–reward framework to three pre-specified team-season cohorts.

Narrative role
--------------
This module is not a taxonomy of team playing styles. It uses three
contrasting team-season cohorts to show that the same transparent model can
distinguish:

- passing structure;
- pass difficulty;
- execution relative to xPass expectation;
- potential successful value;
- failure exposure;
- expected net value;
- pressure-related value change.

Three cohorts
-------------
- Bayer Leverkusen, Bundesliga 2023/24
- Athletic Club, La Liga 2015/16
- AS Monaco, Ligue 1 2015/16

Eight core profile dimensions
-----------------------------
1. Under-pressure pass share
2. Ground-pass share
3. High-pass share
4. Mean xPass
5. Execution residual
6. Expected success value per 100 passes
7. Expected failure cost per 100 passes
8. Centered expected net value per 100 passes

Pressure-related net-value change is shown separately in Figure 3D.

No model is fitted or modified.

Required files
--------------
--pass_value_csv:
    result2/02_pctv_h3/pass_value_oof.csv.gz

--pass_metadata_csv:
    passes_preprocessed_with_open_play_flags.csv

--matches_csv:
    statsbomb_matches_selected.csv, required only when a fixed cohort manifest
    is not supplied.

--style_file:
    publication_style.py must be in the same folder as this script.

Optional
--------
--cohort_manifest_csv:
    CSV with columns match_id, cohort, focal_team.
    Use only when automatic identification does not return exactly
    34 + 38 + 38 matches.

Main outputs
------------
Figure_3_three_team_application.png/.pdf/.svg
three_team_match_manifest.csv
three_team_core_profile_table.csv
three_team_core_profile_wide.csv
three_team_core_profile_standardized.csv
three_team_pressure_effects.csv
three_team_pass_height_structure.csv
module06_results_text.txt
module06_figure_captions.txt
module06_audit.json
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
        profile_diverging_cmap,
        save_figure_all_formats,
        team_color_map,
    )
except ImportError as exc:
    raise ImportError(
        "publication_style.py must be saved in the same folder as "
        "18_three_team_analysis.py."
    ) from exc


SCRIPT_VERSION = "2026-07-31-final-audit-1"

EXPECTED_MATCHES = 110
EXPECTED_PASSES = 54_795

COHORT_ORDER = [
    "Bayer Leverkusen 2023/24",
    "Athletic Club 2015/16",
    "AS Monaco 2015/16",
]

COHORT_SPECIFICATIONS = {
    "Bayer Leverkusen 2023/24": {
        "canonical_team": "Bayer Leverkusen",
        "team_aliases": [
            "Bayer Leverkusen",
            "Bayer 04 Leverkusen",
            "Leverkusen",
        ],
        "season_digits": "20232024",
        "competition_aliases": [
            "Bundesliga",
            "1. Bundesliga",
        ],
        "expected_matches": 34,
    },
    "Athletic Club 2015/16": {
        "canonical_team": "Athletic Club",
        "team_aliases": [
            "Athletic Club",
            "Athletic Bilbao",
        ],
        "season_digits": "20152016",
        "competition_aliases": [
            "La Liga",
        ],
        "expected_matches": 38,
    },
    "AS Monaco 2015/16": {
        "canonical_team": "AS Monaco",
        "team_aliases": [
            "AS Monaco",
            "Monaco",
        ],
        "season_digits": "20152016",
        "competition_aliases": [
            "Ligue 1",
        ],
        "expected_matches": 38,
    },
}

PRESSURE_ORDER = ["Not under pressure", "Under pressure"]
HEIGHT_ORDER = ["Ground Pass", "Low Pass", "High Pass"]

COLUMN_CANDIDATES = {
    "match_id": ["match_id", "matchid"],
    "event_id": ["id", "event_id", "event_uuid"],
    "team": ["team", "team_name", "possession_team_name"],
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

MATCH_COLUMN_CANDIDATES = {
    "match_id": ["match_id", "matchid"],
    "season": [
        "season_name",
        "season",
        "season_season_name",
    ],
    "competition": [
        "competition_name",
        "competition",
        "competition_competition_name",
    ],
    "home_team": [
        "home_team_name",
        "home_team",
        "home_team_home_team_name",
    ],
    "away_team": [
        "away_team_name",
        "away_team",
        "away_team_away_team_name",
    ],
}

CORE_METRICS = [
    {
        "key": "under_pressure_share",
        "label": "Under-pressure\npass share",
        "unit": "proportion",
        "type": "share",
        "category_column": "pressure_status",
        "category": "Under pressure",
    },
    {
        "key": "ground_pass_share",
        "label": "Ground-pass\nshare",
        "unit": "proportion",
        "type": "share",
        "category_column": "pass_height_group",
        "category": "Ground Pass",
    },
    {
        "key": "high_pass_share",
        "label": "High-pass\nshare",
        "unit": "proportion",
        "type": "share",
        "category_column": "pass_height_group",
        "category": "High Pass",
    },
    {
        "key": "mean_xpass",
        "label": "Mean\nxPass",
        "unit": "probability",
        "type": "mean",
        "source": "p_success_main_oof",
        "scale": 1.0,
    },
    {
        "key": "execution_residual",
        "label": "Execution\nresidual",
        "unit": "probability difference",
        "type": "mean",
        "source": "execution_residual",
        "scale": 1.0,
    },
    {
        "key": "expected_success_value_per100",
        "label": "Expected success\nvalue / 100",
        "unit": "value per 100 passes",
        "type": "mean",
        "source": "expected_success_value",
        "scale": 100.0,
    },
    {
        "key": "expected_failure_cost_per100",
        "label": "Expected failure\ncost / 100",
        "unit": "value per 100 passes",
        "type": "mean",
        "source": "expected_failure_cost",
        "scale": 100.0,
    },
    {
        "key": "centered_expected_net_value_per100",
        "label": "Centered expected\nnet value / 100",
        "unit": "value per 100 passes",
        "type": "mean",
        "source": "expected_net_value_centered",
        "scale": 100.0,
    },
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
            "Create the compressed three-team illustrative application "
            "and publication-quality Figure 3."
        )
    )
    parser.add_argument("--pass_value_csv", type=Path, required=True)
    parser.add_argument(
        "--pass_metadata_csv",
        type=Path,
        required=True,
    )
    parser.add_argument("--matches_csv", type=Path)
    parser.add_argument("--cohort_manifest_csv", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260731)
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


def normalize_pressure(value: object) -> str:
    """Normalize StatsBomb under_pressure values to two categories."""
    if pd.isna(value):
        return "Not under pressure"

    if isinstance(value, (bool, np.bool_)):
        return "Under pressure" if bool(value) else "Not under pressure"

    text = str(value).strip().lower()

    if text in {
        "1",
        "1.0",
        "true",
        "yes",
        "y",
        "under pressure",
        "under_pressure",
    }:
        return "Under pressure"

    if text in {
        "0",
        "0.0",
        "false",
        "no",
        "n",
        "",
        "nan",
        "none",
        "not under pressure",
        "not_under_pressure",
    }:
        return "Not under pressure"

    raise ValueError(
        f"Unrecognized under_pressure value: {value!r}"
    )


def normalize_pass_height(value: object) -> str:
    """Normalize pass-height labels to Ground, Low, High, or Unknown."""
    mapping = {
        "groundpass": "Ground Pass",
        "ground": "Ground Pass",
        "lowpass": "Low Pass",
        "low": "Low Pass",
        "highpass": "High Pass",
        "high": "High Pass",
    }
    return mapping.get(normalize_text(value), "Unknown")


def season_digits(value: object) -> str:
    if pd.isna(value):
        return ""
    return "".join(re.findall(r"\d", str(value)))


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


def prepare_output_dir(
    output_dir: Path,
    overwrite: bool,
) -> None:
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


def exact_alias_match(
    value: object,
    aliases: list[str],
) -> bool:
    normalized_value = normalize_text(value)
    return any(
        normalized_value == normalize_text(alias)
        for alias in aliases
    )


def load_user_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )
    required = {"match_id", "cohort", "focal_team"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise RuntimeError(
            "Cohort manifest is missing columns: "
            + ", ".join(missing)
        )

    manifest = manifest[
        ["match_id", "cohort", "focal_team"]
    ].copy()
    manifest["match_id"] = manifest["match_id"].map(
        normalize_id
    )

    if manifest["match_id"].duplicated().any():
        raise RuntimeError(
            "The cohort manifest contains duplicate match IDs."
        )

    return manifest


def identify_cohort_matches(
    matches_csv: Path,
) -> pd.DataFrame:
    matches = pd.read_csv(
        matches_csv,
        encoding="utf-8-sig",
        low_memory=False,
    )

    detected = {
        logical_name: detect_column(
            matches.columns,
            candidates,
            logical_name,
            required=logical_name != "competition",
        )
        for logical_name, candidates in MATCH_COLUMN_CANDIDATES.items()
    }

    rows: list[dict[str, object]] = []

    for _, row in matches.iterrows():
        match_id = normalize_id(
            row[detected["match_id"]]
        )
        season_value = row[detected["season"]]
        home_team = row[detected["home_team"]]
        away_team = row[detected["away_team"]]
        competition_value = (
            row[detected["competition"]]
            if detected["competition"] is not None
            else ""
        )

        for cohort, specification in (
            COHORT_SPECIFICATIONS.items()
        ):
            season_match = (
                specification["season_digits"]
                in season_digits(season_value)
            )
            team_match = (
                exact_alias_match(
                    home_team,
                    specification["team_aliases"],
                )
                or exact_alias_match(
                    away_team,
                    specification["team_aliases"],
                )
            )

            if detected["competition"] is not None:
                competition_match = any(
                    normalize_text(competition_value)
                    == normalize_text(alias)
                    for alias in specification[
                        "competition_aliases"
                    ]
                )
            else:
                competition_match = True

            if season_match and team_match and competition_match:
                rows.append(
                    {
                        "match_id": match_id,
                        "cohort": cohort,
                        "focal_team": specification[
                            "canonical_team"
                        ],
                        "season_source": str(season_value),
                        "competition_source": str(
                            competition_value
                        ),
                        "home_team_source": str(home_team),
                        "away_team_source": str(away_team),
                    }
                )

    manifest = pd.DataFrame(rows)

    if manifest.empty:
        raise RuntimeError(
            "Automatic cohort identification returned no matches. "
            "Use --cohort_manifest_csv."
        )

    if manifest["match_id"].duplicated().any():
        duplicates = manifest.loc[
            manifest["match_id"].duplicated(keep=False)
        ]
        raise RuntimeError(
            "Some matches were assigned to more than one cohort:\n"
            + duplicates.to_string(index=False)
        )

    counts = (
        manifest.groupby("cohort")["match_id"]
        .nunique()
        .to_dict()
    )

    errors = []
    for cohort, specification in COHORT_SPECIFICATIONS.items():
        observed = int(counts.get(cohort, 0))
        expected = int(specification["expected_matches"])
        if observed != expected:
            errors.append(
                f"{cohort}: expected {expected}, found {observed}"
            )

    if errors:
        raise RuntimeError(
            "Automatic cohort identification did not reproduce the "
            "locked 34/38/38 match counts:\n"
            + "\n".join(errors)
            + "\nUse --cohort_manifest_csv after checking the metadata."
        )

    return manifest.sort_values(
        ["cohort", "match_id"]
    ).reset_index(drop=True)



def load_selected_pass_values(
    pass_value_csv: Path,
    selected_match_ids: set[str],
    chunksize: int,
) -> tuple[
    pd.DataFrame,
    dict[str, str | None],
    float,
    int,
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
            not in {"team", "under_pressure", "pass_height"},
        )

    usecols = sorted(
        {
            column
            for column in detected.values()
            if column is not None
        }
    )
    match_column = detected["match_id"]
    event_column = detected["event_id"]
    net_column = detected["expected_net_value"]

    selected_parts: list[pd.DataFrame] = []
    global_net_sum = 0.0
    global_net_count = 0

    for chunk in pd.read_csv(
        pass_value_csv,
        encoding="utf-8-sig",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        net_values = pd.to_numeric(
            chunk[net_column],
            errors="coerce",
        )
        valid = net_values.notna()
        global_net_sum += float(
            net_values.loc[valid].sum()
        )
        global_net_count += int(valid.sum())

        normalized_matches = chunk[
            match_column
        ].map(normalize_id)
        mask = normalized_matches.isin(
            selected_match_ids
        )

        selected = chunk.loc[mask].copy()
        if not selected.empty:
            selected["_match_id_norm"] = (
                normalized_matches.loc[
                    selected.index
                ].to_numpy()
            )
            selected_parts.append(selected)

    if not selected_parts:
        raise RuntimeError(
            "No selected cohort rows were found in the pass-value file."
        )
    if global_net_count == 0:
        raise RuntimeError(
            "No valid expected-net-value rows were found."
        )

    selected_frame = pd.concat(
        selected_parts,
        ignore_index=True,
    )
    selected_frame["_event_id_norm"] = selected_frame[
        event_column
    ].map(normalize_id)

    duplicate_mask = selected_frame[
        "_event_id_norm"
    ].duplicated(keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_ids = int(
        selected_frame.loc[
            duplicate_mask,
            "_event_id_norm",
        ].nunique()
    )

    if duplicate_ids > 0:
        raise RuntimeError(
            "Duplicate event IDs were found in selected pass-value rows: "
            f"{duplicate_ids} duplicated IDs across "
            f"{duplicate_rows} rows."
        )

    load_audit = {
        "selected_pass_value_rows": int(len(selected_frame)),
        "selected_pass_value_matches": int(
            selected_frame["_match_id_norm"].nunique()
        ),
        "selected_pass_value_unique_event_ids": int(
            selected_frame["_event_id_norm"].nunique()
        ),
        "selected_pass_value_duplicate_event_id_rows": (
            duplicate_rows
        ),
        "selected_pass_value_duplicate_event_id_count": (
            duplicate_ids
        ),
        "global_formal_rows": int(global_net_count),
    }

    return (
        selected_frame,
        detected,
        global_net_sum / global_net_count,
        global_net_count,
        load_audit,
    )


def load_selected_metadata(
    pass_metadata_csv: Path,
    selected_match_ids: set[str],
    chunksize: int,
) -> tuple[
    pd.DataFrame,
    dict[str, str],
    dict[str, object],
]:
    header = pd.read_csv(
        pass_metadata_csv,
        encoding="utf-8-sig",
        nrows=0,
    )
    columns = header.columns.tolist()

    detected = {
        logical_name: detect_column(
            columns,
            COLUMN_CANDIDATES[logical_name],
            logical_name,
        )
        for logical_name in [
            "match_id",
            "event_id",
            "team",
            "under_pressure",
            "pass_height",
        ]
    }

    usecols = list(dict.fromkeys(detected.values()))
    match_column = detected["match_id"]

    selected_parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        pass_metadata_csv,
        encoding="utf-8-sig",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        normalized_matches = chunk[
            match_column
        ].map(normalize_id)
        mask = normalized_matches.isin(
            selected_match_ids
        )

        selected = chunk.loc[mask].copy()
        if not selected.empty:
            selected["_match_id_norm"] = (
                normalized_matches.loc[
                    selected.index
                ].to_numpy()
            )
            selected_parts.append(selected)

    if not selected_parts:
        raise RuntimeError(
            "No selected cohort rows were found in the pass metadata."
        )

    metadata = pd.concat(
        selected_parts,
        ignore_index=True,
    )
    metadata["_event_id_norm"] = metadata[
        detected["event_id"]
    ].map(normalize_id)

    duplicate_mask = metadata[
        "_event_id_norm"
    ].duplicated(keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_ids = int(
        metadata.loc[
            duplicate_mask,
            "_event_id_norm",
        ].nunique()
    )

    if duplicate_ids > 0:
        raise RuntimeError(
            "Duplicate event IDs were found in selected metadata rows: "
            f"{duplicate_ids} duplicated IDs across "
            f"{duplicate_rows} rows. Duplicates are not silently removed "
            "in the final-audit script."
        )

    metadata_audit = {
        "selected_metadata_rows": int(len(metadata)),
        "selected_metadata_matches": int(
            metadata["_match_id_norm"].nunique()
        ),
        "selected_metadata_unique_event_ids": int(
            metadata["_event_id_norm"].nunique()
        ),
        "selected_metadata_duplicate_event_id_rows": (
            duplicate_rows
        ),
        "selected_metadata_duplicate_event_id_count": (
            duplicate_ids
        ),
    }

    return metadata, detected, metadata_audit


def merge_and_filter_own_team_events(
    pass_values: pd.DataFrame,
    pass_value_columns: dict[str, str | None],
    metadata: pd.DataFrame,
    metadata_columns: dict[str, str],
    manifest: pd.DataFrame,
    global_net_mean: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    pass_event_column = pass_value_columns["event_id"]

    if "_event_id_norm" not in pass_values.columns:
        pass_values["_event_id_norm"] = pass_values[
            pass_event_column
        ].map(normalize_id)

    pass_duplicate_mask = pass_values[
        "_event_id_norm"
    ].duplicated(keep=False)
    metadata_duplicate_mask = metadata[
        "_event_id_norm"
    ].duplicated(keep=False)

    pass_duplicate_ids = int(
        pass_values.loc[
            pass_duplicate_mask,
            "_event_id_norm",
        ].nunique()
    )
    metadata_duplicate_ids = int(
        metadata.loc[
            metadata_duplicate_mask,
            "_event_id_norm",
        ].nunique()
    )

    if pass_duplicate_ids > 0 or metadata_duplicate_ids > 0:
        raise RuntimeError(
            "Duplicate event IDs remain before the event-level merge."
        )

    metadata_keep = [
        "_event_id_norm",
        metadata_columns["team"],
        metadata_columns["under_pressure"],
        metadata_columns["pass_height"],
    ]
    metadata_small = metadata[metadata_keep].copy()

    merged = pass_values.merge(
        metadata_small,
        on="_event_id_norm",
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_metadata"),
    )

    matched_rows = int((merged["_merge"] == "both").sum())
    unmatched_rows = int(
        (merged["_merge"] == "left_only").sum()
    )

    if unmatched_rows > 0:
        raise RuntimeError(
            "Some selected pass-value events did not match metadata: "
            f"{unmatched_rows} unmatched rows."
        )

    merged = merged.drop(columns=["_merge"])

    manifest_small = manifest[
        ["match_id", "cohort", "focal_team"]
    ].rename(
        columns={"match_id": "_match_id_norm"}
    )

    merged = merged.merge(
        manifest_small,
        on="_match_id_norm",
        how="inner",
        validate="many_to_one",
    )

    rows_before_team_filter = int(len(merged))
    matches_before_team_filter = int(
        merged["_match_id_norm"].nunique()
    )

    team_column = metadata_columns["team"]

    own_team_mask = np.array(
        [
            exact_alias_match(
                team_value,
                COHORT_SPECIFICATIONS[cohort][
                    "team_aliases"
                ],
            )
            for team_value, cohort in zip(
                merged[team_column],
                merged["cohort"],
            )
        ],
        dtype=bool,
    )
    own_team_rows = int(own_team_mask.sum())
    opponent_rows = int(
        rows_before_team_filter - own_team_rows
    )
    merged = merged.loc[own_team_mask].copy()

    rename_map = {
        pass_value_columns["event_id"]: "event_id",
        pass_value_columns["xpass_success"]: "xpass_success",
        pass_value_columns[
            "p_success_main_oof"
        ]: "p_success_main_oof",
        pass_value_columns[
            "execution_residual"
        ]: "execution_residual",
        pass_value_columns[
            "expected_success_value"
        ]: "expected_success_value",
        pass_value_columns[
            "expected_failure_cost"
        ]: "expected_failure_cost",
        pass_value_columns[
            "expected_net_value"
        ]: "expected_net_value",
        pass_value_columns[
            "realized_value"
        ]: "realized_value",
        team_column: "team_source",
        metadata_columns[
            "under_pressure"
        ]: "under_pressure_source",
        metadata_columns[
            "pass_height"
        ]: "pass_height_source",
    }
    merged = merged.rename(columns=rename_map)

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
        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce",
        )

    if merged[numeric_columns].isna().any().any():
        missing = merged[numeric_columns].isna().sum()
        raise RuntimeError(
            "Missing analysis values in selected team events:\n"
            + missing[missing > 0].to_string()
        )

    merged["match_id"] = merged[
        "_match_id_norm"
    ]
    merged["event_id"] = merged[
        "event_id"
    ].map(normalize_id)
    merged["pressure_status"] = merged[
        "under_pressure_source"
    ].map(normalize_pressure)
    merged["pass_height_group"] = merged[
        "pass_height_source"
    ].map(normalize_pass_height)
    merged["expected_net_value_centered"] = (
        merged["expected_net_value"] - global_net_mean
    )

    keep_columns = [
        "event_id",
        "match_id",
        "cohort",
        "focal_team",
        "team_source",
        "pressure_status",
        "pass_height_group",
        "xpass_success",
        "p_success_main_oof",
        "execution_residual",
        "expected_success_value",
        "expected_failure_cost",
        "expected_net_value",
        "expected_net_value_centered",
        "realized_value",
    ]

    retained_matches = set(merged["match_id"].unique())
    expected_matches = set(
        manifest["match_id"].map(normalize_id)
    )
    zero_pass_matches = sorted(
        expected_matches - retained_matches
    )

    if zero_pass_matches:
        raise RuntimeError(
            "Some selected cohort matches retained no focal-team passes: "
            + ", ".join(zero_pass_matches)
        )

    merge_audit = {
        "pass_value_rows_before_merge": int(len(pass_values)),
        "metadata_rows_before_merge": int(len(metadata_small)),
        "matched_event_rows": matched_rows,
        "unmatched_event_rows": unmatched_rows,
        "rows_before_team_filter": rows_before_team_filter,
        "matches_before_team_filter": (
            matches_before_team_filter
        ),
        "own_team_rows_retained": own_team_rows,
        "opponent_rows_excluded": opponent_rows,
        "retained_matches": int(len(retained_matches)),
        "matches_with_zero_retained_passes": int(
            len(zero_pass_matches)
        ),
        "zero_retained_pass_match_ids": zero_pass_matches,
        "unknown_pass_height_rows": int(
            np.sum(merged["pass_height_group"] == "Unknown")
        ),
    }

    return (
        merged[keep_columns].reset_index(drop=True),
        merge_audit,
    )

def bootstrap_mean_by_match(
    team_events: pd.DataFrame,
    source_column: str,
    scale: float,
    bootstrap_reps: int,
    seed: int,
) -> tuple[float, float, float]:
    matches = sorted(
        team_events["match_id"].unique().tolist()
    )
    grouped = (
        team_events.groupby("match_id")[source_column]
        .agg(["sum", "count"])
        .reindex(matches)
        .fillna(0.0)
    )

    sums = grouped["sum"].to_numpy(dtype=np.float64)
    counts = grouped["count"].to_numpy(dtype=np.float64)

    estimate = float(
        sums.sum() / counts.sum() * scale
    )

    rng = np.random.default_rng(seed)
    bootstrap_values = np.empty(
        bootstrap_reps,
        dtype=np.float64,
    )

    for repetition in range(bootstrap_reps):
        sampled = rng.integers(
            0,
            len(matches),
            size=len(matches),
        )
        bootstrap_values[repetition] = (
            sums[sampled].sum()
            / counts[sampled].sum()
            * scale
        )

    lower, upper = np.quantile(
        bootstrap_values,
        [0.025, 0.975],
    )
    return estimate, float(lower), float(upper)


def bootstrap_share_by_match(
    team_events: pd.DataFrame,
    category_column: str,
    category: str,
    bootstrap_reps: int,
    seed: int,
) -> tuple[float, float, float]:
    matches = sorted(
        team_events["match_id"].unique().tolist()
    )

    total_counts = (
        team_events.groupby("match_id")
        .size()
        .reindex(matches)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    category_counts = (
        team_events.loc[
            team_events[category_column].eq(category)
        ]
        .groupby("match_id")
        .size()
        .reindex(matches)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )

    estimate = float(
        category_counts.sum() / total_counts.sum()
    )

    rng = np.random.default_rng(seed)
    bootstrap_values = np.empty(
        bootstrap_reps,
        dtype=np.float64,
    )

    for repetition in range(bootstrap_reps):
        sampled = rng.integers(
            0,
            len(matches),
            size=len(matches),
        )
        bootstrap_values[repetition] = (
            category_counts[sampled].sum()
            / total_counts[sampled].sum()
        )

    lower, upper = np.quantile(
        bootstrap_values,
        [0.025, 0.975],
    )
    return estimate, float(lower), float(upper)


def build_core_profile_table(
    events: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for team_index, cohort in enumerate(COHORT_ORDER):
        team_events = events.loc[
            events["cohort"].eq(cohort)
        ].copy()

        if team_events.empty:
            raise RuntimeError(
                f"No team events were found for {cohort}."
            )

        for metric_index, specification in enumerate(
            CORE_METRICS
        ):
            metric_seed = (
                seed
                + 10_000 * team_index
                + 100 * metric_index
            )

            if specification["type"] == "share":
                estimate, lower, upper = (
                    bootstrap_share_by_match(
                        team_events,
                        specification["category_column"],
                        specification["category"],
                        bootstrap_reps,
                        metric_seed,
                    )
                )
            else:
                estimate, lower, upper = (
                    bootstrap_mean_by_match(
                        team_events,
                        specification["source"],
                        float(specification["scale"]),
                        bootstrap_reps,
                        metric_seed,
                    )
                )

            rows.append(
                {
                    "cohort": cohort,
                    "matches": int(
                        team_events["match_id"].nunique()
                    ),
                    "passes": int(len(team_events)),
                    "metric": specification["key"],
                    "metric_label": specification["label"].replace(
                        "\n",
                        " ",
                    ),
                    "unit": specification["unit"],
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "bootstrap_reps": bootstrap_reps,
                    "bootstrap_unit": "match_id",
                }
            )

    return pd.DataFrame(rows)


def build_pass_height_structure(
    events: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for team_index, cohort in enumerate(COHORT_ORDER):
        team_events = events.loc[
            events["cohort"].eq(cohort)
        ]

        for height_index, height in enumerate(HEIGHT_ORDER):
            estimate, lower, upper = (
                bootstrap_share_by_match(
                    team_events,
                    "pass_height_group",
                    height,
                    bootstrap_reps,
                    seed
                    + 10_000 * team_index
                    + 100 * height_index,
                )
            )

            rows.append(
                {
                    "cohort": cohort,
                    "pass_height": height,
                    "share": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "bootstrap_reps": bootstrap_reps,
                    "bootstrap_unit": "match_id",
                }
            )

    return pd.DataFrame(rows)


def pressure_effects_by_team(
    events: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    metric_specs = [
        {
            "metric": "execution_residual",
            "label": "Execution residual",
            "source": "execution_residual",
            "scale": 1.0,
        },
        {
            "metric": "centered_expected_net_value_per100",
            "label": (
                "Centered expected net value per 100 passes"
            ),
            "source": "expected_net_value_centered",
            "scale": 100.0,
        },
    ]

    rows: list[dict[str, object]] = []

    for team_index, cohort in enumerate(COHORT_ORDER):
        team_events = events.loc[
            events["cohort"].eq(cohort)
        ]
        matches = sorted(
            team_events["match_id"].unique().tolist()
        )
        n_matches = len(matches)

        for metric_index, specification in enumerate(
            metric_specs
        ):
            grouped = (
                team_events.groupby(
                    ["match_id", "pressure_status"],
                    observed=True,
                )[specification["source"]]
                .agg(["sum", "count"])
                .reset_index()
            )

            arrays: dict[str, dict[str, np.ndarray]] = {}
            match_index = {
                match_id: index
                for index, match_id in enumerate(matches)
            }

            for pressure in PRESSURE_ORDER:
                sums = np.zeros(
                    n_matches,
                    dtype=np.float64,
                )
                counts = np.zeros(
                    n_matches,
                    dtype=np.float64,
                )

                selected = grouped.loc[
                    grouped["pressure_status"].eq(
                        pressure
                    )
                ]
                for _, row in selected.iterrows():
                    position = match_index[row["match_id"]]
                    sums[position] = float(row["sum"])
                    counts[position] = float(row["count"])

                arrays[pressure] = {
                    "sum": sums,
                    "count": counts,
                }

            def calculate(
                pressure: str,
                sampled: np.ndarray | None,
            ) -> float:
                sums = arrays[pressure]["sum"]
                counts = arrays[pressure]["count"]

                if sampled is None:
                    numerator = sums.sum()
                    denominator = counts.sum()
                else:
                    numerator = sums[sampled].sum()
                    denominator = counts[sampled].sum()

                return float(
                    numerator
                    / denominator
                    * specification["scale"]
                )

            estimate = (
                calculate("Under pressure", None)
                - calculate("Not under pressure", None)
            )

            rng = np.random.default_rng(
                seed
                + 10_000 * team_index
                + 100 * metric_index
            )
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
                bootstrap_values[repetition] = (
                    calculate("Under pressure", sampled)
                    - calculate(
                        "Not under pressure",
                        sampled,
                    )
                )

            lower, upper = np.quantile(
                bootstrap_values,
                [0.025, 0.975],
            )

            rows.append(
                {
                    "cohort": cohort,
                    "metric": specification["metric"],
                    "metric_label": specification["label"],
                    "contrast": (
                        "Under pressure minus not under pressure"
                    ),
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


def standardized_profile_table(
    core_profile: pd.DataFrame,
) -> pd.DataFrame:
    wide = core_profile.pivot(
        index="cohort",
        columns="metric",
        values="estimate",
    ).loc[COHORT_ORDER]

    standardized = wide.copy()

    for column in standardized.columns:
        values = standardized[column].to_numpy(
            dtype=np.float64
        )
        mean = float(values.mean())
        standard_deviation = float(
            values.std(ddof=0)
        )

        if standard_deviation > 0:
            standardized[column] = (
                values - mean
            ) / standard_deviation
        else:
            standardized[column] = 0.0

    return (
        standardized.reset_index()
        .melt(
            id_vars="cohort",
            var_name="metric",
            value_name="standardized_value",
        )
    )


def wide_profile_table(
    core_profile: pd.DataFrame,
) -> pd.DataFrame:
    estimates = core_profile.pivot(
        index=["cohort", "matches", "passes"],
        columns="metric",
        values="estimate",
    ).reset_index()
    estimates.columns.name = None
    return estimates


def plot_figure_3(
    core_profile: pd.DataFrame,
    standardized: pd.DataFrame,
    pass_height_structure: pd.DataFrame,
    pressure_effects: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Main-text Figure 3.

    Layout:
        A spans the full top row.
        B, C, and D occupy the bottom row.

    A: Eight-dimension standardized profile heatmap.
    B: Centered expected net value.
    C: Ground/Low/High pass structure.
    D: Pressure-related change in centered expected net value.
    """
    figure = plt.figure(
        figsize=(
            STYLE.double_column_width_in,
            STYLE.figure3_height_in,
        )
    )
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=[1.13, 1.0],
        width_ratios=[1.0, 1.05, 1.0],
        left=0.13,
        right=0.965,
        bottom=0.11,
        top=0.91,
        wspace=0.62,
        hspace=0.74,
    )

    ax_a = figure.add_subplot(grid[0, :])
    ax_b = figure.add_subplot(grid[1, 0])
    ax_c = figure.add_subplot(grid[1, 1])
    ax_d = figure.add_subplot(grid[1, 2])

    team_colors = team_color_map()
    height_colors = height_color_map()
    short_team_labels = [
        "Bayer Leverkusen",
        "Athletic Club",
        "AS Monaco",
    ]
    y_positions = np.arange(len(COHORT_ORDER))

    # ------------------------------------------------------------------
    # Panel A: standardized core profile heatmap.
    # ------------------------------------------------------------------
    metric_order = [
        specification["key"]
        for specification in CORE_METRICS
    ]
    short_metric_labels = [
        "Under\npressure",
        "Ground\nshare",
        "High\nshare",
        "Mean\nxPass",
        "Execution",
        "Success\nvalue",
        "Failure\ncost",
        "Net\nvalue",
    ]

    matrix = (
        standardized.pivot(
            index="cohort",
            columns="metric",
            values="standardized_value",
        )
        .loc[COHORT_ORDER, metric_order]
        .to_numpy(dtype=np.float64)
    )

    image = ax_a.imshow(
        matrix,
        aspect="auto",
        cmap=profile_diverging_cmap(),
        vmin=-1.5,
        vmax=1.5,
    )

    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels(short_team_labels)
    ax_a.set_xticks(np.arange(len(metric_order)))
    ax_a.set_xticklabels(
        short_metric_labels,
        rotation=0,
        ha="center",
    )
    ax_a.tick_params(axis="both", length=0)

    # Subtle visual grouping of the eight dimensions.
    for separator in [2.5, 4.5]:
        ax_a.axvline(
            separator,
            color="white",
            linewidth=1.25,
            zorder=4,
        )

    group_labels = [
        (1.0, "Passing structure"),
        (3.5, "Difficulty & execution"),
        (6.0, "Risk–reward value"),
    ]
    for x_position, label in group_labels:
        ax_a.text(
            x_position,
            1.11,
            label,
            transform=ax_a.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=STYLE.group_label_size,
            fontweight="bold",
            color=STYLE.dark_slate,
        )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            text_color = (
                "white"
                if abs(value) >= 0.90
                else STYLE.dark_slate
            )
            ax_a.text(
                column_index,
                row_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=STYLE.heatmap_value_size,
                color=text_color,
                zorder=5,
            )

    ax_a.set_title(
        "Core pass risk–reward profiles",
        loc="left",
        pad=31,
    )
    for spine in ax_a.spines.values():
        spine.set_visible(False)

    colorbar = figure.colorbar(
        image,
        ax=ax_a,
        fraction=0.018,
        pad=0.018,
        aspect=18,
    )
    colorbar.ax.tick_params(
        labelsize=STYLE.tick_label_size - 0.3
    )
    colorbar.set_label(
        "Relative position (z score)",
        fontsize=STYLE.axis_label_size,
    )
    add_panel_label(ax_a, "A", x=-0.095, y=1.29)

    # ------------------------------------------------------------------
    # Panel B: centered expected net value.
    # ------------------------------------------------------------------
    net_value = (
        core_profile.loc[
            core_profile["metric"].eq(
                "centered_expected_net_value_per100"
            )
        ]
        .set_index("cohort")
        .loc[COHORT_ORDER]
        .reset_index()
    )

    net_bounds: list[float] = [0.0]

    for index, row in net_value.iterrows():
        color = team_colors[row["cohort"]]
        net_bounds.extend(
            [
                float(row["ci_lower"]),
                float(row["ci_upper"]),
            ]
        )
        ax_b.errorbar(
            float(row["estimate"]),
            y_positions[index],
            xerr=np.array(
                [
                    [
                        float(row["estimate"])
                        - float(row["ci_lower"])
                    ],
                    [
                        float(row["ci_upper"])
                        - float(row["estimate"])
                    ],
                ]
            ),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=STYLE.marker_size,
            markeredgecolor="white",
            markeredgewidth=0.65,
            elinewidth=STYLE.error_line_width,
            capsize=STYLE.cap_size,
            zorder=3,
        )

    net_min = min(net_bounds)
    net_max = max(net_bounds)
    net_span = max(net_max - net_min, 0.35)
    ax_b.set_xlim(
        net_min - 0.16 * net_span,
        net_max + 0.31 * net_span,
    )

    for index, row in net_value.iterrows():
        annotate_horizontal_estimate(
            ax_b,
            float(row["estimate"]),
            float(y_positions[index]),
            text=f"{float(row['estimate']):.2f}",
            pad_fraction=0.025,
        )

    ax_b.axvline(
        0.0,
        color=STYLE.light_gray,
        linewidth=STYLE.reference_line_width,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax_b.set_yticks(y_positions)
    ax_b.set_yticklabels(short_team_labels)
    ax_b.invert_yaxis()
    ax_b.set_xlabel(
        "Centered expected net value\nper 100 passes"
    )
    ax_b.set_title(
        "Expected net value",
        loc="left",
        pad=5,
    )
    clean_axis(ax_b, grid_axis="x")
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(axis="y", length=0)
    add_panel_label(ax_b, "B", x=-0.31)

    # ------------------------------------------------------------------
    # Panel C: Ground / Low / High structure.
    # ------------------------------------------------------------------
    height_wide = (
        pass_height_structure.pivot(
            index="cohort",
            columns="pass_height",
            values="share",
        )
        .loc[COHORT_ORDER, HEIGHT_ORDER]
    )

    left_positions = np.zeros(len(COHORT_ORDER))

    for height in HEIGHT_ORDER:
        values = height_wide[height].to_numpy(
            dtype=np.float64
        )
        bars = ax_c.barh(
            y_positions,
            values,
            left=left_positions,
            color=height_colors[height],
            edgecolor="white",
            linewidth=0.45,
            height=0.62,
            label=height.replace(" Pass", ""),
            zorder=2,
        )

        for team_index, (bar, value) in enumerate(
            zip(bars, values)
        ):
            if value >= 0.08:
                text_color = (
                    STYLE.dark_slate
                    if height == "Low Pass"
                    else "white"
                )
                ax_c.text(
                    left_positions[team_index]
                    + value / 2.0,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{value * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=STYLE.annotation_size,
                    color=text_color,
                    fontweight="bold",
                    zorder=4,
                )

        left_positions += values

    ax_c.set_yticks(y_positions)
    ax_c.set_yticklabels(short_team_labels)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0.0, 1.0)
    ax_c.set_xlabel("Share of passes")
    ax_c.set_title(
        "Pass-height structure",
        loc="left",
        pad=22,
    )
    clean_axis(ax_c, grid_axis="x")
    ax_c.spines["left"].set_visible(False)
    ax_c.tick_params(axis="y", length=0)
    ax_c.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.65,
        borderaxespad=0.0,
    )
    add_panel_label(ax_c, "C", x=-0.31, y=1.17)

    # ------------------------------------------------------------------
    # Panel D: pressure-related net-value change.
    # ------------------------------------------------------------------
    pressure_net = (
        pressure_effects.loc[
            pressure_effects["metric"].eq(
                "centered_expected_net_value_per100"
            )
        ]
        .set_index("cohort")
        .loc[COHORT_ORDER]
        .reset_index()
    )

    pressure_bounds: list[float] = [0.0]

    for index, row in pressure_net.iterrows():
        color = team_colors[row["cohort"]]
        pressure_bounds.extend(
            [
                float(row["ci_lower"]),
                float(row["ci_upper"]),
            ]
        )
        ax_d.errorbar(
            float(row["estimate"]),
            y_positions[index],
            xerr=np.array(
                [
                    [
                        float(row["estimate"])
                        - float(row["ci_lower"])
                    ],
                    [
                        float(row["ci_upper"])
                        - float(row["estimate"])
                    ],
                ]
            ),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=STYLE.marker_size,
            markeredgecolor="white",
            markeredgewidth=0.65,
            elinewidth=STYLE.error_line_width,
            capsize=STYLE.cap_size,
            zorder=3,
        )

    pressure_min = min(pressure_bounds)
    pressure_max = max(pressure_bounds)
    pressure_span = max(
        pressure_max - pressure_min,
        0.6,
    )
    ax_d.set_xlim(
        pressure_min - 0.12 * pressure_span,
        pressure_max + 0.31 * pressure_span,
    )

    for index, row in pressure_net.iterrows():
        annotate_horizontal_estimate(
            ax_d,
            float(row["estimate"]),
            float(y_positions[index]),
            text=f"{float(row['estimate']):.2f}",
            pad_fraction=0.023,
        )

    ax_d.axvline(
        0.0,
        color=STYLE.light_gray,
        linewidth=STYLE.reference_line_width,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax_d.set_yticks(y_positions)
    ax_d.set_yticklabels(short_team_labels)
    ax_d.invert_yaxis()
    ax_d.set_xlabel(
        "Change in centered expected\nnet value / 100"
    )
    ax_d.set_title(
        "Pressure-related net-value change",
        loc="left",
        pad=5,
    )
    clean_axis(ax_d, grid_axis="x")
    ax_d.spines["left"].set_visible(False)
    ax_d.tick_params(axis="y", length=0)
    add_panel_label(ax_d, "D", x=-0.31)

    save_figure_all_formats(
        figure,
        output_dir,
        "Figure_3_three_team_application",
    )
    plt.close(figure)


def build_results_text(
    core_profile: pd.DataFrame,
    pressure_effects: pd.DataFrame,
) -> str:
    wide = (
        core_profile.pivot(
            index="cohort",
            columns="metric",
            values="estimate",
        )
        .loc[COHORT_ORDER]
    )

    pressure_net = (
        pressure_effects.loc[
            pressure_effects["metric"].eq(
                "centered_expected_net_value_per100"
            )
        ]
        .set_index("cohort")
        .loc[COHORT_ORDER]
    )

    paragraphs = [
        "THREE-TEAM ILLUSTRATIVE APPLICATION",
        "",
        (
            "The three prespecified team-season cohorts were used "
            "to illustrate the explanatory value of the locked pass "
            "risk–reward framework rather than to construct a "
            "general taxonomy of team playing styles."
        ),
        "",
    ]

    for cohort in COHORT_ORDER:
        row = wide.loc[cohort]
        pressure_row = pressure_net.loc[cohort]

        paragraphs.append(
            (
                f"{cohort} showed an under-pressure pass share of "
                f"{row['under_pressure_share']:.3f}, a ground-pass "
                f"share of {row['ground_pass_share']:.3f}, and a "
                f"high-pass share of {row['high_pass_share']:.3f}. "
                f"Mean xPass was {row['mean_xpass']:.3f}, execution "
                f"residual was {row['execution_residual']:.3f}, "
                f"expected success value was "
                f"{row['expected_success_value_per100']:.3f} per "
                f"100 passes, expected failure cost was "
                f"{row['expected_failure_cost_per100']:.3f} per "
                f"100 passes, and centered expected net value was "
                f"{row['centered_expected_net_value_per100']:.3f} "
                f"per 100 passes. The under-pressure minus "
                f"not-under-pressure change in centered expected "
                f"net value was {pressure_row['estimate']:.3f} "
                f"(95% CI {pressure_row['ci_lower']:.3f} to "
                f"{pressure_row['ci_upper']:.3f})."
            )
        )

    paragraphs.extend(
        [
            "",
            (
                "The standardized profile panel is intended only "
                "as a compact visualization across the three "
                "selected cohorts. Higher standardized values do "
                "not uniformly indicate better performance because "
                "some dimensions represent usage or failure cost."
            ),
        ]
    )

    return "\n".join(paragraphs)



def main() -> None:
    print(f"Module 06 script version: {SCRIPT_VERSION}")
    args = parse_args()

    script_path = Path(__file__).resolve()
    style_path = script_path.with_name(
        "publication_style.py"
    )

    required_paths = [
        args.pass_value_csv,
        args.pass_metadata_csv,
        style_path,
    ]
    if args.cohort_manifest_csv is not None:
        required_paths.append(args.cohort_manifest_csv)
    elif args.matches_csv is not None:
        required_paths.append(args.matches_csv)
    else:
        raise ValueError(
            "Provide either --cohort_manifest_csv or --matches_csv."
        )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    if args.bootstrap_reps < 200:
        raise ValueError(
            "Use at least 200 match-cluster bootstrap repetitions."
        )

    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, args.overwrite)
    configure_matplotlib()

    if args.cohort_manifest_csv is not None:
        manifest = load_user_manifest(
            args.cohort_manifest_csv.resolve()
        )
        manifest_source = "user-supplied manifest"
    else:
        manifest = identify_cohort_matches(
            args.matches_csv.resolve()
        )
        manifest_source = "automatic match identification"

    manifest.to_csv(
        output_dir / "three_team_match_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_match_ids = set(
        manifest["match_id"].map(normalize_id)
    )

    (
        pass_values,
        pass_value_columns,
        global_net_mean,
        global_net_rows,
        pass_value_load_audit,
    ) = load_selected_pass_values(
        args.pass_value_csv.resolve(),
        selected_match_ids,
        args.chunksize,
    )

    (
        metadata,
        metadata_columns,
        metadata_load_audit,
    ) = load_selected_metadata(
        args.pass_metadata_csv.resolve(),
        selected_match_ids,
        args.chunksize,
    )

    events, merge_audit = merge_and_filter_own_team_events(
        pass_values,
        pass_value_columns,
        metadata,
        metadata_columns,
        manifest,
        global_net_mean,
    )

    # Save a compact, deterministic event-level audit dataset.
    event_output_columns = [
        "event_id",
        "match_id",
        "cohort",
        "pressure_status",
        "pass_height_group",
        "xpass_success",
        "p_success_main_oof",
        "execution_residual",
        "expected_success_value",
        "expected_failure_cost",
        "expected_net_value",
        "expected_net_value_centered",
    ]
    event_output = events[
        event_output_columns
    ].sort_values(
        ["cohort", "match_id", "event_id"],
        kind="mergesort",
    )
    event_output.to_csv(
        output_dir / "three_team_event_cohort.csv.gz",
        index=False,
        encoding="utf-8",
        compression={
            "method": "gzip",
            "compresslevel": 6,
            "mtime": 0,
        },
    )

    core_profile = build_core_profile_table(
        events,
        args.bootstrap_reps,
        args.seed,
    )
    pass_height_structure = build_pass_height_structure(
        events,
        args.bootstrap_reps,
        args.seed + 1,
    )
    pressure_effects = pressure_effects_by_team(
        events,
        args.bootstrap_reps,
        args.seed + 2,
    )
    standardized = standardized_profile_table(
        core_profile
    )
    wide_profile = wide_profile_table(
        core_profile
    )

    core_profile.to_csv(
        output_dir / "three_team_core_profile_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    wide_profile.to_csv(
        output_dir / "three_team_core_profile_wide.csv",
        index=False,
        encoding="utf-8-sig",
    )
    standardized.to_csv(
        output_dir / "three_team_core_profile_standardized.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pressure_effects.to_csv(
        output_dir / "three_team_pressure_effects.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pass_height_structure.to_csv(
        output_dir / "three_team_pass_height_structure.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_figure_3(
        core_profile,
        standardized,
        pass_height_structure,
        pressure_effects,
        output_dir,
    )

    results_text = build_results_text(
        core_profile,
        pressure_effects,
    )
    (
        output_dir / "module06_results_text.txt"
    ).write_text(
        results_text,
        encoding="utf-8",
    )

    captions = """Figure 3. Illustrative application of the locked H=3 pass risk–reward framework to three prespecified team-season cohorts. (A) Standardized values across eight grouped profile dimensions covering passing structure, difficulty and execution, and risk–reward value. Standardization was performed only across the three selected cohorts; positive and negative values denote positions above and below the three-cohort mean and do not uniformly indicate better or worse performance. (B) Centered expected net value per 100 passes, with percentile 95% confidence intervals from match-cluster bootstrap resampling. The zero reference denotes the full-sample mean. (C) Within-cohort shares of Ground, Low, and High passes; labels show rounded percentages for segments representing at least 8% of passes. (D) Under-pressure minus not-under-pressure changes in centered expected net value per 100 passes, with percentile 95% confidence intervals from match-cluster bootstrap resampling. Panels B-D provide within-cohort descriptive estimates and do not by themselves test pairwise differences between cohorts.
"""
    (
        output_dir / "module06_figure_captions.txt"
    ).write_text(
        captions,
        encoding="utf-8",
    )

    cohort_counts = (
        events.groupby("cohort")
        .agg(
            matches=("match_id", "nunique"),
            passes=("event_id", "size"),
            unknown_height=(
                "pass_height_group",
                lambda values: int(
                    np.sum(values == "Unknown")
                ),
            ),
        )
        .reset_index()
    )

    manifest_counts = (
        manifest.groupby("cohort")["match_id"]
        .nunique()
        .to_dict()
    )

    checks = {
        "all_three_cohorts_present": (
            set(events["cohort"].unique())
            == set(COHORT_ORDER)
        ),
        "expected_total_matches": (
            events["match_id"].nunique()
            == EXPECTED_MATCHES
        ),
        "expected_total_passes": (
            len(events) == EXPECTED_PASSES
        ),
        "expected_manifest_match_counts": all(
            int(manifest_counts.get(cohort, 0))
            == int(
                COHORT_SPECIFICATIONS[cohort][
                    "expected_matches"
                ]
            )
            for cohort in COHORT_ORDER
        ),
        "all_pass_height_categories_present": (
            set(HEIGHT_ORDER)
            <= set(events["pass_height_group"].unique())
        ),
        "pressure_binary_after_normalization": (
            set(events["pressure_status"].unique())
            <= set(PRESSURE_ORDER)
        ),
        "all_analysis_metrics_complete": (
            not events[
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
        "eight_core_metrics_created": (
            core_profile["metric"].nunique()
            == len(CORE_METRICS)
        ),
        "no_duplicate_pass_value_event_ids": (
            pass_value_load_audit[
                "selected_pass_value_duplicate_event_id_count"
            ]
            == 0
        ),
        "no_duplicate_metadata_event_ids": (
            metadata_load_audit[
                "selected_metadata_duplicate_event_id_count"
            ]
            == 0
        ),
        "no_unmatched_event_rows": (
            merge_audit["unmatched_event_rows"] == 0
        ),
        "no_matches_with_zero_retained_passes": (
            merge_audit[
                "matches_with_zero_retained_passes"
            ]
            == 0
        ),
    }

    output_hashes = collect_file_hashes(
        output_dir,
        excluded_names={
            "module06_audit.json",
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
            "Compressed illustrative application to three "
            "prespecified team-season cohorts."
        ),
        "manifest_source": manifest_source,
        "cohort_counts": cohort_counts.to_dict(
            orient="records"
        ),
        "total_matches": int(
            events["match_id"].nunique()
        ),
        "total_passes": int(len(events)),
        "global_centering_reference": {
            "formal_rows": int(global_net_rows),
            "expected_net_value_per_pass": (
                global_net_mean
            ),
            "expected_net_value_per100": (
                global_net_mean * 100.0
            ),
        },
        "bootstrap": {
            "unit": "match_id",
            "repetitions": args.bootstrap_reps,
            "seed": args.seed,
        },
        "pass_value_load_audit": pass_value_load_audit,
        "metadata_load_audit": metadata_load_audit,
        "merge_and_filter_audit": merge_audit,
        "checks": checks,
        "input_hashes": {
            "pass_value_csv_sha256": sha256_file(
                args.pass_value_csv.resolve()
            ),
            "pass_metadata_csv_sha256": sha256_file(
                args.pass_metadata_csv.resolve()
            ),
            "matches_csv_sha256": (
                sha256_file(args.matches_csv.resolve())
                if args.matches_csv is not None
                else None
            ),
            "cohort_manifest_csv_sha256": (
                sha256_file(args.cohort_manifest_csv.resolve())
                if args.cohort_manifest_csv is not None
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
                "module06_audit.json",
                "SHA256SUMS.txt",
            ]
        ),
    }

    audit_path = output_dir / "module06_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_sha256sums(output_dir)

    print("=" * 100)
    print("Final-audit Module 06 completed")
    print("=" * 100)
    print(f"Matches: {events['match_id'].nunique():,}")
    print(f"Passes: {len(events):,}")
    print(f"Core metrics: {core_profile['metric'].nunique()}")
    print(f"Status: {audit['status']}")
    print(f"Outputs: {output_dir}")

if __name__ == "__main__":
    main()
