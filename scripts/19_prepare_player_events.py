#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
19_prepare_player_events.py

Extract and merge player/position metadata for the locked three-team
event cohort without loading or exporting the full 3.4-million-pass
analytical dataset.

Inputs
------
1. three_team_event_cohort.csv.gz
2. passes_preprocessed_with_open_play_flags.csv

Outputs
-------
1. three_team_player_event_data.csv.gz
2. three_team_player_event_audit.json

The merge is performed using event ID and match ID and is required to be
one-to-one. The script does not modify xPass, PCTV, or any value estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


VERSION = "2026-07-29-player-event-preparation-v1"

REQUIRED_COHORT_COLUMNS = {
    "event_id",
    "match_id",
    "cohort",
    "xpass_success",
    "p_success_main_oof",
    "execution_residual",
    "expected_success_value",
    "expected_failure_cost",
    "expected_net_value",
    "expected_net_value_centered",
}

REQUIRED_METADATA_COLUMNS = {
    "id",
    "match_id",
    "team",
    "player",
    "position",
}

OPTIONAL_METADATA_COLUMNS = [
    "minute",
    "second",
    "pass_length",
    "pass_angle",
    "pass_height",
    "under_pressure",
    "play_pattern",
    "pass_type",
    "pass_body_part",
    "pass_technique",
    "pass_recipient",
    "possession",
    "possession_team",
    "competition_id",
    "season_id",
    "competition_name",
    "season_name",
    "match_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "start_x_model",
    "start_y_model",
    "end_x_model",
    "end_y_model",
    "primary_open_play",
    "strict_open_play",
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
            "Prepare the locked three-team player-event table by merging "
            "player and position metadata onto the final event cohort."
        )
    )
    parser.add_argument(
        "--event_cohort",
        required=True,
        type=Path,
        help="three_team_event_cohort.csv.gz",
    )
    parser.add_argument(
        "--metadata_csv",
        required=True,
        type=Path,
        help="passes_preprocessed_with_open_play_flags.csv",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--expected_rows", type=int, default=54_795)
    parser.add_argument("--expected_matches", type=int, default=110)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--version", action="version", version=VERSION)
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
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
        **kwargs,
    )


def main() -> None:
    args = parse_args()

    cohort_path = args.event_cohort.resolve()
    metadata_path = args.metadata_csv.resolve()
    output_dir = args.output_dir.resolve()

    for path in (cohort_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "three_team_player_event_data.csv.gz"
    audit_json = output_dir / "three_team_player_event_audit.json"

    existing = [p for p in (output_csv, audit_json) if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Outputs already exist. Use --overwrite only when replacing them:\n"
            + "\n".join(str(p) for p in existing)
        )
    for path in existing:
        path.unlink()

    cohort = read_table(cohort_path)
    missing_cohort = sorted(REQUIRED_COHORT_COLUMNS - set(cohort.columns))
    if missing_cohort:
        raise RuntimeError(
            "Event cohort is missing required columns: "
            + ", ".join(missing_cohort)
        )

    cohort["event_id"] = cohort["event_id"].map(normalize_id)
    cohort["match_id"] = cohort["match_id"].map(normalize_id)

    if cohort.duplicated(["event_id", "match_id"]).any():
        raise RuntimeError(
            "Duplicate event-match keys were found in the event cohort."
        )

    event_ids = set(cohort["event_id"])
    match_ids = set(cohort["match_id"])

    header = pd.read_csv(
        metadata_path,
        encoding="utf-8-sig",
        nrows=0,
    )
    available_columns = header.columns.tolist()

    missing_metadata = sorted(
        REQUIRED_METADATA_COLUMNS - set(available_columns)
    )
    if missing_metadata:
        raise RuntimeError(
            "Metadata file is missing required columns: "
            + ", ".join(missing_metadata)
        )

    usecols = list(REQUIRED_METADATA_COLUMNS) + [
        column
        for column in OPTIONAL_METADATA_COLUMNS
        if column in available_columns
        and column not in REQUIRED_METADATA_COLUMNS
    ]

    selected_parts: list[pd.DataFrame] = []
    rows_scanned = 0

    for chunk in pd.read_csv(
        metadata_path,
        encoding="utf-8-sig",
        usecols=usecols,
        chunksize=args.chunksize,
        low_memory=False,
    ):
        rows_scanned += len(chunk)
        chunk["_event_id_norm"] = chunk["id"].map(normalize_id)
        chunk["_match_id_norm"] = chunk["match_id"].map(normalize_id)

        mask = (
            chunk["_event_id_norm"].isin(event_ids)
            & chunk["_match_id_norm"].isin(match_ids)
        )
        selected = chunk.loc[mask].copy()
        if not selected.empty:
            selected_parts.append(selected)

        if rows_scanned % 1_000_000 < args.chunksize:
            print(f"[scan] {rows_scanned:,} metadata rows")

    if not selected_parts:
        raise RuntimeError(
            "No matching metadata rows were found for the three-team cohort."
        )

    metadata = pd.concat(selected_parts, ignore_index=True)

    duplicate_metadata = metadata.duplicated(
        ["_event_id_norm", "_match_id_norm"],
        keep=False,
    )
    duplicate_metadata_rows = int(duplicate_metadata.sum())
    if duplicate_metadata_rows:
        raise RuntimeError(
            "Duplicate event-match keys were found in selected metadata: "
            f"{duplicate_metadata_rows} rows."
        )

    metadata = metadata.rename(
        columns={
            "id": "metadata_event_id",
            "match_id": "metadata_match_id",
        }
    )

    merged = cohort.merge(
        metadata,
        left_on=["event_id", "match_id"],
        right_on=["_event_id_norm", "_match_id_norm"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    matched_rows = int((merged["_merge"] == "both").sum())
    unmatched_rows = int((merged["_merge"] == "left_only").sum())

    if unmatched_rows:
        examples = merged.loc[
            merged["_merge"].eq("left_only"),
            ["event_id", "match_id"],
        ].head(10)
        raise RuntimeError(
            f"{unmatched_rows:,} cohort events did not match metadata.\n"
            + examples.to_string(index=False)
        )

    merged = merged.drop(
        columns=[
            "_merge",
            "_event_id_norm",
            "_match_id_norm",
            "metadata_event_id",
            "metadata_match_id",
        ],
        errors="ignore",
    )

    row_count = int(len(merged))
    match_count = int(merged["match_id"].nunique())
    cohort_count = int(merged["cohort"].nunique())
    missing_player_rows = int(
        merged["player"].astype("string").fillna("").str.strip().eq("").sum()
    )
    missing_position_rows = int(
        merged["position"].astype("string").fillna("").str.strip().eq("").sum()
    )

    checks = {
        "expected_rows": row_count == args.expected_rows,
        "expected_matches": match_count == args.expected_matches,
        "three_cohorts": cohort_count == 3,
        "all_rows_matched": unmatched_rows == 0,
        "event_match_keys_unique": not merged.duplicated(
            ["event_id", "match_id"]
        ).any(),
        "no_missing_player": missing_player_rows == 0,
        "no_missing_position": missing_position_rows == 0,
        "all_model_metrics_complete": not merged[
            [
                "xpass_success",
                "p_success_main_oof",
                "execution_residual",
                "expected_success_value",
                "expected_failure_cost",
                "expected_net_value",
                "expected_net_value_centered",
            ]
        ].isna().any().any(),
    }

    if not all(checks.values()):
        raise RuntimeError(
            "Player-event preparation audit failed:\n"
            + json.dumps(checks, ensure_ascii=False, indent=2, default=json_default)
        )

    merged.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
        compression="gzip",
    )

    audit = {
        "status": "PASS",
        "version": VERSION,
        "inputs": {
            "event_cohort": str(cohort_path),
            "metadata_csv": str(metadata_path),
            "event_cohort_sha256": sha256_file(cohort_path),
            "metadata_csv_sha256": sha256_file(metadata_path),
        },
        "counts": {
            "metadata_rows_scanned": rows_scanned,
            "selected_metadata_rows": int(len(metadata)),
            "matched_rows": matched_rows,
            "unmatched_rows": unmatched_rows,
            "formal_rows": row_count,
            "matches": match_count,
            "cohorts": cohort_count,
            "players": int(merged["player"].nunique()),
            "positions": int(merged["position"].nunique()),
            "missing_player_rows": missing_player_rows,
            "missing_position_rows": missing_position_rows,
        },
        "checks": checks,
        "columns": merged.columns.tolist(),
        "outputs": {
            "player_event_data": str(output_csv),
            "audit": str(audit_json),
        },
    }

    audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    print("=" * 88)
    print("Three-team player-event preparation completed")
    print("=" * 88)
    print(f"Rows: {row_count:,}")
    print(f"Matches: {match_count:,}")
    print(f"Players: {merged['player'].nunique():,}")
    print(f"Positions: {merged['position'].nunique():,}")
    print(f"Output: {output_csv}")
    print(f"Audit: {audit_json}")


if __name__ == "__main__":
    main()
