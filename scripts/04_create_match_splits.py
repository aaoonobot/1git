#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create one deterministic match-level split manifest for xPass modeling.

Primary split:
- training:    70%
- tuning:      10%
- calibration: 10%
- test:        10%

Assignment is performed at match level, never at pass-event level. Matches are
stratified by competition gender and match-date quantile band, then ordered by
a stable SHA-256 key. An optional focal-team match list can be reserved as a
completely separate "focal_holdout" cohort before the four-way split.

The script does not train a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def to_json_native(value):
    """Recursively convert NumPy/Pandas scalar values to JSON-native types."""
    if isinstance(value, dict):
        return {str(key): to_json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_native(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if pd.isna(value):
        return None
    return value


SPLIT_ORDER = ["training", "tuning", "calibration", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic match-level xPass split manifest."
    )
    parser.add_argument(
        "--match_audit_csv",
        type=Path,
        required=True,
        help="match_level_target_audit.csv produced by 03_audit_xpass_target.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--tuning_fraction", type=float, default=0.10)
    parser.add_argument("--calibration_fraction", type=float, default=0.10)
    parser.add_argument("--test_fraction", type=float, default=0.10)
    parser.add_argument(
        "--time_bands",
        type=int,
        default=8,
        help="Number of match-date quantile bands used for stratification.",
    )
    parser.add_argument(
        "--expected_matches",
        type=int,
        default=3926,
    )
    parser.add_argument(
        "--focal_match_ids_csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing match_id values to reserve as "
            "focal_holdout before model-development splitting."
        ),
    )
    parser.add_argument(
        "--focal_match_col",
        default="match_id",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_match_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except Exception:
            pass
    return text


def stable_hash(seed: int, match_id: str) -> str:
    payload = f"{seed}|{match_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def allocate_counts(n: int, proportions: dict[str, float]) -> dict[str, int]:
    """
    Largest-remainder allocation. Counts sum exactly to n.
    """
    raw = {name: n * proportions[name] for name in SPLIT_ORDER}
    allocated = {name: math.floor(raw[name]) for name in SPLIT_ORDER}
    remainder = n - sum(allocated.values())

    ranked = sorted(
        SPLIT_ORDER,
        key=lambda name: (
            raw[name] - allocated[name],
            -SPLIT_ORDER.index(name),
        ),
        reverse=True,
    )

    for name in ranked[:remainder]:
        allocated[name] += 1

    return allocated


def load_focal_ids(path: Path | None, column: str) -> set[str]:
    if path is None:
        return set()

    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if column not in frame.columns:
        raise RuntimeError(
            f"Focal match file does not contain column '{column}'."
        )

    ids = {
        normalize_match_id(value)
        for value in frame[column]
        if normalize_match_id(value)
    }
    return ids


def main() -> None:
    args = parse_args()

    input_csv = args.match_audit_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    outputs = [
        output_dir / "xpass_match_split_manifest.csv",
        output_dir / "xpass_split_summary.csv",
        output_dir / "xpass_split_gender_summary.csv",
        output_dir / "xpass_split_year_summary.csv",
        output_dir / "xpass_split_stratum_summary.csv",
        output_dir / "xpass_split_config.json",
        output_dir / "xpass_split_validation.json",
    ]

    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Split outputs already exist. Use --overwrite only after review:\n"
            + "\n".join(str(path) for path in existing)
        )

    for path in existing:
        path.unlink()

    proportions = {
        "training": args.train_fraction,
        "tuning": args.tuning_fraction,
        "calibration": args.calibration_fraction,
        "test": args.test_fraction,
    }

    if not math.isclose(sum(proportions.values()), 1.0, abs_tol=1e-12):
        raise ValueError(
            f"Split fractions must sum to 1.0, got {sum(proportions.values())}"
        )

    frame = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        "match_id",
        "primary_passes",
        "primary_successes",
        "primary_failures",
        "primary_failure_rate",
        "competition_gender",
        "competition_name",
        "season_name",
        "match_date",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    frame["match_id"] = frame["match_id"].map(normalize_match_id)

    if frame["match_id"].eq("").any():
        raise RuntimeError("Missing match_id values are present.")

    if frame["match_id"].duplicated().any():
        duplicates = frame.loc[
            frame["match_id"].duplicated(keep=False),
            "match_id",
        ].tolist()
        raise RuntimeError(
            f"Duplicate match_id values in match audit: {duplicates[:20]}"
        )

    if len(frame) != args.expected_matches:
        raise RuntimeError(
            f"Expected {args.expected_matches:,} matches, found {len(frame):,}."
        )

    numeric_columns = [
        "primary_passes",
        "primary_successes",
        "primary_failures",
        "primary_failure_rate",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    invalid_totals = frame[
        frame["primary_passes"]
        != frame["primary_successes"] + frame["primary_failures"]
    ]
    if not invalid_totals.empty:
        raise RuntimeError(
            "Some matches have primary_passes != successes + failures."
        )

    frame["match_date"] = pd.to_datetime(
        frame["match_date"],
        errors="raise",
    )
    frame["match_year"] = frame["match_date"].dt.year

    gender = (
        frame["competition_gender"]
        .astype("string")
        .fillna("unknown")
        .str.strip()
        .str.lower()
        .replace("", "unknown")
    )
    frame["competition_gender"] = gender

    # Date quantile bands use no outcome information.
    date_rank = frame["match_date"].rank(
        method="first",
        pct=True,
    )
    q = max(1, int(args.time_bands))
    frame["time_band"] = (
        ((date_rank - 1e-12) * q)
        .astype(int)
        .clip(0, q - 1)
    )
    frame["stratum"] = (
        frame["competition_gender"].astype(str)
        + "|T"
        + frame["time_band"].astype(str)
    )

    focal_ids = load_focal_ids(
        args.focal_match_ids_csv,
        args.focal_match_col,
    )

    unknown_focal_ids = focal_ids - set(frame["match_id"])
    if unknown_focal_ids:
        raise RuntimeError(
            "Some focal match IDs are absent from the 3,926-match audit: "
            + ", ".join(sorted(unknown_focal_ids)[:20])
        )

    frame["split"] = ""
    frame["stable_hash"] = frame["match_id"].map(
        lambda value: stable_hash(args.seed, value)
    )

    if focal_ids:
        frame.loc[
            frame["match_id"].isin(focal_ids),
            "split",
        ] = "focal_holdout"

    development = frame.loc[frame["split"].eq("")].copy()

    for stratum, group in development.groupby("stratum", sort=True):
        group = group.sort_values(
            ["stable_hash", "match_id"],
            kind="stable",
        )
        counts = allocate_counts(len(group), proportions)

        start = 0
        for split_name in SPLIT_ORDER:
            end = start + counts[split_name]
            assigned_indices = group.index[start:end]
            frame.loc[assigned_indices, "split"] = split_name
            start = end

        if start != len(group):
            raise RuntimeError(
                f"Internal allocation error for stratum {stratum}."
            )

    if frame["split"].eq("").any():
        raise RuntimeError("Some matches were not assigned to a split.")

    # Exact one-row-per-match manifest.
    manifest_columns = [
        "match_id",
        "split",
        "stable_hash",
        "stratum",
        "competition_gender",
        "time_band",
        "match_year",
        "match_date",
        "competition_name",
        "season_name",
        "primary_passes",
        "primary_successes",
        "primary_failures",
        "primary_failure_rate",
        "strict_passes",
        "strict_successes",
        "strict_failures",
        "strict_failure_rate",
    ]
    manifest_columns = [
        column for column in manifest_columns if column in frame.columns
    ]

    manifest = frame[manifest_columns].sort_values(
        ["split", "match_date", "match_id"]
    )
    manifest.to_csv(
        output_dir / "xpass_match_split_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    def weighted_failure_rate(group: pd.DataFrame) -> float:
        passes = group["primary_passes"].sum()
        return (
            group["primary_failures"].sum() / passes
            if passes
            else float("nan")
        )

    split_summary_rows = []
    for split_name, group in frame.groupby("split", sort=False):
        split_summary_rows.append(
            {
                "split": split_name,
                "matches": group["match_id"].nunique(),
                "match_percentage": len(group) / len(frame) * 100.0,
                "primary_passes": int(group["primary_passes"].sum()),
                "primary_successes": int(
                    group["primary_successes"].sum()
                ),
                "primary_failures": int(
                    group["primary_failures"].sum()
                ),
                "primary_failure_rate": weighted_failure_rate(group),
                "first_match_date": group["match_date"].min().date(),
                "last_match_date": group["match_date"].max().date(),
                "unique_competitions": group[
                    "competition_name"
                ].nunique(),
                "unique_seasons": group["season_name"].nunique(),
            }
        )

    split_summary = pd.DataFrame(split_summary_rows)
    split_summary.to_csv(
        output_dir / "xpass_split_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gender_summary = (
        frame.groupby(["split", "competition_gender"], dropna=False)
        .agg(
            matches=("match_id", "nunique"),
            primary_passes=("primary_passes", "sum"),
            primary_failures=("primary_failures", "sum"),
        )
        .reset_index()
    )
    gender_summary["primary_failure_rate"] = (
        gender_summary["primary_failures"]
        / gender_summary["primary_passes"]
    )
    gender_summary.to_csv(
        output_dir / "xpass_split_gender_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    year_summary = (
        frame.groupby(["split", "match_year"], dropna=False)
        .agg(
            matches=("match_id", "nunique"),
            primary_passes=("primary_passes", "sum"),
            primary_failures=("primary_failures", "sum"),
        )
        .reset_index()
    )
    year_summary["primary_failure_rate"] = (
        year_summary["primary_failures"]
        / year_summary["primary_passes"]
    )
    year_summary.to_csv(
        output_dir / "xpass_split_year_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stratum_summary = (
        frame.groupby(["split", "stratum"], dropna=False)
        .agg(
            matches=("match_id", "nunique"),
            primary_passes=("primary_passes", "sum"),
            primary_failures=("primary_failures", "sum"),
        )
        .reset_index()
    )
    stratum_summary["primary_failure_rate"] = (
        stratum_summary["primary_failures"]
        / stratum_summary["primary_passes"]
    )
    stratum_summary.to_csv(
        output_dir / "xpass_split_stratum_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    development_frame = frame[
        frame["split"].isin(SPLIT_ORDER)
    ].copy()
    overall_dev_failure_rate = weighted_failure_rate(
        development_frame
    )

    failure_rate_deviations = {}
    for split_name in SPLIT_ORDER:
        group = frame.loc[frame["split"].eq(split_name)]
        rate = weighted_failure_rate(group)
        failure_rate_deviations[split_name] = (
            rate - overall_dev_failure_rate
        )

    split_counts = (
        frame["split"].value_counts().to_dict()
    )

    checks = {
        "expected_total_matches":
            len(frame) == args.expected_matches,
        "unique_match_ids":
            frame["match_id"].nunique() == len(frame),
        "all_matches_assigned_once":
            frame["split"].ne("").all(),
        "training_nonempty":
            split_counts.get("training", 0) > 0,
        "tuning_nonempty":
            split_counts.get("tuning", 0) > 0,
        "calibration_nonempty":
            split_counts.get("calibration", 0) > 0,
        "test_nonempty":
            split_counts.get("test", 0) > 0,
        "focal_count_matches_input":
            split_counts.get("focal_holdout", 0) == len(focal_ids),
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    config = {
        "status": status,
        "seed": args.seed,
        "proportions": proportions,
        "time_bands": q,
        "stratification": (
            "competition_gender crossed with match-date quantile band"
        ),
        "assignment": (
            "stable SHA-256 ordering within each stratum followed by "
            "largest-remainder count allocation"
        ),
        "split_unit": "match_id",
        "focal_match_ids_csv": (
            str(args.focal_match_ids_csv.resolve())
            if args.focal_match_ids_csv is not None
            else None
        ),
        "focal_holdout_matches": len(focal_ids),
        "test_set_policy": (
            "The test split must remain untouched until all model, "
            "feature, and calibration decisions are locked."
        ),
    }

    with (output_dir / "xpass_split_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(to_json_native(config), file, ensure_ascii=False, indent=2)

    validation = {
        "status": status,
        "checks": {key: bool(value) for key, value in checks.items()},
        "total_matches": int(len(frame)),
        "split_match_counts": {
            str(key): int(value)
            for key, value in split_counts.items()
        },
        "development_overall_failure_rate": float(
            overall_dev_failure_rate
        ),
        "split_failure_rate_deviation_from_development_overall": {
            str(key): float(value)
            for key, value in failure_rate_deviations.items()
        },
        "manifest_sha256": hashlib.sha256(
            (output_dir / "xpass_match_split_manifest.csv").read_bytes()
        ).hexdigest(),
    }

    with (output_dir / "xpass_split_validation.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            to_json_native(validation),
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 100)
    print("Deterministic match-level xPass split completed")
    print("=" * 100)
    print(f"Total matches: {len(frame):,}")
    for split_name in SPLIT_ORDER + ["focal_holdout"]:
        if split_name in split_counts:
            print(
                f"{split_name}: "
                f"{split_counts[split_name]:,} matches"
            )
    print(
        "Overall failure rate in the development cohort: "
        f"{overall_dev_failure_rate:.4%}"
    )
    for split_name in SPLIT_ORDER:
        print(
            f"{split_name} failure-rate deviation: "
            f"{failure_rate_deviations[split_name]:+.4%}"
        )
    print(f"Status: {status}")
    print(
        "Match split manifest: "
        f"{output_dir / 'xpass_match_split_manifest.csv'}"
    )
    print(
        "Validation report: "
        f"{output_dir / 'xpass_split_validation.json'}"
    )

    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
