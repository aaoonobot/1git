#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Audit the xPass target before model training.

Main target convention:
- pass_outcome missing/blank -> successful pass (xpass_success = 1)
- pass_outcome non-missing   -> unsuccessful pass (xpass_success = 0)

The script does not yet train a model or write another multi-gigabyte event file.
It produces class-balance, subgroup, missingness, and match-level audit tables
for the primary and strict open-play definitions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


AUDIT_COLUMNS = [
    "id",
    "match_id",
    "competition_id",
    "season_id",
    "competition_name",
    "season_name",
    "match_date",
    "team",
    "player",
    "position",
    "pass_outcome",
    "pass_type",
    "pass_height",
    "pass_body_part",
    "under_pressure",
    "primary_open_play",
    "strict_open_play",
    "start_x_model",
    "start_y_model",
    "end_x_model",
    "end_y_model",
    "pass_angle",
    "pass_length",
]

MODEL_CANDIDATE_COLUMNS = [
    "match_id",
    "competition_id",
    "season_id",
    "match_date",
    "team",
    "player",
    "position",
    "pass_type",
    "pass_height",
    "pass_body_part",
    "under_pressure",
    "start_x_model",
    "start_y_model",
    "end_x_model",
    "end_y_model",
    "pass_angle",
    "pass_length",
]

POST_OUTCOME_OR_EXCLUDED_PREDICTORS = [
    "pass_outcome",
    "pass_aerial_won",
    "pass_miscommunication",
    "pass_recovery",
    "pass_recipient",
    "related_events",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit StatsBomb xPass labels and class balance."
    )
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--competitions_csv",
        type=Path,
        default=None,
        help=(
            "Optional competitions_selected.csv used to attach "
            "competition_gender and country_name."
        ),
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--progress_every_chunks", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def normalize_id_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except Exception:
            pass
    return text


def make_key(comp_id: Any, season_id: Any) -> tuple[str, str]:
    return (
        normalize_id_value(comp_id),
        normalize_id_value(season_id),
    )


def load_competition_map(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if path is None:
        return {}

    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    mapping: dict[tuple[str, str], dict[str, str]] = {}

    for _, row in frame.iterrows():
        key = make_key(row.get("competition_id"), row.get("season_id"))
        mapping[key] = {
            "competition_gender": str(
                row.get("competition_gender", "unknown")
            ).strip() or "unknown",
            "country_name": str(
                row.get("country_name", "")
            ).strip(),
        }

    return mapping


def add_count(counter: Counter, values: pd.Series) -> None:
    counter.update(values.value_counts(dropna=False).to_dict())


def main() -> None:
    args = parse_args()

    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    output_files = [
        output_dir / "xpass_target_audit_summary.json",
        output_dir / "xpass_target_audit_summary.csv",
        output_dir / "pass_outcome_distribution_primary.csv",
        output_dir / "pass_outcome_distribution_strict.csv",
        output_dir / "outcome_by_pass_type_primary.csv",
        output_dir / "outcome_by_pass_height_primary.csv",
        output_dir / "outcome_by_pressure_primary.csv",
        output_dir / "outcome_by_gender_primary.csv",
        output_dir / "match_level_target_audit.csv",
        output_dir / "candidate_feature_missingness_primary.csv",
        output_dir / "xpass_target_definition.json",
    ]

    existing = [path for path in output_files if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Audit outputs already exist. Use --overwrite after review:\n"
            + "\n".join(str(path) for path in existing)
        )
    for path in existing:
        path.unlink()

    header = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        nrows=0,
    ).columns.tolist()

    required = set(AUDIT_COLUMNS)
    missing = sorted(required - set(header))
    if missing:
        raise RuntimeError(f"Required columns are missing: {missing}")

    usecols = [column for column in AUDIT_COLUMNS if column in header]
    competition_map = load_competition_map(args.competitions_csv)

    raw_rows = 0
    primary_rows = 0
    strict_rows = 0
    primary_success = 0
    primary_failure = 0
    strict_success = 0
    strict_failure = 0

    primary_outcomes: Counter = Counter()
    strict_outcomes: Counter = Counter()
    pass_type_cross: Counter = Counter()
    pass_height_cross: Counter = Counter()
    pressure_cross: Counter = Counter()
    gender_cross: Counter = Counter()

    feature_nonmissing: Counter = Counter()
    feature_missing: Counter = Counter()

    match_stats = defaultdict(
        lambda: {
            "primary_passes": 0,
            "primary_successes": 0,
            "primary_failures": 0,
            "strict_passes": 0,
            "strict_successes": 0,
            "strict_failures": 0,
            "competition_gender": "unknown",
            "competition_name": "",
            "season_name": "",
            "match_date": "",
        }
    )

    unexpected_boolean_values = Counter()

    reader = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        usecols=usecols,
        chunksize=args.chunksize,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        raw_rows += len(chunk)

        primary_flag_text = (
            chunk["primary_open_play"]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.lower()
        )
        strict_flag_text = (
            chunk["strict_open_play"]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.lower()
        )

        valid_true = {"true", "1", "1.0"}
        valid_false = {"false", "0", "0.0"}

        primary_mask = primary_flag_text.isin(valid_true)
        strict_mask = strict_flag_text.isin(valid_true)

        for value, count in primary_flag_text.value_counts().items():
            if value not in valid_true | valid_false:
                unexpected_boolean_values[
                    f"primary_open_play:{value}"
                ] += int(count)

        for value, count in strict_flag_text.value_counts().items():
            if value not in valid_true | valid_false:
                unexpected_boolean_values[
                    f"strict_open_play:{value}"
                ] += int(count)

        outcome_text = normalize_text(chunk["pass_outcome"])
        success = outcome_text.eq("")
        failure = ~success
        outcome_label = outcome_text.where(
            outcome_text.ne(""),
            "<SUCCESS_NULL>",
        )

        primary_chunk = chunk.loc[primary_mask].copy()
        strict_chunk = chunk.loc[strict_mask].copy()

        primary_success_mask = success.loc[primary_mask]
        primary_failure_mask = failure.loc[primary_mask]
        strict_success_mask = success.loc[strict_mask]
        strict_failure_mask = failure.loc[strict_mask]

        primary_rows += len(primary_chunk)
        strict_rows += len(strict_chunk)
        primary_success += int(primary_success_mask.sum())
        primary_failure += int(primary_failure_mask.sum())
        strict_success += int(strict_success_mask.sum())
        strict_failure += int(strict_failure_mask.sum())

        add_count(
            primary_outcomes,
            outcome_label.loc[primary_mask],
        )
        add_count(
            strict_outcomes,
            outcome_label.loc[strict_mask],
        )

        primary_type = normalize_text(
            chunk.loc[primary_mask, "pass_type"]
        ).replace("", "<NA>")
        primary_height = normalize_text(
            chunk.loc[primary_mask, "pass_height"]
        ).replace("", "<NA>")
        primary_pressure = normalize_text(
            chunk.loc[primary_mask, "under_pressure"]
        ).replace("", "<NA>")

        primary_binary = pd.Series(
            primary_failure_mask.astype(int).to_numpy(),
            index=primary_type.index,
        )

        pass_type_cross.update(
            pd.DataFrame(
                {
                    "pass_type": primary_type,
                    "failure": primary_binary,
                }
            ).value_counts(dropna=False).to_dict()
        )

        pass_height_cross.update(
            pd.DataFrame(
                {
                    "pass_height": primary_height,
                    "failure": primary_binary,
                }
            ).value_counts(dropna=False).to_dict()
        )

        pressure_cross.update(
            pd.DataFrame(
                {
                    "under_pressure": primary_pressure,
                    "failure": primary_binary,
                }
            ).value_counts(dropna=False).to_dict()
        )

        comp_ids = chunk.loc[primary_mask, "competition_id"]
        season_ids = chunk.loc[primary_mask, "season_id"]

        genders = []
        for comp_id, season_id in zip(comp_ids, season_ids):
            info = competition_map.get(
                make_key(comp_id, season_id),
                {},
            )
            genders.append(
                info.get("competition_gender", "unknown")
                or "unknown"
            )

        gender_series = pd.Series(
            genders,
            index=primary_type.index,
            dtype="string",
        )

        gender_cross.update(
            pd.DataFrame(
                {
                    "competition_gender": gender_series,
                    "failure": primary_binary,
                }
            ).value_counts(dropna=False).to_dict()
        )

        for column in MODEL_CANDIDATE_COLUMNS:
            if column not in primary_chunk.columns:
                continue

            series = primary_chunk[column]
            if series.dtype == object or str(series.dtype).startswith(
                "string"
            ):
                missing_mask = (
                    series.isna()
                    | series.astype("string").str.strip().eq("")
                )
            else:
                missing_mask = series.isna()

            feature_missing[column] += int(missing_mask.sum())
            feature_nonmissing[column] += int(
                (~missing_mask).sum()
            )

        match_frame = chunk[
            [
                "match_id",
                "competition_id",
                "season_id",
                "competition_name",
                "season_name",
                "match_date",
            ]
        ].copy()
        match_frame["primary"] = primary_mask.astype(int)
        match_frame["strict"] = strict_mask.astype(int)
        match_frame["success"] = success.astype(int)
        match_frame["failure"] = failure.astype(int)

        grouped = match_frame.groupby(
            "match_id",
            dropna=False,
        )

        for match_id, group in grouped:
            match_id_text = normalize_id_value(match_id)
            target = match_stats[match_id_text]

            target["primary_passes"] += int(
                group["primary"].sum()
            )
            target["primary_successes"] += int(
                (group["primary"] * group["success"]).sum()
            )
            target["primary_failures"] += int(
                (group["primary"] * group["failure"]).sum()
            )
            target["strict_passes"] += int(
                group["strict"].sum()
            )
            target["strict_successes"] += int(
                (group["strict"] * group["success"]).sum()
            )
            target["strict_failures"] += int(
                (group["strict"] * group["failure"]).sum()
            )

            first = group.iloc[0]
            info = competition_map.get(
                make_key(
                    first["competition_id"],
                    first["season_id"],
                ),
                {},
            )

            target["competition_gender"] = (
                info.get("competition_gender", "unknown")
                or "unknown"
            )
            target["competition_name"] = str(
                first["competition_name"]
            )
            target["season_name"] = str(
                first["season_name"]
            )
            target["match_date"] = str(
                first["match_date"]
            )

        if (
            args.progress_every_chunks > 0
            and chunk_number
            % args.progress_every_chunks
            == 0
        ):
            print(
                f"[progress] chunks={chunk_number}, "
                f"raw_rows={raw_rows:,}, "
                f"primary_rows={primary_rows:,}, "
                f"primary_failure={primary_failure:,}"
            )

    if unexpected_boolean_values:
        raise RuntimeError(
            "Unexpected values were found in open-play flags: "
            + json.dumps(
                dict(unexpected_boolean_values),
                ensure_ascii=False,
            )
        )

    def outcome_table(counter: Counter) -> pd.DataFrame:
        total = sum(counter.values())
        rows = []
        for outcome, count in counter.most_common():
            rows.append(
                {
                    "pass_outcome_normalized": outcome,
                    "rows": count,
                    "percentage": (
                        count / total * 100.0
                        if total
                        else 0.0
                    ),
                }
            )
        return pd.DataFrame(rows)

    outcome_table(primary_outcomes).to_csv(
        output_dir / "pass_outcome_distribution_primary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    outcome_table(strict_outcomes).to_csv(
        output_dir / "pass_outcome_distribution_strict.csv",
        index=False,
        encoding="utf-8-sig",
    )

    def cross_table(
        counter: Counter,
        group_name: str,
    ) -> pd.DataFrame:
        grouped_counts = defaultdict(
            lambda: {"successes": 0, "failures": 0}
        )

        for (group_value, failure_value), count in counter.items():
            if int(failure_value) == 1:
                grouped_counts[group_value]["failures"] += int(count)
            else:
                grouped_counts[group_value]["successes"] += int(count)

        rows = []
        for group_value, counts in grouped_counts.items():
            total = counts["successes"] + counts["failures"]
            rows.append(
                {
                    group_name: group_value,
                    "passes": total,
                    "successes": counts["successes"],
                    "failures": counts["failures"],
                    "failure_rate": (
                        counts["failures"] / total
                        if total
                        else 0.0
                    ),
                }
            )

        return pd.DataFrame(rows).sort_values(
            "passes",
            ascending=False,
        )

    cross_table(
        pass_type_cross,
        "pass_type",
    ).to_csv(
        output_dir / "outcome_by_pass_type_primary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cross_table(
        pass_height_cross,
        "pass_height",
    ).to_csv(
        output_dir / "outcome_by_pass_height_primary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cross_table(
        pressure_cross,
        "under_pressure",
    ).to_csv(
        output_dir / "outcome_by_pressure_primary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cross_table(
        gender_cross,
        "competition_gender",
    ).to_csv(
        output_dir / "outcome_by_gender_primary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missingness_rows = []
    for column in MODEL_CANDIDATE_COLUMNS:
        missing_count = feature_missing[column]
        nonmissing_count = feature_nonmissing[column]
        total = missing_count + nonmissing_count
        missingness_rows.append(
            {
                "column": column,
                "rows": total,
                "missing_rows": missing_count,
                "nonmissing_rows": nonmissing_count,
                "missing_percentage": (
                    missing_count / total * 100.0
                    if total
                    else 0.0
                ),
            }
        )

    pd.DataFrame(missingness_rows).to_csv(
        output_dir / "candidate_feature_missingness_primary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    match_rows = []
    for match_id, values in match_stats.items():
        primary_total = values["primary_passes"]
        strict_total = values["strict_passes"]

        match_rows.append(
            {
                "match_id": match_id,
                **values,
                "primary_failure_rate": (
                    values["primary_failures"]
                    / primary_total
                    if primary_total
                    else 0.0
                ),
                "strict_failure_rate": (
                    values["strict_failures"]
                    / strict_total
                    if strict_total
                    else 0.0
                ),
            }
        )

    match_table = pd.DataFrame(match_rows)
    match_table.to_csv(
        output_dir / "match_level_target_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    target_definition = {
        "target_name": "xpass_success",
        "success_value": 1,
        "failure_value": 0,
        "success_rule": (
            "pass_outcome is missing or blank"
        ),
        "failure_rule": (
            "pass_outcome contains a non-missing StatsBomb "
            "outcome category"
        ),
        "main_analysis_filter": (
            "primary_open_play == True"
        ),
        "strict_sensitivity_filter": (
            "strict_open_play == True"
        ),
        "predictors_explicitly_excluded": (
            POST_OUTCOME_OR_EXCLUDED_PREDICTORS
        ),
        "split_unit": "match_id",
    }

    with (
        output_dir / "xpass_target_definition.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            target_definition,
            file,
            ensure_ascii=False,
            indent=2,
        )

    primary_failure_rate = (
        primary_failure / primary_rows
        if primary_rows
        else 0.0
    )
    strict_failure_rate = (
        strict_failure / strict_rows
        if strict_rows
        else 0.0
    )

    gender_unknown_rows = 0
    for (gender, failure_value), count in gender_cross.items():
        if str(gender).strip().lower() in {
            "",
            "unknown",
            "nan",
            "<na>",
        }:
            gender_unknown_rows += int(count)

    summary = {
        "status": "PASS",
        "input_csv": str(input_csv),
        "raw_rows_seen": raw_rows,
        "unique_matches": len(match_stats),
        "primary_open_play": {
            "rows": primary_rows,
            "successes": primary_success,
            "failures": primary_failure,
            "failure_rate": primary_failure_rate,
        },
        "strict_open_play": {
            "rows": strict_rows,
            "successes": strict_success,
            "failures": strict_failure,
            "failure_rate": strict_failure_rate,
        },
        "distinct_primary_outcome_categories": (
            len(primary_outcomes)
        ),
        "primary_outcome_categories": dict(
            primary_outcomes
        ),
        "gender_mapping_unknown_rows": gender_unknown_rows,
        "target_definition": target_definition,
        "recommended_next_step": (
            "Create one deterministic match-level split manifest "
            "for training, tuning, calibration, and final testing; "
            "then fit majority, constant-probability, geometry-only, "
            "and full xPass models using the identical split."
        ),
    }

    with (
        output_dir / "xpass_target_audit_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    pd.DataFrame(
        [
            {
                "item": key,
                "value": json.dumps(
                    value,
                    ensure_ascii=False,
                ),
            }
            for key, value in summary.items()
        ]
    ).to_csv(
        output_dir / "xpass_target_audit_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 100)
    print("xPass target and class-balance audit completed")
    print("=" * 100)
    print(f"Raw rows processed: {raw_rows:,}")
    print(f"Unique matches: {len(match_stats):,}")
    print(f"Primary open-play passes: {primary_rows:,}")
    print(f"Primary-sample successes: {primary_success:,}")
    print(f"Primary-sample failures: {primary_failure:,}")
    print(f"Primary-sample failure rate: {primary_failure_rate:.4%}")
    print(f"Strict-sample passes: {strict_rows:,}")
    print(f"Strict-sample failure rate: {strict_failure_rate:.4%}")
    print(
        "Distinct pass_outcome categories in the primary sample: "
        f"{len(primary_outcomes):,}"
    )
    print(f"Rows with unknown competition gender: {gender_unknown_rows:,}")
    print("Status: PASS")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
