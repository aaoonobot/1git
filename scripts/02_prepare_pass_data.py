#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare a validated StatsBomb pass table for downstream modeling.

This stage is non-destructive:
- all original columns are retained unchanged;
- parsed and boundary-clipped model coordinates are added as new columns;
- two auditable open-play flags are added:
  1) primary_open_play: excludes only passes whose own pass_type is a restart;
  2) strict_open_play: additionally excludes passes whose play_pattern indicates
     a restart-derived possession sequence.

No original rows are deleted. Downstream scripts should normally filter on
primary_open_play == True for the main analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


PRIMARY_RESTART_PASS_TYPES = {
    "corner",
    "free kick",
    "goal kick",
    "kick off",
    "throw in",
    "dropped ball",
}

STRICT_RESTART_PLAY_PATTERNS = {
    "from corner",
    "from free kick",
    "from goal kick",
    "from kick off",
    "from throw in",
    "from dropped ball",
}

ADDED_COLUMNS = [
    "start_x_raw",
    "start_y_raw",
    "end_x_raw",
    "end_y_raw",
    "start_x_model",
    "start_y_model",
    "end_x_model",
    "end_y_model",
    "start_coordinate_clipped",
    "end_coordinate_clipped",
    "primary_open_play",
    "strict_open_play",
    "primary_exclusion_reason",
    "strict_exclusion_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create model-ready coordinates and reproducible open-play flags."
    )
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--output_csv_name",
        default="passes_preprocessed_with_open_play_flags.csv",
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--progress_every_chunks", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_hash", action="store_true")
    return parser.parse_args()


def normalize_category(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .str.replace(r"[_\-]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )


def parse_coordinate_series(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    text = series.astype("string").fillna("").str.strip()
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    pattern = rf"^\s*\[\s*({number})\s*,\s*({number})"
    extracted = text.str.extract(pattern, expand=True)
    x = pd.to_numeric(extracted[0], errors="coerce")
    y = pd.to_numeric(extracted[1], errors="coerce")
    malformed = x.isna() | y.isna()
    return x, y, malformed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update_counter(counter: Counter, series: pd.Series) -> None:
    values = series.astype("string").fillna("<NA>").replace("", "<NA>")
    counter.update(values.value_counts(dropna=False).to_dict())


def main() -> None:
    args = parse_args()

    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    output_csv = output_dir / args.output_csv_name
    coordinate_audit_csv = output_dir / "coordinate_out_of_range_full.csv"
    malformed_csv = output_dir / "malformed_coordinate_rows.csv"

    outputs = [
        output_csv,
        coordinate_audit_csv,
        malformed_csv,
        output_dir / "open_play_sample_flow.csv",
        output_dir / "match_level_sample_flow.csv",
        output_dir / "pass_type_distribution.csv",
        output_dir / "play_pattern_distribution.csv",
        output_dir / "pass_type_play_pattern_crosstab.csv",
        output_dir / "preprocessing_config.json",
        output_dir / "preprocessing_summary.json",
    ]

    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Output files already exist. Use --overwrite after reviewing them:\n"
            + joined
        )

    for path in existing:
        path.unlink()

    header = pd.read_csv(input_csv, encoding="utf-8-sig", nrows=0).columns.tolist()
    required = {
        "id",
        "match_id",
        "location",
        "pass_end_location",
        "pass_type",
        "play_pattern",
    }
    missing = sorted(required - set(header))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    duplicated_added = sorted(set(header) & set(ADDED_COLUMNS))
    if duplicated_added:
        raise RuntimeError(
            f"Input already contains preprocessing columns: {duplicated_added}"
        )

    total_rows = 0
    primary_kept = 0
    primary_excluded = 0
    strict_kept = 0
    strict_excluded = 0

    malformed_start = 0
    malformed_end = 0
    start_clipped_count = 0
    end_clipped_count = 0

    pass_type_counts: Counter = Counter()
    play_pattern_counts: Counter = Counter()
    cross_counts: Counter = Counter()

    match_counts = defaultdict(
        lambda: {
            "raw_passes": 0,
            "primary_open_play_passes": 0,
            "strict_open_play_passes": 0,
            "coordinate_clipped_rows": 0,
        }
    )

    coordinate_audit_header_written = False
    malformed_header_written = False
    output_header_written = False

    reader = pd.read_csv(
        input_csv,
        encoding="utf-8-sig",
        chunksize=args.chunksize,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        rows = len(chunk)
        total_rows += rows

        pass_type_norm = normalize_category(chunk["pass_type"])
        play_pattern_norm = normalize_category(chunk["play_pattern"])

        primary_restart = pass_type_norm.isin(PRIMARY_RESTART_PASS_TYPES)
        strict_restart_pattern = play_pattern_norm.isin(
            STRICT_RESTART_PLAY_PATTERNS
        )

        primary_open_play = ~primary_restart
        strict_open_play = ~(primary_restart | strict_restart_pattern)

        primary_kept += int(primary_open_play.sum())
        primary_excluded += int(primary_restart.sum())
        strict_kept += int(strict_open_play.sum())
        strict_excluded += int((~strict_open_play).sum())

        primary_reason = pd.Series("", index=chunk.index, dtype="string")
        primary_reason.loc[primary_restart] = (
            "pass_type:" + pass_type_norm.loc[primary_restart]
        )

        strict_reason = primary_reason.copy()
        pattern_only = (~primary_restart) & strict_restart_pattern
        strict_reason.loc[pattern_only] = (
            "play_pattern:" + play_pattern_norm.loc[pattern_only]
        )

        sx, sy, malformed_s = parse_coordinate_series(chunk["location"])
        ex, ey, malformed_e = parse_coordinate_series(
            chunk["pass_end_location"]
        )

        malformed_start += int(malformed_s.sum())
        malformed_end += int(malformed_e.sum())

        if malformed_s.any() or malformed_e.any():
            bad = chunk.loc[
                malformed_s | malformed_e,
                ["id", "match_id", "location", "pass_end_location"],
            ].copy()
            bad["malformed_start"] = malformed_s.loc[bad.index].to_numpy()
            bad["malformed_end"] = malformed_e.loc[bad.index].to_numpy()
            bad.to_csv(
                malformed_csv,
                mode="a",
                header=not malformed_header_written,
                index=False,
                encoding="utf-8-sig",
            )
            malformed_header_written = True

        start_out = (
            (sx < 0.0) | (sx > 120.0) | (sy < 0.0) | (sy > 80.0)
        ) & ~malformed_s
        end_out = (
            (ex < 0.0) | (ex > 120.0) | (ey < 0.0) | (ey > 80.0)
        ) & ~malformed_e

        start_clipped_count += int(start_out.sum())
        end_clipped_count += int(end_out.sum())

        sx_model = sx.clip(0.0, 120.0)
        sy_model = sy.clip(0.0, 80.0)
        ex_model = ex.clip(0.0, 120.0)
        ey_model = ey.clip(0.0, 80.0)

        if start_out.any() or end_out.any():
            audit_parts = []

            if start_out.any():
                start_audit = chunk.loc[
                    start_out,
                    ["id", "match_id", "location", "pass_end_location"],
                ].copy()
                start_audit["coordinate_field"] = "location"
                start_audit["raw_x"] = sx.loc[start_out].to_numpy()
                start_audit["raw_y"] = sy.loc[start_out].to_numpy()
                start_audit["clipped_x"] = sx_model.loc[start_out].to_numpy()
                start_audit["clipped_y"] = sy_model.loc[start_out].to_numpy()
                audit_parts.append(start_audit)

            if end_out.any():
                end_audit = chunk.loc[
                    end_out,
                    ["id", "match_id", "location", "pass_end_location"],
                ].copy()
                end_audit["coordinate_field"] = "pass_end_location"
                end_audit["raw_x"] = ex.loc[end_out].to_numpy()
                end_audit["raw_y"] = ey.loc[end_out].to_numpy()
                end_audit["clipped_x"] = ex_model.loc[end_out].to_numpy()
                end_audit["clipped_y"] = ey_model.loc[end_out].to_numpy()
                audit_parts.append(end_audit)

            audit_chunk = pd.concat(audit_parts, ignore_index=True)
            audit_chunk.to_csv(
                coordinate_audit_csv,
                mode="a",
                header=not coordinate_audit_header_written,
                index=False,
                encoding="utf-8-sig",
            )
            coordinate_audit_header_written = True

        chunk["start_x_raw"] = sx
        chunk["start_y_raw"] = sy
        chunk["end_x_raw"] = ex
        chunk["end_y_raw"] = ey

        chunk["start_x_model"] = sx_model
        chunk["start_y_model"] = sy_model
        chunk["end_x_model"] = ex_model
        chunk["end_y_model"] = ey_model

        chunk["start_coordinate_clipped"] = start_out
        chunk["end_coordinate_clipped"] = end_out

        chunk["primary_open_play"] = primary_open_play
        chunk["strict_open_play"] = strict_open_play
        chunk["primary_exclusion_reason"] = primary_reason
        chunk["strict_exclusion_reason"] = strict_reason

        chunk.to_csv(
            output_csv,
            mode="a",
            header=not output_header_written,
            index=False,
            encoding="utf-8-sig",
        )
        output_header_written = True

        update_counter(pass_type_counts, pass_type_norm)
        update_counter(play_pattern_counts, play_pattern_norm)

        cross_frame = pd.DataFrame(
            {
                "pass_type": pass_type_norm.replace("", "<NA>"),
                "play_pattern": play_pattern_norm.replace("", "<NA>"),
            }
        )
        cross_counts.update(
            cross_frame.value_counts(dropna=False).to_dict()
        )

        match_ids = chunk["match_id"].astype("string").str.strip()
        clipped_any = start_out | end_out

        match_frame = pd.DataFrame(
            {
                "match_id": match_ids,
                "raw_passes": 1,
                "primary_open_play_passes": primary_open_play.astype(int),
                "strict_open_play_passes": strict_open_play.astype(int),
                "coordinate_clipped_rows": clipped_any.astype(int),
            }
        )
        grouped = (
            match_frame.groupby("match_id", dropna=False)[
                [
                    "raw_passes",
                    "primary_open_play_passes",
                    "strict_open_play_passes",
                    "coordinate_clipped_rows",
                ]
            ]
            .sum()
        )
        for match_id, values in grouped.iterrows():
            target = match_counts[str(match_id)]
            for key in target:
                target[key] += int(values[key])

        if (
            args.progress_every_chunks > 0
            and chunk_number % args.progress_every_chunks == 0
        ):
            print(
                f"[progress] chunks={chunk_number}, rows={total_rows:,}, "
                f"primary_kept={primary_kept:,}, strict_kept={strict_kept:,}, "
                f"clipped_fields={start_clipped_count + end_clipped_count:,}"
            )

    if malformed_start > 0 or malformed_end > 0:
        raise RuntimeError(
            "Malformed coordinates were found. Review "
            "malformed_coordinate_rows.csv before continuing."
        )

    if not coordinate_audit_header_written:
        pd.DataFrame(
            columns=[
                "id",
                "match_id",
                "location",
                "pass_end_location",
                "coordinate_field",
                "raw_x",
                "raw_y",
                "clipped_x",
                "clipped_y",
            ]
        ).to_csv(
            coordinate_audit_csv,
            index=False,
            encoding="utf-8-sig",
        )

    if not malformed_header_written:
        pd.DataFrame(
            columns=[
                "id",
                "match_id",
                "location",
                "pass_end_location",
                "malformed_start",
                "malformed_end",
            ]
        ).to_csv(
            malformed_csv,
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame(
        [
            {
                "stage": "raw_pass_events",
                "rows": total_rows,
                "excluded_from_raw": 0,
            },
            {
                "stage": "primary_open_play",
                "rows": primary_kept,
                "excluded_from_raw": primary_excluded,
            },
            {
                "stage": "strict_open_play_sensitivity",
                "rows": strict_kept,
                "excluded_from_raw": strict_excluded,
            },
        ]
    ).to_csv(
        output_dir / "open_play_sample_flow.csv",
        index=False,
        encoding="utf-8-sig",
    )

    match_rows = [
        {"match_id": match_id, **values}
        for match_id, values in match_counts.items()
    ]
    pd.DataFrame(match_rows).sort_values("match_id").to_csv(
        output_dir / "match_level_sample_flow.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {"pass_type_normalized": key, "rows": value}
            for key, value in pass_type_counts.most_common()
        ]
    ).to_csv(
        output_dir / "pass_type_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {"play_pattern_normalized": key, "rows": value}
            for key, value in play_pattern_counts.most_common()
        ]
    ).to_csv(
        output_dir / "play_pattern_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {
                "pass_type_normalized": key[0],
                "play_pattern_normalized": key[1],
                "rows": value,
            }
            for key, value in cross_counts.most_common()
        ]
    ).to_csv(
        output_dir / "pass_type_play_pattern_crosstab.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "pitch_bounds": {
            "x_min": 0.0,
            "x_max": 120.0,
            "y_min": 0.0,
            "y_max": 80.0,
        },
        "primary_restart_pass_types": sorted(
            PRIMARY_RESTART_PASS_TYPES
        ),
        "strict_restart_play_patterns": sorted(
            STRICT_RESTART_PLAY_PATTERNS
        ),
        "main_analysis_filter": "primary_open_play == True",
        "sensitivity_filter": "strict_open_play == True",
        "raw_coordinate_policy": (
            "Original location strings are retained unchanged. "
            "Only model coordinate copies are clipped."
        ),
    }

    with (output_dir / "preprocessing_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    input_hash = "" if args.skip_hash else sha256_file(input_csv)
    output_hash = "" if args.skip_hash else sha256_file(output_csv)

    summary = {
        "status": "PASS",
        "input_csv": str(input_csv),
        "input_sha256": input_hash,
        "output_csv": str(output_csv),
        "output_sha256": output_hash,
        "raw_pass_rows": total_rows,
        "unique_matches": len(match_counts),
        "primary_open_play_rows": primary_kept,
        "primary_excluded_restart_rows": primary_excluded,
        "strict_open_play_rows": strict_kept,
        "strict_excluded_restart_or_restart_sequence_rows": strict_excluded,
        "malformed_start_coordinates": malformed_start,
        "malformed_end_coordinates": malformed_end,
        "start_coordinate_fields_clipped": start_clipped_count,
        "end_coordinate_fields_clipped": end_clipped_count,
        "total_coordinate_fields_clipped": (
            start_clipped_count + end_clipped_count
        ),
        "main_analysis_filter": "primary_open_play == True",
        "sensitivity_filter": "strict_open_play == True",
    }

    with (output_dir / "preprocessing_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("StatsBomb coordinate processing and open-play classification completed")
    print("=" * 100)
    print(f"Raw pass events: {total_rows:,}")
    print(f"Matches: {len(match_counts):,}")
    print(f"Primary open-play passes: {primary_kept:,}")
    print(f"Passes excluded by direct restart type: {primary_excluded:,}")
    print(f"Strict open-play passes: {strict_kept:,}")
    print(f"Passes excluded by the strict definition: {strict_excluded:,}")
    print(
        "Coordinate fields clipped to pitch boundaries: "
        f"{start_clipped_count + end_clipped_count:,}"
    )
    print(f"Model-ready pass table: {output_csv}")
    print(f"Sample-flow table: {output_dir / 'open_play_sample_flow.csv'}")
    print(f"Preprocessing configuration: {output_dir / 'preprocessing_config.json'}")
    print("Status: PASS")


if __name__ == "__main__":
    main()
