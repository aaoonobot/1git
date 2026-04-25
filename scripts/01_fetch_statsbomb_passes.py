#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch StatsBomb Open Data pass events.

This script converts the exploratory notebook used for downloading event data
into a reproducible command-line workflow. It performs four tasks:

1. Download the available competition-season catalogue.
2. Download match metadata for selected competition-seasons.
3. Download event data match by match and retain pass events.
4. Save pass-event data, metadata, and a download log for reproducibility.

The script is designed for research reproducibility. It avoids local computer
paths, notebook-state variables, and manual retry steps.
"""

from __future__ import annotations

import argparse
import time
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from statsbombpy import sb


PASS_COLUMNS = [
    "id", "index", "team", "player", "position", "minute", "second",
    "location", "pass_end_location", "pass_angle", "pass_length",
    "pass_height", "pass_body_part", "pass_outcome", "pass_type",
    "pass_cross", "pass_switch", "pass_shot_assist", "pass_goal_assist",
    "pass_aerial_won", "pass_cut_back", "pass_deflected",
    "pass_miscommunication", "pass_through_ball", "pass_backheel",
    "pass_outswinging", "pass_inswinging", "pass_straight",
    "pass_recipient", "pass_recovery", "pass_technique",
    "under_pressure", "play_pattern", "related_events",
    "possession", "possession_team", "match_id",
    "competition_id", "season_id", "competition_name", "season_name",
    "match_date", "home_team", "away_team", "home_score", "away_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download StatsBomb Open Data pass events for the xPass and risk-reward pipeline."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where downloaded pass events and metadata will be saved.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="passes_all_matches_fixed.csv",
        help="Name of the pass-event CSV file to create inside output_dir.",
    )
    parser.add_argument(
        "--min_season_year",
        type=int,
        default=2000,
        help="Retain competition-seasons whose season start year is at least this value.",
    )
    parser.add_argument(
        "--max_matches",
        type=int,
        default=None,
        help="Optional cap on the number of matches to download. Useful for testing.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="Number of matches after which downloaded pass events are flushed to disk.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Maximum number of download attempts per match.",
    )
    parser.add_argument(
        "--base_sleep",
        type=float,
        default=1.5,
        help="Base sleep time in seconds for exponential backoff after failed downloads.",
    )
    parser.add_argument(
        "--request_sleep",
        type=float,
        default=0.0,
        help="Optional pause in seconds after each successful match request.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip match_ids that are already present in the output CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files. Ignored when --resume is used.",
    )
    parser.add_argument(
        "--exclude_team",
        type=str,
        default=None,
        help="Optional team name to exclude from the download sample.",
    )
    parser.add_argument(
        "--exclude_competition_contains",
        type=str,
        default=None,
        help="Optional substring. Matches from competitions containing this value are excluded.",
    )
    parser.add_argument(
        "--exclude_season",
        type=str,
        default=None,
        help="Optional season label to exclude, for example '2023/2024'.",
    )
    parser.add_argument(
        "--split_rows",
        type=int,
        default=0,
        help="If greater than 0, split the final pass CSV into part files with this many rows each.",
    )
    return parser.parse_args()


def extract_start_year(season_name: object) -> Optional[int]:
    text = str(season_name)
    if not text or text.lower() == "nan":
        return None
    try:
        if "/" in text:
            return int(text.split("/")[0])
        return int(text[:4])
    except Exception:
        return None


def normalize_string(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns]


def standardize_match_metadata(matches: pd.DataFrame, competition_row: pd.Series) -> pd.DataFrame:
    out = matches.copy()
    out["competition_id"] = int(competition_row["competition_id"])
    out["season_id"] = int(competition_row["season_id"])
    out["competition_name"] = str(competition_row.get("competition_name", ""))
    out["season_name"] = str(competition_row.get("season_name", ""))

    if "competition" not in out.columns:
        out["competition"] = out["competition_name"]
    if "season" not in out.columns:
        out["season"] = out["season_name"]
    return out


def collect_competitions(min_season_year: int) -> pd.DataFrame:
    competitions = sb.competitions().copy()
    competitions["start_year"] = competitions["season_name"].apply(extract_start_year)
    competitions = competitions[competitions["start_year"].notna()].copy()
    competitions["start_year"] = competitions["start_year"].astype(int)
    competitions = competitions[competitions["start_year"] >= min_season_year].reset_index(drop=True)
    return competitions


def collect_matches(competitions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    match_tables = []
    log_rows = []

    for _, row in competitions.iterrows():
        comp_id = int(row["competition_id"])
        season_id = int(row["season_id"])
        comp_name = str(row.get("competition_name", ""))
        season_name = str(row.get("season_name", ""))

        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            matches = standardize_match_metadata(matches, row)
            match_tables.append(matches)
            status = "success"
            message = ""
            n_matches = len(matches)
        except Exception as exc:
            status = "failed"
            message = str(exc)
            n_matches = 0

        log_rows.append({
            "competition_id": comp_id,
            "season_id": season_id,
            "competition_name": comp_name,
            "season_name": season_name,
            "status": status,
            "message": message,
            "n_matches": n_matches,
        })

    if match_tables:
        all_matches = pd.concat(match_tables, ignore_index=True)
        all_matches = all_matches.drop_duplicates(subset=["match_id"]).reset_index(drop=True)
    else:
        all_matches = pd.DataFrame()

    return all_matches, pd.DataFrame(log_rows)


def apply_match_exclusions(
    matches: pd.DataFrame,
    exclude_team: Optional[str],
    exclude_competition_contains: Optional[str],
    exclude_season: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if matches.empty:
        return matches, matches.copy()

    mask = pd.Series(False, index=matches.index)

    if exclude_team:
        home = matches.get("home_team", pd.Series(pd.NA, index=matches.index))
        away = matches.get("away_team", pd.Series(pd.NA, index=matches.index))
        team_mask = home.astype(str).eq(exclude_team) | away.astype(str).eq(exclude_team)
    else:
        team_mask = pd.Series(True, index=matches.index)

    if exclude_competition_contains:
        comp = matches.get("competition_name", matches.get("competition", pd.Series("", index=matches.index)))
        comp_mask = comp.astype(str).str.contains(exclude_competition_contains, case=False, na=False)
    else:
        comp_mask = pd.Series(True, index=matches.index)

    if exclude_season:
        season = matches.get("season_name", matches.get("season", pd.Series("", index=matches.index)))
        season_mask = season.astype(str).eq(exclude_season)
    else:
        season_mask = pd.Series(True, index=matches.index)

    if exclude_team or exclude_competition_contains or exclude_season:
        mask = team_mask & comp_mask & season_mask

    excluded = matches.loc[mask].copy()
    kept = matches.loc[~mask].copy()
    return kept.reset_index(drop=True), excluded.reset_index(drop=True)


def existing_match_ids(output_csv: Path) -> set[int]:
    if not output_csv.exists():
        return set()

    match_ids: set[int] = set()
    try:
        for chunk in pd.read_csv(output_csv, usecols=["match_id"], chunksize=250_000):
            vals = pd.to_numeric(chunk["match_id"], errors="coerce").dropna().astype(int).unique()
            match_ids.update(vals.tolist())
    except Exception:
        return set()
    return match_ids


def fetch_pass_events(match_row: pd.Series) -> pd.DataFrame:
    match_id = int(match_row["match_id"])
    events = sb.events(match_id=match_id)

    if "type" not in events.columns:
        return pd.DataFrame(columns=PASS_COLUMNS)

    passes = events[events["type"].astype(str).eq("Pass")].copy()
    if passes.empty:
        return pd.DataFrame(columns=PASS_COLUMNS)

    passes["match_id"] = match_id
    metadata_cols = [
        "competition_id", "season_id", "competition_name", "season_name",
        "match_date", "home_team", "away_team", "home_score", "away_score",
    ]
    for col in metadata_cols:
        if col in match_row.index:
            passes[col] = match_row[col]

    return ensure_columns(passes, PASS_COLUMNS)


def flush_batch(batch: List[pd.DataFrame], output_csv: Path, write_header: bool) -> bool:
    if not batch:
        return write_header

    out = pd.concat(batch, ignore_index=True)
    out = ensure_columns(out, PASS_COLUMNS)
    out.to_csv(output_csv, mode="a", header=write_header, index=False, encoding="utf-8-sig")
    batch.clear()
    return False


def split_csv(input_csv: Path, rows_per_file: int) -> None:
    if rows_per_file <= 0:
        return

    split_dir = input_csv.parent / f"{input_csv.stem}_parts"
    split_dir.mkdir(parents=True, exist_ok=True)

    for idx, chunk in enumerate(pd.read_csv(input_csv, chunksize=rows_per_file), start=1):
        part_path = split_dir / f"{input_csv.stem}_part_{idx:03d}.csv"
        chunk.to_csv(part_path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = args.output_dir / args.output_csv
    competitions_csv = args.output_dir / "statsbomb_competitions_selected.csv"
    matches_csv = args.output_dir / "statsbomb_matches_selected.csv"
    excluded_matches_csv = args.output_dir / "statsbomb_matches_excluded.csv"
    competition_log_csv = args.output_dir / "statsbomb_competition_download_log.csv"
    download_log_csv = args.output_dir / "statsbomb_event_download_log.csv"

    if output_csv.exists() and args.overwrite and not args.resume:
        output_csv.unlink()

    if not output_csv.exists():
        pd.DataFrame(columns=PASS_COLUMNS).to_csv(output_csv, index=False, encoding="utf-8-sig")

    competitions = collect_competitions(args.min_season_year)
    competitions.to_csv(competitions_csv, index=False, encoding="utf-8-sig")

    all_matches, competition_log = collect_matches(competitions)
    competition_log.to_csv(competition_log_csv, index=False, encoding="utf-8-sig")

    kept_matches, excluded_matches = apply_match_exclusions(
        all_matches,
        exclude_team=args.exclude_team,
        exclude_competition_contains=args.exclude_competition_contains,
        exclude_season=args.exclude_season,
    )

    kept_matches.to_csv(matches_csv, index=False, encoding="utf-8-sig")
    excluded_matches.to_csv(excluded_matches_csv, index=False, encoding="utf-8-sig")

    if kept_matches.empty:
        raise RuntimeError("No matches remain after filtering. Please check the selection parameters.")

    downloaded = existing_match_ids(output_csv) if args.resume else set()
    if args.resume and downloaded:
        kept_matches = kept_matches[~kept_matches["match_id"].astype(int).isin(downloaded)].copy()

    if args.max_matches is not None:
        kept_matches = kept_matches.head(args.max_matches).copy()

    total_matches = len(kept_matches)
    print(f"Selected competition-seasons: {len(competitions)}")
    print(f"Selected matches for event download: {total_matches}")
    print(f"Output pass-event CSV: {output_csv}")

    batch: List[pd.DataFrame] = []
    log_rows = []
    write_header = output_csv.stat().st_size == 0

    for i, (_, match_row) in enumerate(kept_matches.iterrows(), start=1):
        match_id = int(match_row["match_id"])
        status = "failed"
        message = ""
        n_passes = 0

        for attempt in range(1, args.max_retries + 1):
            try:
                passes = fetch_pass_events(match_row)
                n_passes = len(passes)
                if not passes.empty:
                    batch.append(passes)
                status = "success"
                message = ""
                break
            except Exception as exc:
                message = str(exc)
                sleep_seconds = args.base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.8)
                if attempt < args.max_retries:
                    print(
                        f"[{i}/{total_matches}] match_id={match_id} attempt={attempt} failed: "
                        f"{message}. Retrying in {sleep_seconds:.1f}s."
                    )
                    time.sleep(sleep_seconds)

        log_rows.append({
            "match_id": match_id,
            "status": status,
            "message": message,
            "n_passes": n_passes,
            "competition_id": match_row.get("competition_id", pd.NA),
            "season_id": match_row.get("season_id", pd.NA),
            "competition_name": match_row.get("competition_name", pd.NA),
            "season_name": match_row.get("season_name", pd.NA),
            "match_date": match_row.get("match_date", pd.NA),
            "home_team": match_row.get("home_team", pd.NA),
            "away_team": match_row.get("away_team", pd.NA),
        })

        if len(batch) >= args.batch_size:
            write_header = flush_batch(batch, output_csv, write_header)
            print(f"[progress] processed {i}/{total_matches} matches")

        if args.request_sleep > 0:
            time.sleep(args.request_sleep)

    write_header = flush_batch(batch, output_csv, write_header)

    download_log = pd.DataFrame(log_rows)
    download_log.to_csv(download_log_csv, index=False, encoding="utf-8-sig")

    if args.split_rows > 0:
        split_csv(output_csv, args.split_rows)

    failed = download_log[download_log["status"].ne("success")]
    if not failed.empty:
        failed_path = args.output_dir / "failed_match_ids.txt"
        failed["match_id"].astype(str).to_csv(failed_path, index=False, header=False)
        print(f"Failed matches: {len(failed)}. See {failed_path}")

    print("Download completed.")
    print(f"Pass-event data: {output_csv}")
    print(f"Match metadata: {matches_csv}")
    print(f"Event download log: {download_log_csv}")


if __name__ == "__main__":
    main()
