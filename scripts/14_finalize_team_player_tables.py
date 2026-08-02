#!/usr/bin/env python
"""Finalize the three-team team and player summary tables.

The script adds a full-sample expected-net-value reference, centered values,
shifted confidence intervals, eligibility checks, and ranks restricted to
eligible players. Source tables are never overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FULL_SAMPLE_ROWS = 3_396_912
DEFAULT_FULL_SAMPLE_BASELINE_PER_100 = -0.9528630592451951



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
        description="Finalize team and player summary tables."
    )
    parser.add_argument("--team_input", type=Path, required=True)
    parser.add_argument("--player_input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--full_sample_rows",
        type=int,
        default=DEFAULT_FULL_SAMPLE_ROWS,
    )
    parser.add_argument(
        "--full_sample_baseline_per_100",
        type=float,
        default=DEFAULT_FULL_SAMPLE_BASELINE_PER_100,
    )
    parser.add_argument("--min_player_passes", type=int, default=200)
    parser.add_argument("--min_player_matches", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "1.0", "yes", "y", "t"])
    )


def centered_ci_direction(low: float, high: float) -> str:
    if pd.isna(low) or pd.isna(high):
        return "Missing"
    if low > 0:
        return "Above full-sample baseline"
    if high < 0:
        return "Below full-sample baseline"
    return "Includes full-sample baseline"


def execution_ci_direction(low: float, high: float) -> str:
    if pd.isna(low) or pd.isna(high):
        return "Missing"
    if low > 0:
        return "Above xPass expectation"
    if high < 0:
        return "Below xPass expectation"
    return "Includes zero"


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"{label} is missing required columns: {', '.join(missing)}"
        )


def max_abs(series: pd.Series) -> float:
    values = np.abs(series.to_numpy(dtype=float))
    return float(np.nanmax(values)) if len(values) else 0.0


def add_centered_columns(
    frame: pd.DataFrame,
    baseline: float,
) -> pd.DataFrame:
    result = frame.copy()
    result["full_sample_baseline_per_100_passes"] = baseline
    result["centered_expected_net_value_per_100_passes"] = (
        result["expected_net_value_per_100_passes"] - baseline
    )
    result[
        "centered_expected_net_value_per_100_passes_ci_low"
    ] = result["expected_net_value_per_100_passes_ci_low"] - baseline
    result[
        "centered_expected_net_value_per_100_passes_ci_high"
    ] = result["expected_net_value_per_100_passes_ci_high"] - baseline
    result["centered_net_value_ci_direction"] = [
        centered_ci_direction(low, high)
        for low, high in zip(
            result[
                "centered_expected_net_value_per_100_passes_ci_low"
            ],
            result[
                "centered_expected_net_value_per_100_passes_ci_high"
            ],
        )
    ]
    result["execution_ci_direction"] = [
        execution_ci_direction(low, high)
        for low, high in zip(
            result["execution_residual_per_100_passes_ci_low"],
            result["execution_residual_per_100_passes_ci_high"],
        )
    ]
    return result


def main() -> None:
    args = parse_args()
    team_path = args.team_input.resolve()
    player_path = args.player_input.resolve()
    output_dir = args.output_dir.resolve()

    for path in (team_path, player_path):
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "team": output_dir / "team_summary_final.csv",
        "player": output_dir / "player_summary_final.csv",
        "audit": output_dir / "finalization_audit.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Outputs already exist. Use --overwrite to replace them:\n"
            + "\n".join(str(path) for path in existing)
        )
    for path in existing:
        path.unlink()

    team = pd.read_csv(team_path, encoding="utf-8-sig", low_memory=False)
    player = pd.read_csv(player_path, encoding="utf-8-sig", low_memory=False)

    common_required = {
        "expected_net_value_per_100_passes",
        "expected_net_value_per_100_passes_ci_low",
        "expected_net_value_per_100_passes_ci_high",
        "execution_residual_per_100_passes",
        "execution_residual_per_100_passes_ci_low",
        "execution_residual_per_100_passes_ci_high",
    }
    require_columns(
        team,
        common_required | {"team", "n_matches", "n_passes"},
        "team input",
    )
    require_columns(
        player,
        common_required
        | {
            "team",
            "player",
            "n_matches",
            "n_passes",
            "eligible_primary",
        },
        "player input",
    )

    baseline = float(args.full_sample_baseline_per_100)
    team_final = add_centered_columns(team, baseline)
    player_final = add_centered_columns(player, baseline)

    eligible = (
        to_bool(player_final["eligible_primary"])
        & (
            pd.to_numeric(player_final["n_passes"], errors="coerce")
            >= args.min_player_passes
        )
        & (
            pd.to_numeric(player_final["n_matches"], errors="coerce")
            >= args.min_player_matches
        )
        & player_final["player"].astype("string").fillna("").str.strip().ne("")
        & player_final["player"].astype("string").fillna("").str.strip().ne("Unknown")
    )
    player_final["eligible_primary_rechecked"] = eligible

    for column in [
        "corrected_decision_rank_within_team",
        "corrected_execution_rank_within_team",
    ]:
        player_final[column] = pd.Series(
            pd.NA,
            index=player_final.index,
            dtype="Int64",
        )

    player_final.loc[
        eligible,
        "corrected_decision_rank_within_team",
    ] = (
        player_final.loc[eligible]
        .groupby("team")["centered_expected_net_value_per_100_passes"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    player_final.loc[
        eligible,
        "corrected_execution_rank_within_team",
    ] = (
        player_final.loc[eligible]
        .groupby("team")["execution_residual_per_100_passes"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    eligible_counts = (
        player_final.loc[eligible]
        .groupby("team")["player"]
        .size()
        .rename("eligible_players_in_team")
    )
    player_final = player_final.merge(
        eligible_counts,
        left_on="team",
        right_index=True,
        how="left",
    )
    player_final["eligible_players_in_team"] = (
        player_final["eligible_players_in_team"].fillna(0).astype(int)
    )

    team_identity_error = max_abs(
        team_final["centered_expected_net_value_per_100_passes"]
        - (team_final["expected_net_value_per_100_passes"] - baseline)
    )
    player_identity_error = max_abs(
        player_final["centered_expected_net_value_per_100_passes"]
        - (player_final["expected_net_value_per_100_passes"] - baseline)
    )
    team_ci_width_error = max_abs(
        (
            team_final[
                "centered_expected_net_value_per_100_passes_ci_high"
            ]
            - team_final[
                "centered_expected_net_value_per_100_passes_ci_low"
            ]
        )
        - (
            team_final["expected_net_value_per_100_passes_ci_high"]
            - team_final["expected_net_value_per_100_passes_ci_low"]
        )
    )
    player_ci_width_error = max_abs(
        (
            player_final[
                "centered_expected_net_value_per_100_passes_ci_high"
            ]
            - player_final[
                "centered_expected_net_value_per_100_passes_ci_low"
            ]
        )
        - (
            player_final["expected_net_value_per_100_passes_ci_high"]
            - player_final["expected_net_value_per_100_passes_ci_low"]
        )
    )

    rank_columns = [
        "corrected_decision_rank_within_team",
        "corrected_execution_rank_within_team",
    ]
    checks = {
        "team_rows_equal_3": len(team_final) == 3,
        "player_rows_equal_80": len(player_final) == 80,
        "eligible_players_equal_55": int(eligible.sum()) == 55,
        "centered_identity_exact_team": team_identity_error < 1e-12,
        "centered_identity_exact_player": player_identity_error < 1e-12,
        "ci_width_preserved_team": team_ci_width_error < 1e-12,
        "ci_width_preserved_player": player_ci_width_error < 1e-12,
        "all_eligible_players_ranked": (
            player_final.loc[eligible, rank_columns].isna().sum().sum() == 0
        ),
        "no_ineligible_player_ranked": (
            player_final.loc[~eligible, rank_columns].notna().sum().sum() == 0
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Finalization audit failed:\n"
            + json.dumps(checks, ensure_ascii=False, indent=2, default=json_default)
        )

    team_final.to_csv(outputs["team"], index=False, encoding="utf-8-sig")
    player_final.to_csv(outputs["player"], index=False, encoding="utf-8-sig")
    audit = {
        "status": "PASS",
        "full_sample_reference": {
            "model": "H=3 main OOF xPass, primary open-play sample",
            "rows": args.full_sample_rows,
            "baseline_per_100_passes": baseline,
            "interval_treatment": (
                "The baseline is treated as a fixed descriptive reference; "
                "point estimates and interval bounds are shifted by the same constant."
            ),
        },
        "player_primary_eligibility": {
            "minimum_passes": args.min_player_passes,
            "minimum_matches": args.min_player_matches,
            "eligible_total": int(eligible.sum()),
            "eligible_by_team": {
                str(team_name): int(count)
                for team_name, count in eligible_counts.items()
            },
        },
        "checks": checks,
        "inputs": {
            "team_input": str(team_path),
            "player_input": str(player_path),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["audit"].write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    print("=" * 100)
    print("Team/player table finalization completed")
    print("=" * 100)
    print(f"Full-sample baseline per 100 passes: {baseline:.12f}")
    print(f"Team rows: {len(team_final):,}")
    print(f"Player rows: {len(player_final):,}")
    print(f"Primary-eligible players: {int(eligible.sum()):,}")
    print("Status: PASS")
    print(f"Team table: {outputs['team']}")
    print(f"Player table: {outputs['player']}")
    print(f"Audit: {outputs['audit']}")


if __name__ == "__main__":
    main()
