# -*- coding: utf-8 -*-
"""
Robustness and sensitivity analysis for the pass-value framework.

Primary reference
-----------------
H=3 PCTV, main OOF xPass, primary open-play sample, full failure cost.

Sensitivity specifications
--------------------------
1. H=1 and H=5 PCTV horizons.
2. No-team OOF xPass.
3. Strict open-play sample.
4. Start-threat-only and opponent-transition-only failure costs.
5. Player eligibility thresholds of 100, 200, and 300 passes.

Outputs
--------------
1. robustness_team_results.csv
2. robustness_player_results.csv
3. robustness_stability.csv
4. robustness_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VERSION = "2026-07-24-module04-v1"
REFERENCE_SPEC = "H3_main_primary_full"



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
        description="Validate robustness of the OOF pass-value framework."
    )
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument("--h1_pass_value", required=True, type=Path)
    parser.add_argument("--h3_pass_value", required=True, type=Path)
    parser.add_argument("--h5_pass_value", required=True, type=Path)
    parser.add_argument("--cohort_csv", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--player_thresholds",
        default="100,200,300",
        help="Comma-separated pass-count thresholds.",
    )
    parser.add_argument("--min_player_matches", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--expected_rows", type=int, default=54795)
    parser.add_argument("--expected_matches", type=int, default=110)
    parser.add_argument("--expected_teams", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def find_column(
    columns: list[str],
    candidates: list[str],
    label: str,
) -> str:
    lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise RuntimeError(
        f"Could not identify {label}. Tried: {', '.join(candidates)}"
    )


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


def parse_thresholds(text: str) -> list[int]:
    thresholds = sorted(
        {
            int(part.strip())
            for part in text.split(",")
            if part.strip()
        }
    )
    if not thresholds or any(value < 1 for value in thresholds):
        raise ValueError("--player_thresholds must contain positive integers.")
    return thresholds


def load_cohort_keys(path: Path) -> pd.DataFrame:
    cohort = read_table(path)
    event_col = find_column(
        cohort.columns.tolist(),
        ["id", "event_id", "pass_id"],
        "cohort event ID",
    )
    match_col = find_column(
        cohort.columns.tolist(),
        ["match_id", "matchid"],
        "cohort match ID",
    )
    cohort = cohort.rename(
        columns={event_col: "id", match_col: "match_id"}
    )
    cohort["id"] = cohort["id"].map(normalize_id)
    cohort["match_id"] = cohort["match_id"].map(normalize_id)
    keys = cohort[["id", "match_id"]].copy()
    if keys.duplicated(["id", "match_id"]).any():
        raise RuntimeError("Duplicate event-match keys in cohort file.")
    return keys


REQUIRED_VALUE_COLUMNS = {
    "id",
    "match_id",
    "team",
    "player",
    "xpass_success",
    "p_success_main_oof",
    "p_success_no_team_oof",
    "success_value",
    "failure_cost",
    "failure_cost_start",
    "failure_cost_opponent",
    "primary_open_play",
    "strict_open_play",
}


def load_value_cohort(
    path: Path,
    cohort_keys: pd.DataFrame,
    label: str,
    expected_rows: int,
    expected_matches: int,
    expected_teams: int,
) -> pd.DataFrame:
    print(f"[load] {label}: {path}")
    frame = read_table(path)

    missing = sorted(REQUIRED_VALUE_COLUMNS - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"{label} pass-value file is missing: {', '.join(missing)}"
        )

    frame["id"] = frame["id"].map(normalize_id)
    frame["match_id"] = frame["match_id"].map(normalize_id)

    if frame.duplicated(["id", "match_id"]).any():
        raise RuntimeError(f"Duplicate event-match keys in {label}.")

    data = frame.merge(
        cohort_keys,
        on=["id", "match_id"],
        how="inner",
        validate="one_to_one",
    )

    counts = {
        "rows": len(data),
        "matches": data["match_id"].nunique(),
        "teams": data["team"].nunique(),
    }
    expected = {
        "rows": expected_rows,
        "matches": expected_matches,
        "teams": expected_teams,
    }
    if counts != expected:
        raise RuntimeError(
            f"{label} cohort integrity failed. observed={counts}, expected={expected}"
        )

    numeric = [
        "xpass_success",
        "p_success_main_oof",
        "p_success_no_team_oof",
        "success_value",
        "failure_cost",
        "failure_cost_start",
        "failure_cost_opponent",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[numeric].isna().any().any():
        raise RuntimeError(f"{label} contains missing formal metrics.")

    data["team"] = data["team"].astype("string").fillna("Unknown")
    data["player"] = data["player"].astype("string").fillna("Unknown")
    data["primary_open_play"] = to_bool(data["primary_open_play"])
    data["strict_open_play"] = to_bool(data["strict_open_play"])

    return data


def build_specification(
    source: pd.DataFrame,
    spec_name: str,
    horizon: int,
    xpass_type: str,
    sample_type: str,
    failure_type: str,
) -> pd.DataFrame:
    if xpass_type == "main":
        probability_col = "p_success_main_oof"
    elif xpass_type == "no_team":
        probability_col = "p_success_no_team_oof"
    else:
        raise ValueError(xpass_type)

    if sample_type == "primary":
        mask = source["primary_open_play"]
    elif sample_type == "strict":
        mask = source["strict_open_play"]
    else:
        raise ValueError(sample_type)

    if failure_type == "full":
        failure_col = "failure_cost"
    elif failure_type == "start_only":
        failure_col = "failure_cost_start"
    elif failure_type == "opponent_only":
        failure_col = "failure_cost_opponent"
    else:
        raise ValueError(failure_type)

    data = source.loc[
        mask,
        [
            "id",
            "match_id",
            "team",
            "player",
            "xpass_success",
            probability_col,
            "success_value",
            failure_col,
        ],
    ].copy()

    data = data.rename(
        columns={
            probability_col: "xpass_probability",
            failure_col: "failure_cost_selected",
        }
    )

    p = data["xpass_probability"].to_numpy(dtype=np.float64)
    success_value = data["success_value"].to_numpy(dtype=np.float64)
    failure_cost = data["failure_cost_selected"].to_numpy(dtype=np.float64)
    observed = data["xpass_success"].to_numpy(dtype=np.float64)

    data["expected_success_value"] = p * success_value
    data["expected_failure_cost"] = (1.0 - p) * failure_cost
    data["expected_net_value"] = (
        data["expected_success_value"] - data["expected_failure_cost"]
    )
    data["realized_value"] = (
        observed * success_value - (1.0 - observed) * failure_cost
    )
    data["execution_residual"] = observed - p

    data["specification"] = spec_name
    data["horizon"] = horizon
    data["xpass_type"] = xpass_type
    data["sample_type"] = sample_type
    data["failure_type"] = failure_type
    return data


def summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    n = len(group)
    return {
        "n_passes": int(n),
        "n_matches": int(group["match_id"].nunique()),
        "actual_completion_rate": float(group["xpass_success"].mean()),
        "mean_xpass": float(group["xpass_probability"].mean()),
        "execution_residual_per_100_passes": float(
            100.0 * group["execution_residual"].sum() / n
        ),
        "expected_success_value_per_100_passes": float(
            100.0 * group["expected_success_value"].sum() / n
        ),
        "expected_failure_cost_per_100_passes": float(
            100.0 * group["expected_failure_cost"].sum() / n
        ),
        "expected_net_value_per_100_passes": float(
            100.0 * group["expected_net_value"].sum() / n
        ),
        "realized_value_per_100_passes": float(
            100.0 * group["realized_value"].sum() / n
        ),
    }


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    values_array = values.to_numpy(dtype=np.float64)
    weights_array = weights.to_numpy(dtype=np.float64)
    keep = (
        np.isfinite(values_array)
        & np.isfinite(weights_array)
        & (weights_array > 0)
    )
    values_array = values_array[keep]
    weights_array = weights_array[keep]
    if len(values_array) == 0:
        return float("nan")
    order = np.argsort(values_array)
    values_array = values_array[order]
    weights_array = weights_array[order]
    cutoff = weights_array.sum() / 2.0
    index = np.searchsorted(np.cumsum(weights_array), cutoff, side="left")
    return float(values_array[min(index, len(values_array) - 1)])


def quadrant(
    decision: float,
    execution: float,
    decision_reference: float,
    execution_reference: float,
) -> str:
    high_decision = decision >= decision_reference
    high_execution = execution >= execution_reference
    if high_decision and high_execution:
        return "High decision value / High execution"
    if high_decision:
        return "High decision value / Low execution"
    if high_execution:
        return "Low decision value / High execution"
    return "Low decision value / Low execution"


def player_summary_for_threshold(
    spec_data: pd.DataFrame,
    threshold: int,
    min_matches: int,
) -> pd.DataFrame:
    rows = []
    for (team, player), group in spec_data.groupby(
        ["team", "player"],
        sort=True,
        dropna=False,
    ):
        row = {
            "specification": str(group["specification"].iloc[0]),
            "team": str(team),
            "player": str(player),
            "player_threshold": threshold,
            **summarize_group(group),
        }
        row["eligible"] = bool(
            row["n_passes"] >= threshold
            and row["n_matches"] >= min_matches
            and str(player) != "Unknown"
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["decision_execution_quadrant"] = "Not eligible"

    eligible = summary["eligible"]
    for team, group in summary[eligible].groupby("team", sort=True):
        decision_ref = weighted_median(
            group["expected_net_value_per_100_passes"],
            group["n_passes"],
        )
        execution_ref = weighted_median(
            group["execution_residual_per_100_passes"],
            group["n_passes"],
        )
        mask = eligible & (summary["team"] == team)
        summary.loc[mask, "decision_reference"] = decision_ref
        summary.loc[mask, "execution_reference"] = execution_ref
        summary.loc[mask, "decision_execution_quadrant"] = [
            quadrant(d, e, decision_ref, execution_ref)
            for d, e in zip(
                summary.loc[mask, "expected_net_value_per_100_passes"],
                summary.loc[mask, "execution_residual_per_100_passes"],
            )
        ]

    summary["decision_rank_within_team"] = (
        summary.groupby("team")["expected_net_value_per_100_passes"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    summary["execution_rank_within_team"] = (
        summary.groupby("team")["execution_residual_per_100_passes"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    return summary


def mean_top_k_overlap(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    metric: str,
    top_k: int,
) -> float:
    overlaps = []
    teams = sorted(set(reference["team"]) & set(comparison["team"]))
    for team in teams:
        ref_team = reference[reference["team"] == team].nlargest(top_k, metric)
        cmp_team = comparison[comparison["team"] == team].nlargest(top_k, metric)
        ref_set = set(ref_team["player"])
        cmp_set = set(cmp_team["player"])
        denominator = min(top_k, len(ref_set), len(cmp_set))
        if denominator > 0:
            overlaps.append(len(ref_set & cmp_set) / denominator)
    return float(np.mean(overlaps)) if overlaps else float("nan")


def compare_players(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    top_k: int,
) -> dict[str, Any]:
    ref = reference[reference["eligible"]].copy()
    cmp = comparison[comparison["eligible"]].copy()
    merged = ref.merge(
        cmp,
        on=["team", "player"],
        suffixes=("_reference", "_comparison"),
        how="inner",
    )

    if len(merged) < 3:
        return {
            "common_eligible_players": int(len(merged)),
            "decision_spearman": float("nan"),
            "execution_spearman": float("nan"),
            "quadrant_agreement": float("nan"),
            "decision_top_k_overlap": float("nan"),
            "execution_top_k_overlap": float("nan"),
        }

    decision_spearman = merged[
        [
            "expected_net_value_per_100_passes_reference",
            "expected_net_value_per_100_passes_comparison",
        ]
    ].corr(method="spearman").iloc[0, 1]

    execution_spearman = merged[
        [
            "execution_residual_per_100_passes_reference",
            "execution_residual_per_100_passes_comparison",
        ]
    ].corr(method="spearman").iloc[0, 1]

    quadrant_agreement = (
        merged["decision_execution_quadrant_reference"]
        == merged["decision_execution_quadrant_comparison"]
    ).mean()

    decision_overlap = mean_top_k_overlap(
        ref,
        cmp,
        "expected_net_value_per_100_passes",
        top_k,
    )
    execution_overlap = mean_top_k_overlap(
        ref,
        cmp,
        "execution_residual_per_100_passes",
        top_k,
    )

    return {
        "common_eligible_players": int(len(merged)),
        "decision_spearman": float(decision_spearman),
        "execution_spearman": float(execution_spearman),
        "quadrant_agreement": float(quadrant_agreement),
        "decision_top_k_overlap": float(decision_overlap),
        "execution_top_k_overlap": float(execution_overlap),
    }


def load_horizon_performance(value_path: Path) -> dict[str, Any] | None:
    summary_path = value_path.parent / "pass_value_summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    return {
        "path": str(summary_path),
        "global_performance": summary.get("global_performance", {}),
        "counts": summary.get("counts", {}),
        "status": summary.get("status"),
    }


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.player_thresholds)

    paths = {
        "H1": args.h1_pass_value.resolve(),
        "H3": args.h3_pass_value.resolve(),
        "H5": args.h5_pass_value.resolve(),
    }
    cohort_path = args.cohort_csv.resolve()
    output_dir = args.output_dir.resolve()

    for path in [*paths.values(), cohort_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "team": output_dir / "robustness_team_results.csv",
        "player": output_dir / "robustness_player_results.csv",
        "stability": output_dir / "robustness_stability.csv",
        "summary": output_dir / "robustness_summary.json",
    }

    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Outputs already exist; use --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )
    for path in existing:
        path.unlink()

    cohort_keys = load_cohort_keys(cohort_path)
    horizon_data = {
        label: load_value_cohort(
            path,
            cohort_keys,
            label,
            args.expected_rows,
            args.expected_matches,
            args.expected_teams,
        )
        for label, path in paths.items()
    }

    specifications = [
        (
            "H3_main_primary_full",
            horizon_data["H3"],
            3,
            "main",
            "primary",
            "full",
        ),
        (
            "H1_main_primary_full",
            horizon_data["H1"],
            1,
            "main",
            "primary",
            "full",
        ),
        (
            "H5_main_primary_full",
            horizon_data["H5"],
            5,
            "main",
            "primary",
            "full",
        ),
        (
            "H3_no_team_primary_full",
            horizon_data["H3"],
            3,
            "no_team",
            "primary",
            "full",
        ),
        (
            "H3_main_strict_full",
            horizon_data["H3"],
            3,
            "main",
            "strict",
            "full",
        ),
        (
            "H3_main_primary_start_only",
            horizon_data["H3"],
            3,
            "main",
            "primary",
            "start_only",
        ),
        (
            "H3_main_primary_opponent_only",
            horizon_data["H3"],
            3,
            "main",
            "primary",
            "opponent_only",
        ),
    ]

    spec_data: dict[str, pd.DataFrame] = {}
    team_rows = []
    player_frames = []

    for (
        spec_name,
        source,
        horizon,
        xpass_type,
        sample_type,
        failure_type,
    ) in specifications:
        data = build_specification(
            source,
            spec_name,
            horizon,
            xpass_type,
            sample_type,
            failure_type,
        )
        spec_data[spec_name] = data

        print(
            f"[spec] {spec_name}: "
            f"passes={len(data):,}, matches={data['match_id'].nunique():,}"
        )

        for team, group in data.groupby("team", sort=True):
            team_rows.append(
                {
                    "specification": spec_name,
                    "horizon": horizon,
                    "xpass_type": xpass_type,
                    "sample_type": sample_type,
                    "failure_type": failure_type,
                    "team": str(team),
                    **summarize_group(group),
                }
            )

        for threshold in thresholds:
            player_frames.append(
                player_summary_for_threshold(
                    data,
                    threshold,
                    args.min_player_matches,
                )
            )

    team_results = pd.DataFrame(team_rows)
    player_results = pd.concat(player_frames, ignore_index=True)

    stability_rows = []
    for threshold in thresholds:
        reference = player_results[
            (player_results["specification"] == REFERENCE_SPEC)
            & (player_results["player_threshold"] == threshold)
        ].copy()

        for spec_name, *_ in specifications:
            comparison = player_results[
                (player_results["specification"] == spec_name)
                & (player_results["player_threshold"] == threshold)
            ].copy()
            metrics = compare_players(reference, comparison, args.top_k)

            ref_team = team_results[
                team_results["specification"] == REFERENCE_SPEC
            ][["team", "expected_net_value_per_100_passes"]].rename(
                columns={
                    "expected_net_value_per_100_passes": "reference_team_value"
                }
            )
            cmp_team = team_results[
                team_results["specification"] == spec_name
            ][["team", "expected_net_value_per_100_passes"]].rename(
                columns={
                    "expected_net_value_per_100_passes": "comparison_team_value"
                }
            )
            team_compare = ref_team.merge(cmp_team, on="team", how="inner")
            sign_agreement = (
                np.sign(team_compare["reference_team_value"])
                == np.sign(team_compare["comparison_team_value"])
            ).mean()

            reference_order = tuple(
                team_compare.sort_values(
                    "reference_team_value",
                    ascending=False,
                )["team"]
            )
            comparison_order = tuple(
                team_compare.sort_values(
                    "comparison_team_value",
                    ascending=False,
                )["team"]
            )

            stability_rows.append(
                {
                    "reference_specification": REFERENCE_SPEC,
                    "comparison_specification": spec_name,
                    "player_threshold": threshold,
                    "minimum_matches": args.min_player_matches,
                    **metrics,
                    "team_sign_agreement": float(sign_agreement),
                    "team_order_identical": bool(
                        reference_order == comparison_order
                    ),
                    "reference_team_order": " > ".join(reference_order),
                    "comparison_team_order": " > ".join(comparison_order),
                }
            )

    stability = pd.DataFrame(stability_rows)

    team_results.to_csv(
        outputs["team"],
        index=False,
        encoding="utf-8-sig",
    )
    player_results.to_csv(
        outputs["player"],
        index=False,
        encoding="utf-8-sig",
    )
    stability.to_csv(
        outputs["stability"],
        index=False,
        encoding="utf-8-sig",
    )

    horizon_performance = {
        label: load_horizon_performance(path)
        for label, path in paths.items()
    }

    reference_200 = stability[
        (stability["player_threshold"] == 200)
        & (
            stability["comparison_specification"]
            != REFERENCE_SPEC
        )
    ]

    summary = {
        "status": "PASS",
        "version": VERSION,
        "reference_specification": REFERENCE_SPEC,
        "formal_specifications": [
            {
                "name": name,
                "horizon": horizon,
                "xpass_type": xpass_type,
                "sample_type": sample_type,
                "failure_type": failure_type,
                "passes": int(len(spec_data[name])),
                "matches": int(spec_data[name]["match_id"].nunique()),
            }
            for (
                name,
                _,
                horizon,
                xpass_type,
                sample_type,
                failure_type,
            ) in specifications
        ],
        "player_thresholds": thresholds,
        "minimum_player_matches": args.min_player_matches,
        "top_k": args.top_k,
        "horizon_model_performance": horizon_performance,
        "primary_threshold_200_stability_range": {
            "minimum_decision_spearman": float(
                reference_200["decision_spearman"].min()
            ),
            "minimum_execution_spearman": float(
                reference_200["execution_spearman"].min()
            ),
            "minimum_quadrant_agreement": float(
                reference_200["quadrant_agreement"].min()
            ),
            "minimum_team_sign_agreement": float(
                reference_200["team_sign_agreement"].min()
            ),
        },
        "interpretation_rules": {
            "decision_rank_stable": "Spearman rho >= 0.80",
            "execution_rank_stable": "Spearman rho >= 0.80",
            "quadrant_reasonably_stable": "agreement >= 0.70",
            "team_direction_stable": "sign agreement = 1.00",
            "note": (
                "These are prespecified descriptive robustness benchmarks, "
                "not null-hypothesis significance tests."
            ),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }

    outputs["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    print("=" * 100)
    print("Robustness and sensitivity analysis completed")
    print("=" * 100)
    print(f"Specifications: {len(specifications)}")
    print(f"Player thresholds: {thresholds}")
    print(f"Team-result rows: {len(team_results):,}")
    print(f"Player-result rows: {len(player_results):,}")
    print(f"Stability-comparison rows: {len(stability):,}")
    print("Status: PASS")
    print(f"Team results: {outputs['team']}")
    print(f"Player results: {outputs['player']}")
    print(f"Stability results: {outputs['stability']}")
    print(f"Summary: {outputs['summary']}")


if __name__ == "__main__":
    main()
