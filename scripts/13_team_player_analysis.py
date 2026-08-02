# -*- coding: utf-8 -*-
"""
Summarize team- and player-level pass value for three prespecified
complete team-season cohorts.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = "2026-08-02-team-player-analysis-v1"



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
    p = argparse.ArgumentParser(description="Three-team pass-value analysis.")
    p.add_argument("--version", action="version", version=VERSION)
    p.add_argument("--pass_value", required=True, type=Path)
    p.add_argument("--cohort_csv", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--min_player_passes", type=int, default=200)
    p.add_argument("--min_player_matches", type=int, default=5)
    p.add_argument("--bootstrap_reps", type=int, default=1000)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--expected_rows", type=int, default=54795)
    p.add_argument("--expected_matches", type=int, default=110)
    p.add_argument("--expected_teams", type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def norm_id(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s


def find_col(columns, names, required=True):
    lookup = {c.lower(): c for c in columns}
    for name in names:
        if name in columns:
            return name
        if name.lower() in lookup:
            return lookup[name.lower()]
    if required:
        raise RuntimeError(f"Missing required column; tried {names}")
    return None


def per100(value, n):
    return np.nan if n <= 0 else 100.0 * float(value) / float(n)


def summarize(group: pd.DataFrame) -> dict:
    n = len(group)
    out = {
        "n_matches": int(group["match_id"].nunique()),
        "n_passes": int(n),
        "actual_completion_rate": float(group["xpass_success"].mean()),
        "mean_xpass": float(group["p_success_main_oof"].mean()),
        "completion_over_expected": float(group["execution_residual"].mean()),
        "mean_success_value": float(group["success_value"].mean()),
        "mean_failure_cost": float(group["failure_cost"].mean()),
    }
    for c in [
        "expected_success_value", "expected_failure_cost", "expected_net_value",
        "realized_value", "execution_residual",
    ]:
        total = float(group[c].sum())
        out[f"total_{c}"] = total
        out[f"{c}_per_100_passes"] = per100(total, n)
    return out


def match_stats(group: pd.DataFrame) -> pd.DataFrame:
    return group.groupby("match_id", sort=False).agg(
        n_passes=("xpass_success", "size"),
        actual_success=("xpass_success", "sum"),
        expected_success=("p_success_main_oof", "sum"),
        expected_success_value=("expected_success_value", "sum"),
        expected_failure_cost=("expected_failure_cost", "sum"),
        expected_net=("expected_net_value", "sum"),
        realized=("realized_value", "sum"),
        execution=("execution_residual", "sum"),
    ).reset_index()


def bootstrap_ci(stats: pd.DataFrame, reps: int, rng) -> dict:
    if stats.empty:
        return {}
    arrays = {c: stats[c].to_numpy(float) for c in stats.columns if c != "match_id"}
    m = len(stats)
    values = {
        "actual_completion_rate": [],
        "mean_xpass": [],
        "execution_residual_per_100_passes": [],
        "expected_success_value_per_100_passes": [],
        "expected_failure_cost_per_100_passes": [],
        "expected_net_value_per_100_passes": [],
        "realized_value_per_100_passes": [],
    }
    for _ in range(reps):
        idx = rng.integers(0, m, size=m)
        n = arrays["n_passes"][idx].sum()
        if n <= 0:
            continue
        values["actual_completion_rate"].append(arrays["actual_success"][idx].sum() / n)
        values["mean_xpass"].append(arrays["expected_success"][idx].sum() / n)
        values["execution_residual_per_100_passes"].append(
            100 * arrays["execution"][idx].sum() / n
        )
        values["expected_success_value_per_100_passes"].append(
            100 * arrays["expected_success_value"][idx].sum() / n
        )
        values["expected_failure_cost_per_100_passes"].append(
            100 * arrays["expected_failure_cost"][idx].sum() / n
        )
        values["expected_net_value_per_100_passes"].append(
            100 * arrays["expected_net"][idx].sum() / n
        )
        values["realized_value_per_100_passes"].append(
            100 * arrays["realized"][idx].sum() / n
        )
    out = {}
    for metric, vals in values.items():
        lo, hi = np.quantile(np.asarray(vals), [0.025, 0.975])
        out[f"{metric}_ci_low"] = float(lo)
        out[f"{metric}_ci_high"] = float(hi)
    return out


def weighted_median(values, weights):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[keep], weights[keep]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    pos = np.searchsorted(np.cumsum(weights), weights.sum() / 2)
    return float(values[min(pos, len(values) - 1)])


def quadrant(d, e, d_ref, e_ref):
    if d >= d_ref and e >= e_ref:
        return "High decision value / High execution"
    if d >= d_ref:
        return "High decision value / Low execution"
    if e >= e_ref:
        return "Low decision value / High execution"
    return "Low decision value / Low execution"


def main():
    a = parse_args()
    pass_path = a.pass_value.resolve()
    cohort_path = a.cohort_csv.resolve()
    out_dir = a.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "team": out_dir / "team_summary.csv",
        "player": out_dir / "player_summary.csv",
        "summary": out_dir / "team_analysis_summary.json",
    }
    existing = [p for p in outputs.values() if p.exists()]
    if existing and not a.overwrite:
        raise FileExistsError("Outputs exist; use --overwrite:\n" + "\n".join(map(str, existing)))
    for p in existing:
        p.unlink()

    print(f"[load] pass value: {pass_path}")
    value = read_table(pass_path)
    print(f"[load] cohort: {cohort_path}")
    cohort = read_table(cohort_path)

    required = {
        "id", "match_id", "team", "player", "xpass_success",
        "p_success_main_oof", "success_value", "failure_cost",
        "expected_net_value", "realized_value", "execution_residual",
    }
    missing = sorted(required - set(value.columns))
    if missing:
        raise RuntimeError("Pass-value file missing: " + ", ".join(missing))

    expected_success_col = find_col(
        value.columns,
        ["expected_success_value", "expected_gain"],
        required=True,
    )
    expected_failure_col = find_col(
        value.columns,
        ["expected_failure_cost", "expected_risk"],
        required=True,
    )

    if expected_success_col != "expected_success_value":
        value = value.rename(
            columns={expected_success_col: "expected_success_value"}
        )
    if expected_failure_col != "expected_failure_cost":
        value = value.rename(
            columns={expected_failure_col: "expected_failure_cost"}
        )

    print(
        "[schema] expected success column: "
        f"{expected_success_col} -> expected_success_value"
    )
    print(
        "[schema] expected failure column: "
        f"{expected_failure_col} -> expected_failure_cost"
    )

    cid = find_col(cohort.columns, ["id", "event_id", "pass_id"])
    cmatch = find_col(cohort.columns, ["match_id", "matchid"])
    cohort = cohort.rename(columns={cid: "id", cmatch: "match_id"})

    value["id"] = value["id"].map(norm_id)
    value["match_id"] = value["match_id"].map(norm_id)
    cohort["id"] = cohort["id"].map(norm_id)
    cohort["match_id"] = cohort["match_id"].map(norm_id)

    if value.duplicated(["id", "match_id"]).any():
        raise RuntimeError("Duplicate event-match keys in pass-value file.")
    if cohort.duplicated(["id", "match_id"]).any():
        raise RuntimeError("Duplicate event-match keys in cohort file.")

    meta_cols = ["id", "match_id"]
    for names in [
        ["team", "cohort_team", "selected_team", "target_team"],
        ["competition", "competition_name"],
        ["season", "season_name"],
        ["cohort", "cohort_name", "team_season"],
    ]:
        col = find_col(cohort.columns, names, required=False)
        if col and col not in meta_cols:
            meta_cols.append(col)

    meta = cohort[meta_cols].copy()
    meta = meta.rename(
        columns={c: f"cohort_{c}" for c in meta_cols if c not in {"id", "match_id"}}
    )
    data = value.merge(meta, on=["id", "match_id"], how="inner", validate="one_to_one")

    if len(data) != len(cohort):
        raise RuntimeError(
            f"Cohort merge incomplete: cohort={len(cohort):,}, matched={len(data):,}"
        )

    numeric = [
        "xpass_success", "p_success_main_oof", "success_value", "failure_cost",
        "expected_success_value", "expected_failure_cost", "expected_net_value",
        "realized_value", "execution_residual",
    ]
    for c in numeric:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    if data[numeric].isna().any().any():
        raise RuntimeError("Missing numeric values after cohort merge.")

    data["team"] = data["team"].astype("string").fillna("Unknown")
    data["player"] = data["player"].astype("string").fillna("Unknown")

    counts = {
        "rows": int(len(data)),
        "matches": int(data["match_id"].nunique()),
        "teams": int(data["team"].nunique()),
        "players": int(data["player"].nunique()),
    }
    checks = {
        "all_cohort_rows_matched": len(data) == len(cohort),
        "expected_rows": counts["rows"] == a.expected_rows,
        "expected_matches": counts["matches"] == a.expected_matches,
        "expected_teams": counts["teams"] == a.expected_teams,
        "unique_event_match_keys": not data.duplicated(["id", "match_id"]).any(),
        "complete_formal_metrics": not data[numeric].isna().any().any(),
    }
    if not all(checks.values()):
        print(json.dumps(checks, ensure_ascii=False, indent=2, default=json_default))
        raise RuntimeError("Locked cohort integrity checks failed.")

    rng = np.random.default_rng(a.random_state)

    team_rows = []
    for team, g in data.groupby("team", sort=True):
        row = {"team": str(team), **summarize(g)}
        row.update(bootstrap_ci(match_stats(g), a.bootstrap_reps, rng))
        team_rows.append(row)
    team_summary = pd.DataFrame(team_rows)

    player_rows = []
    for (team, player), g in data.groupby(["team", "player"], sort=True):
        row = {"team": str(team), "player": str(player), **summarize(g)}
        row["eligible_primary"] = bool(
            len(g) >= a.min_player_passes
            and g["match_id"].nunique() >= a.min_player_matches
            and str(player) != "Unknown"
        )
        row.update(bootstrap_ci(match_stats(g), a.bootstrap_reps, rng))
        player_rows.append(row)
    player_summary = pd.DataFrame(player_rows)

    player_summary["decision_execution_quadrant"] = "Not eligible"
    player_summary["team_decision_reference"] = np.nan
    player_summary["team_execution_reference"] = np.nan

    for team, g in player_summary[player_summary["eligible_primary"]].groupby("team"):
        d_ref = weighted_median(
            g["expected_net_value_per_100_passes"], g["n_passes"]
        )
        e_ref = weighted_median(
            g["execution_residual_per_100_passes"], g["n_passes"]
        )
        mask = (player_summary["team"] == team) & player_summary["eligible_primary"]
        player_summary.loc[mask, "team_decision_reference"] = d_ref
        player_summary.loc[mask, "team_execution_reference"] = e_ref
        player_summary.loc[mask, "decision_execution_quadrant"] = [
            quadrant(d, e, d_ref, e_ref)
            for d, e in zip(
                player_summary.loc[mask, "expected_net_value_per_100_passes"],
                player_summary.loc[mask, "execution_residual_per_100_passes"],
            )
        ]

    player_summary["decision_rank_within_team"] = (
        player_summary.groupby("team")["expected_net_value_per_100_passes"]
        .rank(ascending=False, method="min").astype("Int64")
    )
    player_summary["execution_rank_within_team"] = (
        player_summary.groupby("team")["execution_residual_per_100_passes"]
        .rank(ascending=False, method="min").astype("Int64")
    )

    team_summary.to_csv(outputs["team"], index=False, encoding="utf-8-sig")
    player_summary.to_csv(outputs["player"], index=False, encoding="utf-8-sig")

    summary = {
        "status": "PASS",
        "version": VERSION,
        "scope": (
            "Application of locked full-sample OOF xPass and PCTV values to "
            "three supplementary complete team-season cohorts."
        ),
        "inputs": {"pass_value": str(pass_path), "cohort_csv": str(cohort_path)},
        "counts": counts,
        "checks": checks,
        "player_primary_eligibility": {
            "minimum_passes": a.min_player_passes,
            "minimum_matches": a.min_player_matches,
            "eligible_players": int(player_summary["eligible_primary"].sum()),
            "note": "All players are retained; threshold affects primary interpretation only.",
        },
        "uncertainty": {
            "method": "nonparametric bootstrap clustered by match",
            "repetitions": a.bootstrap_reps,
            "confidence_interval": "percentile 95%",
            "random_state": a.random_state,
        },
        "metric_definitions": {
            "expected_success_value": (
                "p_success_main_oof * success_value"
            ),
            "expected_failure_cost": (
                "(1 - p_success_main_oof) * failure_cost"
            ),
            "expected_net_value": (
                "expected_success_value - expected_failure_cost"
            ),
        },
        "quadrant_definition": {
            "decision_axis": "expected_net_value_per_100_passes",
            "execution_axis": "execution_residual_per_100_passes",
            "reference": "pass-count-weighted within-team median among eligible players",
            "interpretation": "comparative within-team classification, not an absolute cutoff",
        },
        "formal_outputs": {k: str(v) for k, v in outputs.items()},
    }
    outputs["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
    )

    print("=" * 100)
    print("Three-team team/player analysis completed")
    print("=" * 100)
    print(f"Passes: {counts['rows']:,}")
    print(f"Matches: {counts['matches']:,}")
    print(f"Teams: {counts['teams']:,}")
    print(f"Players: {counts['players']:,}")
    print(
        "Primary-eligible players: "
        f"{int(player_summary['eligible_primary'].sum()):,}"
    )
    print(
        f"Bootstrap: {a.bootstrap_reps:,} match-cluster repetitions"
    )
    print("Status: PASS")
    print(f"Team summary: {outputs['team']}")
    print(f"Player summary: {outputs['player']}")
    print(f"Analysis summary: {outputs['summary']}")


if __name__ == "__main__":
    main()
