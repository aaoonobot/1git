#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
20_player_decision_value_analysis.py

Player-level application of the locked H=3 pass risk-reward framework.

Primary question
----------------
Can traditional passing efficiency conceal differences in passing decision
value? The module separates:
1. pass completion efficiency and modeled difficulty;
2. expected successful-state gain;
3. expected failure exposure;
4. execution relative to xPass expectation.

The analysis is descriptive and illustrative. It does not rank overall player
quality and does not estimate causal player effects.

Required inputs
---------------
1. three_team_player_event_data.csv.gz
2. player_summary_final.csv
3. publication_style.py in the same folder as this script

Main outputs
------------
player_role_assignments.csv
player_risk_reward_profiles.csv
player_profile_correlations.csv
player_rank_discordance.csv
traditional_similarity_pairs.csv
selected_player_pairs.csv
selected_player_pair_profiles.csv
Figure_4_player_decision_value.png/.pdf/.svg
module07_results_text.txt
module07_figure_caption.txt
module07_audit.json
SHA256SUMS.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    from publication_style import (
        STYLE,
        add_panel_label,
        clean_axis,
        configure_matplotlib,
        save_figure_all_formats,
        team_color_map,
    )
except ImportError as exc:
    raise ImportError(
        "publication_style.py must be saved in the same folder as "
        "20_player_decision_value_analysis.py."
    ) from exc


SCRIPT_VERSION = "2026-07-29-module07-final-v2-top-journal"
FULL_SAMPLE_ROWS = 3_396_912
FULL_SAMPLE_BASELINE_PER_100 = -0.9528630592451951

TEAM_ORDER = [
    "Bayer Leverkusen",
    "Athletic Club",
    "AS Monaco",
]

TEAM_DISPLAY = {
    "Bayer Leverkusen": "Bayer Leverkusen",
    "Athletic Club": "Athletic Club",
    "AS Monaco": "AS Monaco",
}

ROLE_ORDER = [
    "Central defender",
    "Fullback/wing-back",
    "Central/defensive midfielder",
    "Attacking/wide midfielder",
    "Forward",
    "Goalkeeper",
]

ROLE_MARKERS = {
    "Central defender": "s",
    "Fullback/wing-back": "D",
    "Central/defensive midfielder": "o",
    "Attacking/wide midfielder": "^",
    "Forward": "P",
    "Goalkeeper": "X",
}

ROLE_SHORT = {
    "Central defender": "Central defender",
    "Fullback/wing-back": "Fullback/wing-back",
    "Central/defensive midfielder": "Central/defensive midfielder",
    "Attacking/wide midfielder": "Attacking/wide midfielder",
    "Forward": "Forward",
    "Goalkeeper": "Goalkeeper",
}

POSITION_TO_ROLE = {
    "Goalkeeper": "Goalkeeper",
    "Left Center Back": "Central defender",
    "Right Center Back": "Central defender",
    "Center Back": "Central defender",
    "Left Back": "Fullback/wing-back",
    "Right Back": "Fullback/wing-back",
    "Left Wing Back": "Fullback/wing-back",
    "Right Wing Back": "Fullback/wing-back",
    "Left Defensive Midfield": "Central/defensive midfielder",
    "Right Defensive Midfield": "Central/defensive midfielder",
    "Center Defensive Midfield": "Central/defensive midfielder",
    "Left Center Midfield": "Central/defensive midfielder",
    "Right Center Midfield": "Central/defensive midfielder",
    "Center Attacking Midfield": "Attacking/wide midfielder",
    "Left Attacking Midfield": "Attacking/wide midfielder",
    "Right Attacking Midfield": "Attacking/wide midfielder",
    "Left Midfield": "Attacking/wide midfielder",
    "Right Midfield": "Attacking/wide midfielder",
    "Left Wing": "Attacking/wide midfielder",
    "Right Wing": "Attacking/wide midfielder",
    "Center Forward": "Forward",
    "Left Center Forward": "Forward",
    "Right Center Forward": "Forward",
}

SHORT_NAME_OVERRIDES = {
    "Fábio Alexandre da Silva Coentrão": "Fábio Coentrão",
    "Mikel Balenziaga Oruesagasti": "Mikel Balenziaga",
    "Stephan El Shaarawy": "El Shaarawy",
    "Mikel San José Domínguez": "Mikel San José",
    "João Filipe Iria Santos Moutinho": "João Moutinho",
    "Ricardo Alberto Silveira de Carvalho": "Ricardo Carvalho",
    "Wallace Fortuna dos Santos": "Wallace",
    "Xabier Etxeita Gorritxategi": "Xabier Etxeita",
    "Óscar de Marcos Arana": "Óscar de Marcos",
    "Elderson Uwa Echiejile": "Elderson Echiejile",
}

REQUIRED_EVENT_COLUMNS = {
    "event_id",
    "match_id",
    "cohort",
    "team",
    "player",
    "position",
    "pressure_status",
    "pass_height_group",
    "xpass_success",
    "p_success_main_oof",
    "execution_residual",
    "expected_success_value",
    "expected_failure_cost",
    "expected_net_value",
    "expected_net_value_centered",
}

REQUIRED_SUMMARY_COLUMNS = {
    "team",
    "player",
    "n_matches",
    "n_passes",
    "actual_completion_rate",
    "mean_xpass",
    "completion_over_expected",
    "expected_success_value_per_100_passes",
    "expected_failure_cost_per_100_passes",
    "expected_net_value_per_100_passes",
    "execution_residual_per_100_passes",
    "eligible_primary_rechecked",
    "centered_expected_net_value_per_100_passes",
    "centered_expected_net_value_per_100_passes_ci_low",
    "centered_expected_net_value_per_100_passes_ci_high",
    "execution_residual_per_100_passes_ci_low",
    "execution_residual_per_100_passes_ci_high",
}


# -----------------------------------------------------------------------------
# Command-line interface and general helpers
# -----------------------------------------------------------------------------


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
            "Analyze player-level decision value, execution, role profiles, "
            "and traditional-efficiency-matched player pairs."
        )
    )
    parser.add_argument("--player_event_csv", required=True, type=Path)
    parser.add_argument("--player_summary_csv", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--full_sample_rows",
        type=int,
        default=FULL_SAMPLE_ROWS,
    )
    parser.add_argument(
        "--full_sample_baseline_per_100",
        type=float,
        default=FULL_SAMPLE_BASELINE_PER_100,
    )
    parser.add_argument("--min_passes", type=int, default=200)
    parser.add_argument("--min_matches", type=int, default=5)
    parser.add_argument("--minimum_role_share", type=float, default=0.60)
    parser.add_argument(
        "--completion_caliper",
        type=float,
        default=0.025,
        help="Maximum absolute difference in completion rate.",
    )
    parser.add_argument(
        "--xpass_caliper",
        type=float,
        default=0.030,
        help="Maximum absolute difference in mean xPass.",
    )
    parser.add_argument(
        "--volume_relative_caliper",
        type=float,
        default=0.30,
        help="Maximum relative difference in passes per observed match.",
    )
    parser.add_argument("--maximum_selected_pairs", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sha256sums(directory: Path, filename: str = "SHA256SUMS.txt") -> Path:
    manifest = directory / filename
    lines = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != filename:
            lines.append(f"{sha256_file(path)}  {path.name}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}\n"
            "Use --overwrite only when intentionally replacing outputs."
        )
    if overwrite:
        for path in output_dir.iterdir():
            if path.is_file():
                path.unlink()


def short_name(name: str) -> str:
    if name in SHORT_NAME_OVERRIDES:
        return SHORT_NAME_OVERRIDES[name]
    parts = str(name).strip().split()
    if len(parts) <= 2:
        return str(name)
    return f"{parts[0]} {parts[-1]}"


def safe_spearman(frame: pd.DataFrame, x: str, y: str) -> float:
    selected = frame[[x, y]].dropna()
    if len(selected) < 3:
        return float("nan")
    if selected[x].nunique() < 2 or selected[y].nunique() < 2:
        return float("nan")
    return float(selected.corr(method="spearman").iloc[0, 1])


# -----------------------------------------------------------------------------
# Role assignment and player profile construction
# -----------------------------------------------------------------------------

def assign_roles(events: pd.DataFrame, minimum_share: float) -> pd.DataFrame:
    unknown_positions = sorted(set(events["position"]) - set(POSITION_TO_ROLE))
    if unknown_positions:
        raise RuntimeError(
            "Unmapped StatsBomb positions: " + ", ".join(unknown_positions)
        )

    working = events[["team", "player", "position"]].copy()
    working["role_group"] = working["position"].map(POSITION_TO_ROLE)

    detailed = (
        working.groupby(["team", "player", "role_group"], sort=True)
        .size()
        .rename("role_passes")
        .reset_index()
    )
    totals = (
        working.groupby(["team", "player"], sort=True)
        .size()
        .rename("all_role_passes")
        .reset_index()
    )
    detailed = detailed.merge(totals, on=["team", "player"], validate="many_to_one")
    detailed["role_share"] = detailed["role_passes"] / detailed["all_role_passes"]

    dominant = (
        detailed.sort_values(
            ["team", "player", "role_passes", "role_group"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["team", "player"])
        .rename(
            columns={
                "role_group": "dominant_role",
                "role_passes": "dominant_role_passes",
                "role_share": "dominant_role_share",
            }
        )
    )
    dominant["stable_role"] = dominant["dominant_role_share"] >= minimum_share
    dominant["analysis_role"] = np.where(
        dominant["stable_role"],
        dominant["dominant_role"],
        "Mixed role",
    )

    role_detail_text = (
        detailed.sort_values(["team", "player", "role_passes"], ascending=[True, True, False])
        .assign(
            role_component=lambda x: (
                x["role_group"]
                + ": "
                + (100.0 * x["role_share"]).round(1).astype(str)
                + "%"
            )
        )
        .groupby(["team", "player"], sort=True)["role_component"]
        .agg("; ".join)
        .rename("role_distribution")
        .reset_index()
    )

    return dominant.merge(
        role_detail_text,
        on=["team", "player"],
        validate="one_to_one",
    )[
        [
            "team",
            "player",
            "dominant_role",
            "dominant_role_passes",
            "all_role_passes",
            "dominant_role_share",
            "stable_role",
            "analysis_role",
            "role_distribution",
        ]
    ]


def build_context_profiles(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby(["team", "player"], sort=True)
        .agg(
            event_rows=("event_id", "size"),
            event_matches=("match_id", "nunique"),
            event_completion_rate=("xpass_success", "mean"),
            event_mean_xpass=("p_success_main_oof", "mean"),
            under_pressure_share=(
                "pressure_status",
                lambda values: float(np.mean(values == "Under pressure")),
            ),
            ground_pass_share=(
                "pass_height_group",
                lambda values: float(np.mean(values == "Ground Pass")),
            ),
            low_pass_share=(
                "pass_height_group",
                lambda values: float(np.mean(values == "Low Pass")),
            ),
            high_pass_share=(
                "pass_height_group",
                lambda values: float(np.mean(values == "High Pass")),
            ),
            event_esv_per_100=(
                "expected_success_value",
                lambda values: 100.0 * float(np.mean(values)),
            ),
            event_efc_per_100=(
                "expected_failure_cost",
                lambda values: 100.0 * float(np.mean(values)),
            ),
            event_env_per_100=(
                "expected_net_value",
                lambda values: 100.0 * float(np.mean(values)),
            ),
            event_centered_env_per_100=(
                "expected_net_value_centered",
                lambda values: 100.0 * float(np.mean(values)),
            ),
            event_execution_per_100=(
                "execution_residual",
                lambda values: 100.0 * float(np.mean(values)),
            ),
        )
        .reset_index()
    )


def profile_quadrant(decision: float, execution: float) -> str:
    if decision >= 0.0 and execution >= 0.0:
        return "Above-average decision / Above-expected execution"
    if decision >= 0.0:
        return "Above-average decision / Below-expected execution"
    if execution >= 0.0:
        return "Below-average decision / Above-expected execution"
    return "Below-average decision / Below-expected execution"


def build_player_profiles(
    summary: pd.DataFrame,
    context: pd.DataFrame,
    roles: pd.DataFrame,
    min_passes: int,
    min_matches: int,
) -> pd.DataFrame:
    profiles = summary.merge(
        context,
        on=["team", "player"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not profiles["_merge"].eq("both").all():
        raise RuntimeError(
            "Player identities differ between event and summary inputs."
        )
    profiles = profiles.drop(columns="_merge")
    profiles = profiles.merge(roles, on=["team", "player"], validate="one_to_one")

    profiles["passes_per_observed_match"] = (
        profiles["n_passes"] / profiles["n_matches"]
    )
    profiles["eligible_summary_flag"] = to_bool(profiles["eligible_primary_rechecked"])
    profiles["eligible_by_locked_threshold"] = (
        (pd.to_numeric(profiles["n_passes"], errors="coerce") >= min_passes)
        & (pd.to_numeric(profiles["n_matches"], errors="coerce") >= min_matches)
        & profiles["player"].astype("string").fillna("").str.strip().ne("")
        & profiles["player"].astype("string").fillna("").str.strip().ne("Unknown")
    )
    profiles["primary_eligible"] = (
        profiles["eligible_summary_flag"]
        & profiles["eligible_by_locked_threshold"]
    )
    profiles["primary_outfield_role_stable"] = (
        profiles["primary_eligible"]
        & profiles["stable_role"]
        & profiles["analysis_role"].ne("Goalkeeper")
    )

    profiles["decision_execution_profile"] = [
        profile_quadrant(decision, execution)
        for decision, execution in zip(
            profiles["centered_expected_net_value_per_100_passes"],
            profiles["execution_residual_per_100_passes"],
        )
    ]

    profiles["completion_rank_within_role"] = pd.Series(
        pd.NA, index=profiles.index, dtype="Float64"
    )
    profiles["decision_value_rank_within_role"] = pd.Series(
        pd.NA, index=profiles.index, dtype="Float64"
    )
    primary = profiles["primary_outfield_role_stable"]
    profiles.loc[primary, "completion_rank_within_role"] = (
        profiles.loc[primary]
        .groupby("analysis_role")["actual_completion_rate"]
        .rank(method="average", ascending=False)
        .astype(float)
    )
    profiles.loc[primary, "decision_value_rank_within_role"] = (
        profiles.loc[primary]
        .groupby("analysis_role")["centered_expected_net_value_per_100_passes"]
        .rank(method="average", ascending=False)
        .astype(float)
    )
    profiles["completion_minus_decision_rank"] = (
        profiles["completion_rank_within_role"]
        - profiles["decision_value_rank_within_role"]
    )

    # Exact numerical identity checks against the event-level input.
    identity_pairs = {
        "n_passes": "event_rows",
        "n_matches": "event_matches",
        "actual_completion_rate": "event_completion_rate",
        "mean_xpass": "event_mean_xpass",
        "expected_success_value_per_100_passes": "event_esv_per_100",
        "expected_failure_cost_per_100_passes": "event_efc_per_100",
        "expected_net_value_per_100_passes": "event_env_per_100",
        "centered_expected_net_value_per_100_passes": "event_centered_env_per_100",
        "execution_residual_per_100_passes": "event_execution_per_100",
    }
    for summary_col, event_col in identity_pairs.items():
        difference = np.abs(
            pd.to_numeric(profiles[summary_col], errors="coerce")
            - pd.to_numeric(profiles[event_col], errors="coerce")
        )
        profiles[f"audit_difference__{summary_col}"] = difference

    return profiles.sort_values(
        ["team", "analysis_role", "player"]
    ).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Correlations, rank discordance, and traditional-profile matching
# -----------------------------------------------------------------------------

def build_correlations(primary: pd.DataFrame) -> pd.DataFrame:
    variables = [
        ("actual_completion_rate", "Pass-completion rate"),
        ("mean_xpass", "Mean xPass"),
        ("passes_per_observed_match", "Passes per observed match"),
        ("execution_residual_per_100_passes", "Execution residual per 100 passes"),
        ("expected_success_value_per_100_passes", "Expected success value per 100 passes"),
        ("expected_failure_cost_per_100_passes", "Expected failure cost per 100 passes"),
    ]
    groups: list[tuple[str, pd.DataFrame]] = [("All primary outfield players", primary)]
    for role in ROLE_ORDER:
        selected = primary.loc[primary["analysis_role"].eq(role)]
        if not selected.empty and role != "Goalkeeper":
            groups.append((role, selected))

    rows: list[dict[str, Any]] = []
    outcome = "centered_expected_net_value_per_100_passes"
    for group_label, group in groups:
        for variable, label in variables:
            rows.append(
                {
                    "group": group_label,
                    "n_players": int(len(group)),
                    "variable": variable,
                    "variable_label": label,
                    "outcome": outcome,
                    "spearman_rho": safe_spearman(group, variable, outcome),
                    "interpretation": "descriptive player-level association",
                }
            )
    return pd.DataFrame(rows)


def build_similarity_pairs(
    primary: pd.DataFrame,
    completion_caliper: float,
    xpass_caliper: float,
    volume_caliper: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    matching_variables = [
        "actual_completion_rate",
        "mean_xpass",
        "passes_per_observed_match",
    ]

    for role, group in primary.groupby("analysis_role", sort=True):
        if role == "Goalkeeper" or len(group) < 2:
            continue
        group = group.copy()

        z = pd.DataFrame(index=group.index)
        for variable in matching_variables:
            values = pd.to_numeric(group[variable], errors="coerce")
            standard_deviation = float(values.std(ddof=0))
            if standard_deviation > 0:
                z[variable] = (values - float(values.mean())) / standard_deviation
            else:
                z[variable] = 0.0

        for left_index, right_index in combinations(group.index, 2):
            left = group.loc[left_index]
            right = group.loc[right_index]
            if left["team"] == right["team"]:
                continue

            completion_difference = abs(
                float(left["actual_completion_rate"])
                - float(right["actual_completion_rate"])
            )
            xpass_difference = abs(
                float(left["mean_xpass"]) - float(right["mean_xpass"])
            )
            left_volume = float(left["passes_per_observed_match"])
            right_volume = float(right["passes_per_observed_match"])
            average_volume = 0.5 * (left_volume + right_volume)
            volume_relative_difference = (
                abs(left_volume - right_volume) / average_volume
                if average_volume > 0
                else float("nan")
            )
            standardized_distance = float(
                np.sqrt(
                    np.sum(
                        (
                            z.loc[left_index, matching_variables]
                            - z.loc[right_index, matching_variables]
                        )
                        ** 2
                    )
                )
            )
            env_difference = (
                float(left["centered_expected_net_value_per_100_passes"])
                - float(right["centered_expected_net_value_per_100_passes"])
            )

            qualifies = bool(
                completion_difference <= completion_caliper
                and xpass_difference <= xpass_caliper
                and volume_relative_difference <= volume_caliper
            )

            row: dict[str, Any] = {
                "analysis_role": role,
                "left_team": left["team"],
                "left_player": left["player"],
                "right_team": right["team"],
                "right_player": right["player"],
                "completion_rate_difference": completion_difference,
                "mean_xpass_difference": xpass_difference,
                "passes_per_match_relative_difference": volume_relative_difference,
                "traditional_profile_standardized_distance": standardized_distance,
                "qualifies_primary_calipers": qualifies,
                "centered_env_difference_left_minus_right": env_difference,
                "absolute_centered_env_difference": abs(env_difference),
                "esv_difference_left_minus_right": (
                    float(left["expected_success_value_per_100_passes"])
                    - float(right["expected_success_value_per_100_passes"])
                ),
                "efc_difference_left_minus_right": (
                    float(left["expected_failure_cost_per_100_passes"])
                    - float(right["expected_failure_cost_per_100_passes"])
                ),
                "execution_difference_left_minus_right": (
                    float(left["execution_residual_per_100_passes"])
                    - float(right["execution_residual_per_100_passes"])
                ),
            }
            for prefix, player_row in [("left", left), ("right", right)]:
                for variable in [
                    "n_passes",
                    "n_matches",
                    "passes_per_observed_match",
                    "actual_completion_rate",
                    "mean_xpass",
                    "under_pressure_share",
                    "ground_pass_share",
                    "low_pass_share",
                    "high_pass_share",
                    "expected_success_value_per_100_passes",
                    "expected_failure_cost_per_100_passes",
                    "expected_net_value_per_100_passes",
                    "centered_expected_net_value_per_100_passes",
                    "execution_residual_per_100_passes",
                ]:
                    row[f"{prefix}_{variable}"] = player_row[variable]
            rows.append(row)

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    return pairs.sort_values(
        [
            "qualifies_primary_calipers",
            "absolute_centered_env_difference",
            "traditional_profile_standardized_distance",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def select_pairs(pairs: pd.DataFrame, maximum_pairs: int) -> pd.DataFrame:
    candidates = pairs.loc[pairs["qualifies_primary_calipers"]].copy()
    candidates = candidates.sort_values(
        ["absolute_centered_env_difference", "traditional_profile_standardized_distance"],
        ascending=[False, True],
    )

    selected_rows = []
    used_players: set[tuple[str, str]] = set()
    used_roles: set[str] = set()

    for _, row in candidates.iterrows():
        left_key = (str(row["left_team"]), str(row["left_player"]))
        right_key = (str(row["right_team"]), str(row["right_player"]))
        role = str(row["analysis_role"])
        if role in used_roles:
            continue
        if left_key in used_players or right_key in used_players:
            continue

        selected_rows.append(row.to_dict())
        used_roles.add(role)
        used_players.update([left_key, right_key])
        if len(selected_rows) >= maximum_pairs:
            break

    selected = pd.DataFrame(selected_rows)
    if not selected.empty:
        selected.insert(0, "selected_pair", np.arange(1, len(selected) + 1))
        selected["selection_rule"] = (
            "Cross-team, same stable role, primary calipers met; greedy ordering "
            "by absolute centered-ENV difference, maximum one pair per role, "
            "no repeated player."
        )
    return selected


def selected_pair_profiles(
    selected_pairs: pd.DataFrame,
    primary: pd.DataFrame,
) -> pd.DataFrame:
    if selected_pairs.empty:
        return pd.DataFrame()
    indexed = primary.set_index(["team", "player"], drop=False)
    rows = []
    for _, pair in selected_pairs.iterrows():
        for side in ["left", "right"]:
            key = (str(pair[f"{side}_team"]), str(pair[f"{side}_player"]))
            player = indexed.loc[key]
            rows.append(
                {
                    "selected_pair": int(pair["selected_pair"]),
                    "analysis_role": pair["analysis_role"],
                    "team": player["team"],
                    "player": player["player"],
                    "short_player": short_name(str(player["player"])),
                    "n_matches": int(player["n_matches"]),
                    "n_passes": int(player["n_passes"]),
                    "passes_per_observed_match": float(player["passes_per_observed_match"]),
                    "actual_completion_rate": float(player["actual_completion_rate"]),
                    "mean_xpass": float(player["mean_xpass"]),
                    "under_pressure_share": float(player["under_pressure_share"]),
                    "ground_pass_share": float(player["ground_pass_share"]),
                    "low_pass_share": float(player["low_pass_share"]),
                    "high_pass_share": float(player["high_pass_share"]),
                    "expected_success_value_per_100_passes": float(
                        player["expected_success_value_per_100_passes"]
                    ),
                    "expected_failure_cost_per_100_passes": float(
                        player["expected_failure_cost_per_100_passes"]
                    ),
                    "expected_net_value_per_100_passes": float(
                        player["expected_net_value_per_100_passes"]
                    ),
                    "centered_expected_net_value_per_100_passes": float(
                        player["centered_expected_net_value_per_100_passes"]
                    ),
                    "centered_env_ci_low": float(
                        player["centered_expected_net_value_per_100_passes_ci_low"]
                    ),
                    "centered_env_ci_high": float(
                        player["centered_expected_net_value_per_100_passes_ci_high"]
                    ),
                    "execution_residual_per_100_passes": float(
                        player["execution_residual_per_100_passes"]
                    ),
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Figure and text outputs
# -----------------------------------------------------------------------------

def plot_figure_4(
    primary: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    selected_profiles: pd.DataFrame,
    output_dir: Path,
    full_sample_baseline_per_100: float,
) -> None:
    """Create the journal-ready player application figure.

    Design principles
    -----------------
    * restrained, colorblind-accessible team palette;
    * identical decision-value scale in Panels A and B;
    * uncertainty displayed only for the six highlighted players;
    * one pair label per connector rather than duplicate endpoint labels;
    * raw expected-net-value confidence intervals in Panel C;
    * no redundant role encoding in the scatter panels.
    """
    from matplotlib.patches import Patch

    team_colors = team_color_map()
    colors = {
        "Bayer Leverkusen": team_colors["Bayer Leverkusen 2023/24"],
        "Athletic Club": team_colors["Athletic Club 2015/16"],
        "AS Monaco": team_colors["AS Monaco 2015/16"],
    }

    figure = plt.figure(figsize=(STYLE.double_column_width_in, 6.15))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.18],
        left=0.10,
        right=0.985,
        bottom=0.095,
        top=0.885,
        wspace=0.30,
        hspace=0.52,
    )
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, :])

    selected_keys: set[tuple[str, str]] = set()
    if not selected_profiles.empty:
        selected_keys = set(zip(selected_profiles["team"], selected_profiles["player"]))

    # Shared y limits make the two upper panels directly comparable.
    y_values = primary["centered_expected_net_value_per_100_passes"].to_numpy(float)
    y_low = float(np.nanmin(y_values))
    y_high = float(np.nanmax(y_values))
    y_pad = max(0.22, 0.07 * (y_high - y_low))
    shared_ylim = (y_low - y_pad, y_high + y_pad)

    # Background players are deliberately subdued; selected players carry the story.
    for _, row in primary.iterrows():
        key = (row["team"], row["player"])
        selected = key in selected_keys
        for ax, x_value in (
            (ax_a, 100.0 * float(row["actual_completion_rate"])),
            (ax_b, float(row["execution_residual_per_100_passes"])),
        ):
            ax.scatter(
                x_value,
                float(row["centered_expected_net_value_per_100_passes"]),
                s=48 if selected else 23,
                c=colors[row["team"]],
                marker="o",
                edgecolors=STYLE.dark_slate if selected else "white",
                linewidths=0.85 if selected else 0.45,
                alpha=1.0 if selected else 0.62,
                zorder=5 if selected else 2,
            )

    # Uncertainty is shown for highlighted players only to avoid visual saturation.
    for _, row in selected_profiles.iterrows():
        y = float(row["centered_expected_net_value_per_100_passes"])
        low = float(row["centered_env_ci_low"])
        high = float(row["centered_env_ci_high"])
        yerr = np.array([[max(0.0, y - low)], [max(0.0, high - y)]])
        for ax, x_value in (
            (ax_a, 100.0 * float(row["actual_completion_rate"])),
            (ax_b, float(row["execution_residual_per_100_passes"])),
        ):
            ax.errorbar(
                x_value,
                y,
                yerr=yerr,
                fmt="none",
                ecolor=STYLE.medium_gray,
                elinewidth=0.75,
                capsize=1.8,
                capthick=0.75,
                alpha=0.85,
                zorder=4,
            )

    # Panel A connectors and a single label placed at each pair midpoint.
    midpoint_offsets = {1: (0, 8), 2: (0, -12), 3: (0, 8)}
    for _, pair in selected_pairs.iterrows():
        left = primary.loc[
            primary["team"].eq(pair["left_team"])
            & primary["player"].eq(pair["left_player"])
        ].iloc[0]
        right = primary.loc[
            primary["team"].eq(pair["right_team"])
            & primary["player"].eq(pair["right_player"])
        ].iloc[0]
        x1 = 100.0 * float(left["actual_completion_rate"])
        x2 = 100.0 * float(right["actual_completion_rate"])
        y1 = float(left["centered_expected_net_value_per_100_passes"])
        y2 = float(right["centered_expected_net_value_per_100_passes"])
        ax_a.plot(
            [x1, x2],
            [y1, y2],
            color=STYLE.medium_gray,
            linewidth=1.0,
            linestyle=(0, (3, 2)),
            alpha=0.85,
            zorder=1,
        )
        pair_number = int(pair["selected_pair"])
        ax_a.annotate(
            f"P{pair_number}",
            ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            xytext=midpoint_offsets.get(pair_number, (0, 8)),
            textcoords="offset points",
            fontsize=STYLE.annotation_size,
            fontweight="bold",
            color=STYLE.dark_slate,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
            },
            zorder=7,
        )

    for ax in (ax_a, ax_b):
        ax.axhline(
            0.0,
            color=STYLE.light_gray,
            linewidth=STYLE.reference_line_width,
            linestyle=(0, (4, 3)),
            zorder=0,
        )
        ax.set_ylim(*shared_ylim)
        clean_axis(ax, grid_axis="both")

    ax_a.set_xlabel("Pass completion (%)")
    ax_a.set_ylabel("Centered expected net value\nper 100 passes")
    ax_a.set_title("Similar completion, different decision value", loc="left", pad=5)
    add_panel_label(ax_a, "A", x=-0.15)

    ax_b.axvline(
        0.0,
        color=STYLE.light_gray,
        linewidth=STYLE.reference_line_width,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax_b.set_xlabel("Execution residual per 100 passes")
    ax_b.set_ylabel("Centered expected net value\nper 100 passes")
    ax_b.set_title("Decision value and execution are distinct", loc="left", pad=5)
    add_panel_label(ax_b, "B", x=-0.15)

    # The zero reference lines provide the decision–execution quadrant structure.

    # Compact shared team legend.
    team_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5.2,
            markerfacecolor=colors[team],
            markeredgecolor="white",
            label=TEAM_DISPLAY[team],
        )
        for team in TEAM_ORDER
    ]
    figure.legend(
        handles=team_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        handletextpad=0.35,
        columnspacing=1.20,
        borderaxespad=0.0,
    )

    # Panel C: successful-state gain, failure exposure, and net value with CIs.
    plot_rows = selected_profiles.copy()
    team_short = {
        "AS Monaco": "Monaco",
        "Athletic Club": "Athletic",
        "Bayer Leverkusen": "Leverkusen",
    }
    plot_rows["display_label"] = (
        "P"
        + plot_rows["selected_pair"].astype(int).astype(str)
        + "  "
        + plot_rows["short_player"]
        + " · "
        + plot_rows["team"].map(team_short)
    )
    plot_rows = plot_rows.sort_values(["selected_pair", "team"], ascending=[True, True])

    y_positions = np.arange(len(plot_rows), dtype=float)
    for pair_number in sorted(plot_rows["selected_pair"].unique()):
        mask = plot_rows["selected_pair"].eq(pair_number).to_numpy()
        y_positions[mask] += 0.48 * (int(pair_number) - 1)
    plot_rows["y_position"] = y_positions

    # Alternating group bands and separators improve pair-wise reading without clutter.
    for pair_number, group in plot_rows.groupby("selected_pair", sort=True):
        y_min = float(group["y_position"].min()) - 0.42
        y_max = float(group["y_position"].max()) + 0.42
        if int(pair_number) % 2 == 1:
            ax_c.axhspan(y_min, y_max, color=STYLE.near_white, zorder=0)
        if int(pair_number) > 1:
            ax_c.axhline(y_min - 0.18, color=STYLE.grid_gray, linewidth=0.55, zorder=0)

    ax_c.barh(
        plot_rows["y_position"],
        plot_rows["expected_success_value_per_100_passes"],
        height=0.52,
        color=STYLE.deep_blue,
        alpha=0.90,
        zorder=2,
    )
    ax_c.barh(
        plot_rows["y_position"],
        -plot_rows["expected_failure_cost_per_100_passes"],
        height=0.52,
        color=STYLE.muted_orange,
        alpha=0.88,
        zorder=2,
    )

    raw_env = plot_rows["expected_net_value_per_100_passes"].to_numpy(float)
    raw_ci_low = (
        plot_rows["centered_env_ci_low"].to_numpy(float)
        + full_sample_baseline_per_100
    )
    raw_ci_high = (
        plot_rows["centered_env_ci_high"].to_numpy(float)
        + full_sample_baseline_per_100
    )
    xerr = np.vstack(
        [
            np.maximum(0.0, raw_env - raw_ci_low),
            np.maximum(0.0, raw_ci_high - raw_env),
        ]
    )
    ax_c.errorbar(
        raw_env,
        plot_rows["y_position"],
        xerr=xerr,
        fmt="D",
        markersize=4.6,
        markerfacecolor=STYLE.dark_slate,
        markeredgecolor="white",
        markeredgewidth=0.65,
        ecolor=STYLE.dark_slate,
        elinewidth=0.85,
        capsize=2.0,
        capthick=0.85,
        zorder=5,
    )

    ax_c.axvline(0.0, color=STYLE.dark_slate, linewidth=0.72, zorder=1)
    ax_c.axvline(
        full_sample_baseline_per_100,
        color=STYLE.light_gray,
        linewidth=STYLE.reference_line_width,
        linestyle=(0, (4, 3)),
        zorder=1,
    )
    ax_c.set_yticks(plot_rows["y_position"])
    ax_c.set_yticklabels(plot_rows["display_label"])
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Value per 100 passes")
    ax_c.set_title("Matched-pair value decomposition", loc="left", pad=5)
    clean_axis(ax_c, grid_axis="x")
    ax_c.spines["left"].set_visible(False)
    ax_c.tick_params(axis="y", length=0)

    component_handles = [
        Patch(facecolor=STYLE.deep_blue, edgecolor="none", label="Successful-state gain"),
        Patch(facecolor=STYLE.muted_orange, edgecolor="none", label="Failure exposure"),
        Line2D(
            [0],
            [0],
            marker="D",
            color=STYLE.dark_slate,
            linewidth=0.85,
            markersize=4.5,
            markerfacecolor=STYLE.dark_slate,
            markeredgecolor="white",
            label="Expected net value (95% CI)",
        ),
        Line2D(
            [0],
            [0],
            color=STYLE.light_gray,
            linewidth=STYLE.reference_line_width,
            linestyle=(0, (4, 3)),
            label="Full-sample ENV reference",
        ),
    ]
    ax_c.legend(
        handles=component_handles,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.005),
        ncol=2,
        handletextpad=0.45,
        columnspacing=0.95,
        borderaxespad=0.0,
    )
    add_panel_label(ax_c, "C", x=-0.073)

    save_figure_all_formats(figure, output_dir, "Figure_4_player_decision_value")
    plt.close(figure)

def build_results_text(
    profiles: pd.DataFrame,
    correlations: pd.DataFrame,
    pairs: pd.DataFrame,
    selected_profiles: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    primary = profiles.loc[profiles["primary_outfield_role_stable"]]

    def overall_rho(variable: str) -> float:
        return float(
            correlations.loc[
                correlations["group"].eq("All primary outfield players")
                & correlations["variable"].eq(variable),
                "spearman_rho",
            ].iloc[0]
        )

    paragraphs = [
        "PLAYER-LEVEL DECISION VALUE APPLICATION",
        "",
        (
            f"Of the {int(profiles['primary_eligible'].sum())} players meeting the "
            f"locked eligibility criteria of at least {args.min_passes} passes and "
            f"{args.min_matches} matches, {len(primary)} outfield players also had "
            f"a stable role assignment, defined as at least "
            f"{100 * args.minimum_role_share:.0f}% of eligible passes from one broad "
            "StatsBomb position group."
        ),
        "",
        (
            "Traditional passing efficiency was related to, but did not fully "
            "determine, decision value. Across the primary outfield sample, the "
            f"Spearman correlation between completion rate and centered expected "
            f"net value was {overall_rho('actual_completion_rate'):.3f}; the "
            f"corresponding correlation for mean xPass was "
            f"{overall_rho('mean_xpass'):.3f}. By comparison, the association "
            "between execution residual and centered expected net value was weaker "
            f"(rho={overall_rho('execution_residual_per_100_passes'):.3f}), "
            "indicating that selecting higher-value passes and completing passes "
            "above model expectation represented related but distinct dimensions."
        ),
        "",
        (
            f"Cross-team player pairs were then identified within the same stable "
            f"role. A pair was considered traditionally similar when completion "
            f"rates differed by no more than {100 * args.completion_caliper:.1f} "
            f"percentage points, mean xPass differed by no more than "
            f"{args.xpass_caliper:.3f}, and passes per observed match differed by "
            f"no more than {100 * args.volume_relative_caliper:.0f}% in relative "
            f"terms. {int(pairs['qualifies_primary_calipers'].sum())} cross-team "
            "pairs met all three conditions. The displayed pairs were selected "
            "algorithmically from this eligible set by descending absolute "
            "difference in centered expected net value, with no repeated player "
            "and at most one pair from each role group."
        ),
    ]

    if not selected_profiles.empty:
        paragraphs.extend(["", "ILLUSTRATIVE MATCHED-PLAYER CONTRASTS", ""])
        for pair_number, group in selected_profiles.groupby("selected_pair", sort=True):
            first = group.iloc[0]
            second = group.iloc[1]
            env_difference = abs(
                first["centered_expected_net_value_per_100_passes"]
                - second["centered_expected_net_value_per_100_passes"]
            )
            higher = (
                first
                if first["centered_expected_net_value_per_100_passes"]
                >= second["centered_expected_net_value_per_100_passes"]
                else second
            )
            lower = second if higher is first else first
            esv_difference = (
                higher["expected_success_value_per_100_passes"]
                - lower["expected_success_value_per_100_passes"]
            )
            efc_difference = (
                higher["expected_failure_cost_per_100_passes"]
                - lower["expected_failure_cost_per_100_passes"]
            )
            paragraphs.append(
                (
                    f"Pair {pair_number} ({first['analysis_role']}): "
                    f"{first['short_player']} and {second['short_player']} had "
                    f"completion rates of {100 * first['actual_completion_rate']:.1f}% "
                    f"and {100 * second['actual_completion_rate']:.1f}% and mean "
                    f"xPass values of {first['mean_xpass']:.3f} and "
                    f"{second['mean_xpass']:.3f}, respectively. Their centered "
                    f"expected net values nevertheless differed by {env_difference:.3f} "
                    f"per 100 passes. {higher['short_player']} combined "
                    f"{abs(esv_difference):.3f} more expected successful-state value "
                    f"with {abs(efc_difference):.3f} "
                    f"{'less' if efc_difference < 0 else 'more'} expected failure "
                    "exposure per 100 passes than the matched player. The contrast "
                    "therefore reflected value composition rather than completion "
                    "rate alone."
                )
            )

    paragraphs.extend(
        [
            "",
            "INTERPRETATION BOUNDARY",
            "",
            (
                "These comparisons describe short-horizon attacking decision value, "
                "not total tactical contribution or overall player quality. The "
                "matched pairs were selected after applying transparent similarity "
                "rules and are presented as illustrations rather than confirmatory "
                "hypothesis tests. Values may also depend on the H=3 threat horizon, "
                "the event-data representation of pressure, and the counterfactual "
                "failure-exposure definition."
            ),
        ]
    )
    return "\n".join(paragraphs) + "\n"


def build_caption(args: argparse.Namespace) -> str:
    return (
        "Figure 4. Traditional passing efficiency, decision value, and execution "
        "within the three illustrative team-season cohorts. (A) Pass completion "
        "versus centered expected net value per 100 passes. Dashed connectors "
        "identify algorithmically selected cross-team pairs from the same stable "
        "role whose completion rates, mean xPass values, and passing volumes met "
        "the prespecified similarity calipers. Error bars are match-cluster "
        "bootstrap 95% confidence intervals for the six highlighted players. "
        "(B) Centered expected net value versus execution residual, distinguishing "
        "pass-selection value from execution relative to xPass expectation. "
        "Horizontal and vertical dashed lines denote the full-sample decision-value "
        "reference and zero execution residual, respectively. (C) Expected "
        "successful-state gain, expected failure exposure, and raw expected net "
        "value for the selected pairs. Failure exposure is displayed on the "
        "negative side for decomposition; diamonds and horizontal error bars show "
        "expected net value and its match-cluster bootstrap 95% confidence interval. "
        "The dashed vertical line marks the full-sample expected-net-value reference. "
        "Colors identify teams in Panels A and B; role similarity was enforced in "
        "the matching procedure rather than encoded as an additional visual channel. "
        "Eligible players completed at least "
        f"{args.min_passes} passes in at least {args.min_matches} matches, and stable "
        f"roles required at least {100 * args.minimum_role_share:.0f}% of passes from "
        "one broad role group. The analysis is descriptive and does not rank overall "
        "player quality.\n"
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    event_path = args.player_event_csv.resolve()
    summary_path = args.player_summary_csv.resolve()
    output_dir = args.output_dir.resolve()

    for path in [event_path, summary_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    if not 0.0 < args.minimum_role_share <= 1.0:
        raise ValueError("--minimum_role_share must be in (0, 1].")
    if args.maximum_selected_pairs < 1:
        raise ValueError("--maximum_selected_pairs must be positive.")

    prepare_output_dir(output_dir, args.overwrite)
    configure_matplotlib()

    events = pd.read_csv(event_path, encoding="utf-8-sig", low_memory=False)
    summary = pd.read_csv(summary_path, encoding="utf-8-sig", low_memory=False)

    missing_events = sorted(REQUIRED_EVENT_COLUMNS - set(events.columns))
    missing_summary = sorted(REQUIRED_SUMMARY_COLUMNS - set(summary.columns))
    if missing_events:
        raise RuntimeError(
            "Player-event input is missing columns: " + ", ".join(missing_events)
        )
    if missing_summary:
        raise RuntimeError(
            "Player-summary input is missing columns: " + ", ".join(missing_summary)
        )

    events["event_id"] = events["event_id"].map(normalize_id)
    events["match_id"] = events["match_id"].map(normalize_id)
    if events.duplicated(["event_id", "match_id"]).any():
        raise RuntimeError("Duplicate event-match keys in player-event input.")

    roles = assign_roles(events, args.minimum_role_share)
    context = build_context_profiles(events)
    profiles = build_player_profiles(
        summary,
        context,
        roles,
        args.min_passes,
        args.min_matches,
    )
    primary = profiles.loc[profiles["primary_outfield_role_stable"]].copy()

    correlations = build_correlations(primary)
    rank_discordance = primary[
        [
            "team",
            "player",
            "analysis_role",
            "n_matches",
            "n_passes",
            "actual_completion_rate",
            "mean_xpass",
            "passes_per_observed_match",
            "centered_expected_net_value_per_100_passes",
            "execution_residual_per_100_passes",
            "completion_rank_within_role",
            "decision_value_rank_within_role",
            "completion_minus_decision_rank",
        ]
    ].sort_values(
        ["analysis_role", "completion_minus_decision_rank"],
        ascending=[True, False],
    )

    pairs = build_similarity_pairs(
        primary,
        args.completion_caliper,
        args.xpass_caliper,
        args.volume_relative_caliper,
    )
    selected_pairs = select_pairs(pairs, args.maximum_selected_pairs)
    selected_profiles = selected_pair_profiles(selected_pairs, primary)

    identity_audit_columns = [
        column for column in profiles.columns if column.startswith("audit_difference__")
    ]
    max_identity_difference = float(
        np.nanmax(profiles[identity_audit_columns].to_numpy(dtype=float))
    )

    checks = {
        "input_rows_equal_54795": len(events) == 54_795,
        "input_matches_equal_110": events["match_id"].nunique() == 110,
        "input_players_equal_80": events["player"].nunique() == 80,
        "no_duplicate_event_match_keys": not events.duplicated(
            ["event_id", "match_id"]
        ).any(),
        "all_positions_mapped": set(events["position"]).issubset(POSITION_TO_ROLE),
        "summary_players_equal_event_players": len(profiles) == 80,
        "eligible_players_equal_55": int(profiles["primary_eligible"].sum()) == 55,
        "primary_outfield_role_stable_players_positive": len(primary) > 0,
        "event_summary_metrics_identical": max_identity_difference < 1e-10,
        "at_least_one_similarity_pair": bool(
            len(pairs) > 0 and pairs["qualifies_primary_calipers"].any()
        ),
        "selected_pairs_created": len(selected_pairs) > 0,
        "selected_players_unique": (
            selected_profiles[["team", "player"]].drop_duplicates().shape[0]
            == len(selected_profiles)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Module 07 audit failed:\n"
            + json.dumps(checks, ensure_ascii=False, indent=2, default=json_default)
        )

    outputs = {
        "roles": output_dir / "player_role_assignments.csv",
        "profiles": output_dir / "player_risk_reward_profiles.csv",
        "correlations": output_dir / "player_profile_correlations.csv",
        "rank_discordance": output_dir / "player_rank_discordance.csv",
        "pairs": output_dir / "traditional_similarity_pairs.csv",
        "selected_pairs": output_dir / "selected_player_pairs.csv",
        "selected_profiles": output_dir / "selected_player_pair_profiles.csv",
        "results_text": output_dir / "module07_results_text.txt",
        "caption": output_dir / "module07_figure_caption.txt",
        "audit": output_dir / "module07_audit.json",
    }

    roles.to_csv(outputs["roles"], index=False, encoding="utf-8-sig")
    profiles.to_csv(outputs["profiles"], index=False, encoding="utf-8-sig")
    correlations.to_csv(outputs["correlations"], index=False, encoding="utf-8-sig")
    rank_discordance.to_csv(
        outputs["rank_discordance"], index=False, encoding="utf-8-sig"
    )
    pairs.to_csv(outputs["pairs"], index=False, encoding="utf-8-sig")
    selected_pairs.to_csv(
        outputs["selected_pairs"], index=False, encoding="utf-8-sig"
    )
    selected_profiles.to_csv(
        outputs["selected_profiles"], index=False, encoding="utf-8-sig"
    )

    plot_figure_4(
        primary,
        selected_pairs,
        selected_profiles,
        output_dir,
        args.full_sample_baseline_per_100,
    )
    outputs["results_text"].write_text(
        build_results_text(profiles, correlations, pairs, selected_profiles, args),
        encoding="utf-8",
    )
    outputs["caption"].write_text(build_caption(args), encoding="utf-8")

    role_counts = (
        profiles.loc[profiles["primary_eligible"]]
        .groupby("analysis_role", sort=True)
        .size()
        .to_dict()
    )
    primary_role_counts = primary.groupby("analysis_role", sort=True).size().to_dict()
    overall_correlations = correlations.loc[
        correlations["group"].eq("All primary outfield players")
    ][["variable", "spearman_rho"]]

    audit = {
        "status": "PASS",
        "script_version": SCRIPT_VERSION,
        "purpose": (
            "Player-level descriptive application testing whether traditional "
            "passing efficiency can conceal differences in short-horizon passing "
            "decision value."
        ),
        "inputs": {
            "player_event_csv": str(event_path),
            "player_summary_csv": str(summary_path),
            "player_event_sha256": sha256_file(event_path),
            "player_summary_sha256": sha256_file(summary_path),
        },
        "locked_references": {
            "full_sample_rows": args.full_sample_rows,
            "full_sample_env_per_100": args.full_sample_baseline_per_100,
            "primary_value_model": "H=3 PCTV, main OOF xPass",
        },
        "eligibility": {
            "minimum_passes": args.min_passes,
            "minimum_matches": args.min_matches,
            "minimum_dominant_role_share": args.minimum_role_share,
            "eligible_players": int(profiles["primary_eligible"].sum()),
            "eligible_role_counts_including_mixed_and_goalkeepers": {
                str(key): int(value) for key, value in role_counts.items()
            },
            "primary_outfield_stable_role_players": int(len(primary)),
            "primary_outfield_role_counts": {
                str(key): int(value) for key, value in primary_role_counts.items()
            },
        },
        "matching": {
            "same_stable_role": True,
            "cross_team_only": True,
            "completion_rate_caliper": args.completion_caliper,
            "mean_xpass_caliper": args.xpass_caliper,
            "passes_per_match_relative_caliper": args.volume_relative_caliper,
            "qualifying_pairs": int(pairs["qualifies_primary_calipers"].sum()),
            "selected_pairs": int(len(selected_pairs)),
            "maximum_one_pair_per_role": True,
            "no_repeated_player": True,
            "selection_basis": (
                "Descending absolute centered expected-net-value difference among "
                "pairs meeting all traditional-profile calipers."
            ),
            "interpretation": "illustrative, not confirmatory inference",
        },
        "overall_spearman_correlations_with_centered_env": {
            str(row["variable"]): float(row["spearman_rho"])
            for _, row in overall_correlations.iterrows()
        },
        "numerical_audit": {
            "maximum_event_summary_metric_difference": max_identity_difference,
        },
        "checks": checks,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "outputs": [
            path.name for path in sorted(output_dir.iterdir()) if path.is_file()
        ]
        + ["module07_audit.json", "SHA256SUMS.txt"],
    }
    outputs["audit"].write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_sha256sums(output_dir)

    print("=" * 96)
    print("Module 07 player decision-value analysis completed")
    print("=" * 96)
    print(f"Input rows: {len(events):,}")
    print(f"Eligible players: {int(profiles['primary_eligible'].sum()):,}")
    print(f"Primary stable-role outfield players: {len(primary):,}")
    print(f"Qualifying traditional-similarity pairs: {int(pairs['qualifies_primary_calipers'].sum()):,}")
    print(f"Selected illustrative pairs: {len(selected_pairs):,}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
