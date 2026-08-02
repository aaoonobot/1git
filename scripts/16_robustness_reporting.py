#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
16_robustness_reporting.py

Reporting layer for the completed Module 04 robustness
analysis. This script does not refit xPass or PCTV. It converts the locked
robustness outputs into:

1. a transparent specification-level interpretation table;
2. matched-player contrast stability estimates;
3. a publication-ready supplementary robustness figure;
4. revised Results text and an audit JSON that distinguishes technical
   completion from substantive robustness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

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
    )
except ImportError as exc:
    raise ImportError(
        "publication_style.py must be saved in the same folder as "
        "16_robustness_reporting.py."
    ) from exc


SCRIPT_VERSION = "2026-07-29-module04-reporting-final-v2"
REFERENCE = "H3_main_primary_full"

SPEC_ORDER = [
    "H3_main_primary_full",
    "H1_main_primary_full",
    "H5_main_primary_full",
    "H3_no_team_primary_full",
    "H3_main_strict_full",
    "H3_main_primary_start_only",
    "H3_main_primary_opponent_only",
]

SPEC_LABEL = {
    "H3_main_primary_full": "Primary H=3",
    "H1_main_primary_full": "Short horizon (H=1)",
    "H5_main_primary_full": "Long horizon (H=5)",
    "H3_no_team_primary_full": "No-team xPass",
    "H3_main_strict_full": "Strict open play",
    "H3_main_primary_start_only": "Start-state only",
    "H3_main_primary_opponent_only": "Opponent-transition only",
}

PAIR_MARKERS = {1: "o", 2: "s", 3: "D"}
PAIR_COLORS = {1: STYLE.deep_blue, 2: STYLE.muted_orange, 3: STYLE.muted_teal}



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
        description="Create final reporting outputs from Module 04 robustness files."
    )
    parser.add_argument("--robustness_summary_json", required=True, type=Path)
    parser.add_argument("--robustness_stability_csv", required=True, type=Path)
    parser.add_argument("--robustness_player_results_csv", required=True, type=Path)
    parser.add_argument("--robustness_team_results_csv", required=True, type=Path)
    parser.add_argument("--selected_player_pairs_csv", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--player_threshold", type=int, default=200)
    parser.add_argument("--minimum_matches", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "1.0", "yes", "y", "t"])
    )


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = list(path.iterdir())
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use --overwrite to replace outputs."
        )
    if overwrite:
        for item in existing:
            if item.is_file():
                item.unlink()


def write_sha256sums(output_dir: Path) -> None:
    lines = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            lines.append(f"{sha256_file(path)}  {path.name}")
    (output_dir / "SHA256SUMS.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def classify_specification(row: pd.Series) -> tuple[str, str]:
    if row["comparison_specification"] == REFERENCE:
        return "Reference", "Primary specification"

    decision_stable = float(row["decision_spearman"]) >= 0.80
    execution_stable = float(row["execution_spearman"]) >= 0.80
    quadrant_stable = float(row["quadrant_agreement"]) >= 0.70
    team_order_stable = bool(row["team_order_identical"])

    if decision_stable and execution_stable and quadrant_stable and team_order_stable:
        return "Stable", "Ranks and profile classifications remained stable"
    if decision_stable and execution_stable and team_order_stable:
        return (
            "Boundary-sensitive",
            "Continuous ranks remained stable, but profile classifications were sensitive",
        )
    if not decision_stable:
        return (
            "Decision-sensitive",
            "Player decision-value ordering changed materially",
        )
    return "Mixed", "At least one prespecified stability criterion was not met"


def build_interpretation_table(stability: pd.DataFrame, threshold: int) -> pd.DataFrame:
    table = stability.loc[
        stability["player_threshold"].eq(threshold)
        & stability["comparison_specification"].isin(SPEC_ORDER)
    ].copy()
    table["comparison_specification"] = pd.Categorical(
        table["comparison_specification"], SPEC_ORDER, ordered=True
    )
    table = table.sort_values("comparison_specification").reset_index(drop=True)
    table["specification_label"] = table["comparison_specification"].map(SPEC_LABEL)
    classifications = table.apply(classify_specification, axis=1, result_type="expand")
    table["robustness_class"] = classifications[0]
    table["interpretation"] = classifications[1]
    return table[
        [
            "reference_specification",
            "comparison_specification",
            "specification_label",
            "player_threshold",
            "common_eligible_players",
            "decision_spearman",
            "execution_spearman",
            "quadrant_agreement",
            "decision_top_k_overlap",
            "execution_top_k_overlap",
            "team_order_identical",
            "reference_team_order",
            "comparison_team_order",
            "robustness_class",
            "interpretation",
        ]
    ]


def build_pair_robustness(
    player_results: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    threshold: int,
) -> pd.DataFrame:
    data = player_results.loc[
        player_results["player_threshold"].eq(threshold)
        & player_results["specification"].isin(SPEC_ORDER)
    ].copy()
    data["eligible"] = normalize_bool(data["eligible"])

    rows: list[dict[str, object]] = []
    for _, pair in selected_pairs.sort_values("selected_pair").iterrows():
        pair_number = int(pair["selected_pair"])
        left_key = (str(pair["left_team"]), str(pair["left_player"]))
        right_key = (str(pair["right_team"]), str(pair["right_player"]))

        primary = data.loc[data["specification"].eq(REFERENCE)]
        left_primary = primary.loc[
            primary["team"].eq(left_key[0]) & primary["player"].eq(left_key[1])
        ]
        right_primary = primary.loc[
            primary["team"].eq(right_key[0]) & primary["player"].eq(right_key[1])
        ]
        if left_primary.empty or right_primary.empty:
            raise RuntimeError(f"Primary pair rows missing for P{pair_number}.")

        left_value = float(left_primary.iloc[0]["expected_net_value_per_100_passes"])
        right_value = float(right_primary.iloc[0]["expected_net_value_per_100_passes"])
        if left_value >= right_value:
            higher_team, higher_player = left_key
            lower_team, lower_player = right_key
        else:
            higher_team, higher_player = right_key
            lower_team, lower_player = left_key

        for specification in SPEC_ORDER:
            spec = data.loc[data["specification"].eq(specification)]
            high = spec.loc[
                spec["team"].eq(higher_team) & spec["player"].eq(higher_player)
            ]
            low = spec.loc[
                spec["team"].eq(lower_team) & spec["player"].eq(lower_player)
            ]

            estimable = (
                not high.empty
                and not low.empty
                and bool(high.iloc[0]["eligible"])
                and bool(low.iloc[0]["eligible"])
            )
            if estimable:
                difference = float(
                    high.iloc[0]["expected_net_value_per_100_passes"]
                    - low.iloc[0]["expected_net_value_per_100_passes"]
                )
                high_passes = int(high.iloc[0]["n_passes"])
                low_passes = int(low.iloc[0]["n_passes"])
            else:
                difference = np.nan
                high_passes = int(high.iloc[0]["n_passes"]) if not high.empty else 0
                low_passes = int(low.iloc[0]["n_passes"]) if not low.empty else 0

            rows.append(
                {
                    "selected_pair": pair_number,
                    "analysis_role": pair["analysis_role"],
                    "specification": specification,
                    "specification_label": SPEC_LABEL[specification],
                    "higher_value_team_primary": higher_team,
                    "higher_value_player_primary": higher_player,
                    "comparison_team": lower_team,
                    "comparison_player": lower_player,
                    "higher_player_passes": high_passes,
                    "comparison_player_passes": low_passes,
                    "estimable": estimable,
                    "oriented_env_difference_per_100": difference,
                    "direction_preserved": bool(difference > 0) if estimable else pd.NA,
                }
            )
    return pd.DataFrame(rows)


def create_robustness_figure(
    interpretation: pd.DataFrame,
    pair_robustness: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure = plt.figure(figsize=(STYLE.double_column_width_in, 5.35))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.15],
        left=0.12,
        right=0.985,
        bottom=0.11,
        top=0.90,
        wspace=0.35,
        hspace=0.58,
    )
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, :])

    alternatives = interpretation.loc[
        ~interpretation["comparison_specification"].eq(REFERENCE)
    ].copy()
    alternatives = alternatives.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(alternatives), dtype=float)

    def status_color(status: str) -> str:
        return {
            "Stable": STYLE.deep_blue,
            "Boundary-sensitive": STYLE.muted_orange,
            "Decision-sensitive": STYLE.muted_red,
            "Mixed": STYLE.medium_gray,
        }.get(status, STYLE.medium_gray)

    colors = [status_color(value) for value in alternatives["robustness_class"]]

    ax_a.hlines(
        y,
        0.0,
        alternatives["decision_spearman"],
        color=STYLE.grid_gray,
        linewidth=0.9,
        zorder=1,
    )
    ax_a.scatter(
        alternatives["decision_spearman"],
        y,
        c=colors,
        s=31,
        edgecolors="white",
        linewidths=0.55,
        zorder=3,
    )
    ax_a.axvline(
        0.80,
        color=STYLE.light_gray,
        linewidth=STYLE.reference_line_width,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax_a.set_xlim(0.0, 1.02)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(alternatives["specification_label"])
    ax_a.set_xlabel("Spearman rank correlation")
    ax_a.set_title("Decision-value rank stability", loc="left", pad=5)
    clean_axis(ax_a, grid_axis="x")
    add_panel_label(ax_a, "A", x=-0.22)

    # Use the same specification-level robustness classes in Panels A and B
    # so that color has one meaning throughout the figure.
    q_colors = colors
    ax_b.hlines(
        y,
        0.0,
        alternatives["quadrant_agreement"],
        color=STYLE.grid_gray,
        linewidth=0.9,
        zorder=1,
    )
    ax_b.scatter(
        alternatives["quadrant_agreement"],
        y,
        c=q_colors,
        s=31,
        edgecolors="white",
        linewidths=0.55,
        zorder=3,
    )
    ax_b.axvline(
        0.70,
        color=STYLE.light_gray,
        linewidth=STYLE.reference_line_width,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax_b.set_xlim(0.0, 1.02)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([])
    ax_b.set_xlabel("Profile-classification agreement")
    ax_b.set_title("Decision–execution profile stability", loc="left", pad=5)
    clean_axis(ax_b, grid_axis="x")
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(axis="y", length=0)
    add_panel_label(ax_b, "B", x=-0.12)

    # Panel C includes the primary specification and all alternatives.
    spec_y = {spec: index for index, spec in enumerate(SPEC_ORDER[::-1])}
    offsets = {1: -0.18, 2: 0.0, 3: 0.18}
    for pair_number in sorted(pair_robustness["selected_pair"].unique()):
        group = pair_robustness.loc[
            pair_robustness["selected_pair"].eq(pair_number)
        ].copy()
        estimable = group.loc[group["estimable"]].copy()
        y_values = np.array(
            [spec_y[spec] + offsets[pair_number] for spec in estimable["specification"]]
        )
        x_values = estimable["oriented_env_difference_per_100"].to_numpy(float)
        ax_c.scatter(
            x_values,
            y_values,
            marker=PAIR_MARKERS[pair_number],
            s=29,
            c=PAIR_COLORS[pair_number],
            edgecolors="white",
            linewidths=0.55,
            zorder=4,
        )

    ax_c.axvline(0.0, color=STYLE.dark_slate, linewidth=0.75, zorder=1)
    ax_c.set_yticks(range(len(SPEC_ORDER)))
    ax_c.set_yticklabels([SPEC_LABEL[spec] for spec in SPEC_ORDER[::-1]])
    ax_c.set_xlabel("Oriented expected-net-value difference per 100 passes")
    ax_c.set_title("Matched-player contrast stability", loc="left", pad=5)
    clean_axis(ax_c, grid_axis="x")
    add_panel_label(ax_c, "C", x=-0.085)

    pair_handles = []
    for _, pair in selected_pairs.sort_values("selected_pair").iterrows():
        number = int(pair["selected_pair"])
        left_short = str(pair["left_player"]).split()[-1]
        right_short = str(pair["right_player"]).split()[-1]
        # Use manuscript-friendly short labels for the known selected pairs.
        label = f"P{number}"
        pair_handles.append(
            Line2D(
                [0],
                [0],
                marker=PAIR_MARKERS[number],
                linestyle="-",
                linewidth=0.85,
                markersize=4.6,
                color=PAIR_COLORS[number],
                markerfacecolor=PAIR_COLORS[number],
                markeredgecolor="white",
                label=label,
            )
        )
    ax_c.legend(
        handles=pair_handles,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.10),
        ncol=3,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )

    status_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=5.2,
            markerfacecolor=STYLE.deep_blue,
            markeredgecolor="white",
            markeredgewidth=0.55,
            label="Stable",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=5.2,
            markerfacecolor=STYLE.muted_orange,
            markeredgecolor="white",
            markeredgewidth=0.55,
            label="Boundary-sensitive",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=5.2,
            markerfacecolor=STYLE.muted_red,
            markeredgecolor="white",
            markeredgewidth=0.55,
            label="Decision-sensitive",
        ),
    ]
    figure.legend(
        handles=status_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.60, 0.985),
        ncol=3,
        handletextpad=0.35,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    save_figure_all_formats(figure, output_dir, "Figure_S4_player_robustness")
    plt.close(figure)


def build_results_text(
    interpretation: pd.DataFrame,
    pair_robustness: pd.DataFrame,
    threshold: int,
) -> str:
    alt = interpretation.loc[
        ~interpretation["comparison_specification"].eq(REFERENCE)
    ].set_index("comparison_specification")

    stable_specs = [
        "H1_main_primary_full",
        "H5_main_primary_full",
        "H3_no_team_primary_full",
        "H3_main_strict_full",
        "H3_main_primary_start_only",
    ]
    decision_min = float(alt.loc[stable_specs, "decision_spearman"].min())
    decision_max = float(alt.loc[stable_specs, "decision_spearman"].max())
    execution_min = float(alt.loc[stable_specs, "execution_spearman"].min())
    execution_max = float(alt.loc[stable_specs, "execution_spearman"].max())
    strict_agreement = float(
        alt.loc["H3_main_strict_full", "quadrant_agreement"]
    )
    opponent_rho = float(
        alt.loc["H3_main_primary_opponent_only", "decision_spearman"]
    )
    opponent_quadrant = float(
        alt.loc["H3_main_primary_opponent_only", "quadrant_agreement"]
    )

    lines = [
        "ROBUSTNESS OF PLAYER-LEVEL DECISION-VALUE PROFILES",
        "",
        (
            f"At the locked eligibility threshold of at least {threshold} passes "
            "and five matches, player decision-value rankings remained highly "
            "consistent under the H=1 and H=5 horizons, the No-team xPass "
            "specification, the strict open-play sample, and the start-state-only "
            f"failure-cost definition (Spearman rho range, {decision_min:.3f} to "
            f"{decision_max:.3f}). Execution rankings were similarly stable "
            f"(rho range, {execution_min:.3f} to {execution_max:.3f}). The strict "
            f"open-play sample retained stable continuous rankings but showed lower "
            f"decision–execution profile agreement ({strict_agreement:.3f}), "
            "consistent with players close to the classification boundaries moving "
            "between descriptive quadrants after sample restriction."
        ),
        "",
        (
            "Stability was substantially lower when failure exposure was restricted "
            "to opponent transition threat alone. Under this specification, the "
            f"decision-value rank correlation declined to {opponent_rho:.3f} and "
            f"profile agreement to {opponent_quadrant:.3f}. The result indicates "
            "that the foregone value of the team’s existing attacking state was an "
            "influential component of the complete failure-exposure formulation. "
            "The three-team ordering remained unchanged across all specifications."
        ),
        "",
        "ROBUSTNESS OF THE MATCHED-PLAYER CONTRASTS",
        "",
    ]

    for pair_number, group in pair_robustness.groupby("selected_pair", sort=True):
        available = group.loc[group["estimable"]].copy()
        minimum = float(available["oriented_env_difference_per_100"].min())
        maximum = float(available["oriented_env_difference_per_100"].max())
        higher = str(available.iloc[0]["higher_value_player_primary"])
        lower = str(available.iloc[0]["comparison_player"])
        missing_rows = group.loc[~group["estimable"]].copy()
        sentence = (
            f"Pair {int(pair_number)}: the expected-net-value advantage of {higher} "
            f"over {lower} ranged from {minimum:.3f} to {maximum:.3f} units per 100 "
            "passes across the estimable specifications, and its direction was "
            "preserved throughout."
        )
        if not missing_rows.empty:
            missing_clauses = []
            for _, missing_row in missing_rows.iterrows():
                below_threshold = []
                if int(missing_row["higher_player_passes"]) < threshold:
                    below_threshold.append(str(missing_row["higher_value_player_primary"]))
                if int(missing_row["comparison_player_passes"]) < threshold:
                    below_threshold.append(str(missing_row["comparison_player"]))

                if len(below_threshold) == 1:
                    reason = (
                        f"{below_threshold[0]} did not meet the locked eligibility "
                        f"threshold of at least {threshold} passes"
                    )
                elif len(below_threshold) == 2:
                    reason = (
                        f"{below_threshold[0]} and {below_threshold[1]} did not meet "
                        f"the locked eligibility threshold of at least {threshold} passes"
                    )
                else:
                    reason = "the locked player-eligibility requirements were not met"

                missing_clauses.append(
                    f"{missing_row['specification_label']} because {reason}"
                )

            sentence += (
                " The contrast was not estimable under "
                + "; ".join(missing_clauses)
                + "."
            )
        lines.append(sentence)

    lines.extend(
        [
            "",
            "INTERPRETATION",
            "",
            (
                "The robustness conclusion is mixed rather than uniformly positive. "
                "Alternative horizons, removal of team identifiers, stricter event "
                "filtering, and retention of the start-state loss produced broadly "
                "stable player profiles. Reducing failure exposure to opponent "
                "transition threat alone materially altered the overall player "
                "ordering, even though the three displayed contrasts retained their "
                "direction. These are descriptive robustness assessments rather than "
                "null-hypothesis significance tests."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def build_caption(pair_robustness: pd.DataFrame, threshold: int) -> str:
    return (
        "Figure S4. Robustness of player-level decision-value profiles and the "
        "illustrative matched-player contrasts. (A) Spearman correlation between "
        "player expected-net-value rankings under each alternative specification "
        "and the primary H=3 Main-xPass, primary-open-play, full-failure-exposure "
        "specification. The dashed line marks the prespecified descriptive benchmark "
        "of 0.80. (B) Agreement in decision–execution profile classifications; the "
        "dashed line marks the prespecified benchmark of 0.70. (C) Expected-net-value "
        "difference for each displayed matched pair across the primary and alternative "
        "specifications. Differences are oriented so that positive values favor the "
        "player with the higher expected net value in the primary specification. "
        "Colors in Panels A and B denote the specification-level interpretation "
        "(stable, boundary-sensitive, or decision-sensitive). The El Shaarawy–Adli "
        "contrast was not estimable in the strict-open-play sample because Stephan "
        f"El Shaarawy did not meet the locked threshold of at least {threshold} passes. "
        "All analyses are descriptive; benchmark lines are not statistical "
        "significance thresholds.\n"
    )


def main() -> None:
    args = parse_args()
    paths = {
        "summary": args.robustness_summary_json.resolve(),
        "stability": args.robustness_stability_csv.resolve(),
        "player": args.robustness_player_results_csv.resolve(),
        "team": args.robustness_team_results_csv.resolve(),
        "pairs": args.selected_player_pairs_csv.resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, args.overwrite)
    configure_matplotlib()

    source_summary = json.loads(paths["summary"].read_text(encoding="utf-8-sig"))
    stability = pd.read_csv(paths["stability"], encoding="utf-8-sig")
    player_results = pd.read_csv(paths["player"], encoding="utf-8-sig", low_memory=False)
    team_results = pd.read_csv(paths["team"], encoding="utf-8-sig")
    selected_pairs = pd.read_csv(paths["pairs"], encoding="utf-8-sig")

    stability["team_order_identical"] = normalize_bool(stability["team_order_identical"])
    interpretation = build_interpretation_table(stability, args.player_threshold)
    pair_robustness = build_pair_robustness(
        player_results, selected_pairs, args.player_threshold
    )

    checks = {
        "reference_specification_present": REFERENCE in set(
            interpretation["comparison_specification"].astype(str)
        ),
        "all_seven_specifications_present": len(interpretation) == 7,
        "three_selected_pairs_present": pair_robustness["selected_pair"].nunique() == 3,
        "team_order_identical_all_specifications": bool(
            interpretation["team_order_identical"].all()
        ),
        "all_estimable_pair_directions_preserved": bool(
            pair_robustness.loc[pair_robustness["estimable"], "direction_preserved"].all()
        ),
        "opponent_only_decision_rank_below_benchmark": bool(
            interpretation.loc[
                interpretation["comparison_specification"].eq(
                    "H3_main_primary_opponent_only"
                ),
                "decision_spearman",
            ].iloc[0]
            < 0.80
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Robustness reporting audit failed:\n"
            + json.dumps(checks, ensure_ascii=False, indent=2, default=json_default)
        )

    interpretation.to_csv(
        output_dir / "robustness_interpretation_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pair_robustness.to_csv(
        output_dir / "matched_player_robustness.csv",
        index=False,
        encoding="utf-8-sig",
    )
    create_robustness_figure(
        interpretation, pair_robustness, selected_pairs, output_dir
    )
    (output_dir / "module04_revised_results_text.txt").write_text(
        build_results_text(interpretation, pair_robustness, args.player_threshold),
        encoding="utf-8",
    )
    (output_dir / "module04_revised_figure_caption.txt").write_text(
        build_caption(pair_robustness, args.player_threshold), encoding="utf-8"
    )

    revised_summary = {
        "execution_status": "PASS",
        "robustness_conclusion": "MIXED",
        "script_version": SCRIPT_VERSION,
        "source_summary_status": source_summary.get("status"),
        "reference_specification": REFERENCE,
        "player_threshold": args.player_threshold,
        "minimum_matches": args.minimum_matches,
        "interpretation": {
            row["comparison_specification"]: {
                "label": row["specification_label"],
                "decision_spearman": float(row["decision_spearman"]),
                "execution_spearman": float(row["execution_spearman"]),
                "quadrant_agreement": float(row["quadrant_agreement"]),
                "team_order_identical": bool(row["team_order_identical"]),
                "robustness_class": row["robustness_class"],
            }
            for _, row in interpretation.iterrows()
        },
        "matched_pair_ranges": {
            f"P{int(pair_number)}": {
                "minimum_difference_per_100": float(
                    group.loc[group["estimable"], "oriented_env_difference_per_100"].min()
                ),
                "maximum_difference_per_100": float(
                    group.loc[group["estimable"], "oriented_env_difference_per_100"].max()
                ),
                "direction_preserved_all_estimable": bool(
                    group.loc[group["estimable"], "direction_preserved"].all()
                ),
                "not_estimable_specifications": group.loc[
                    ~group["estimable"], "specification"
                ].tolist(),
            }
            for pair_number, group in pair_robustness.groupby("selected_pair", sort=True)
        },
        "checks": checks,
        "inputs": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in paths.items()
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (output_dir / "module04_revised_summary.json").write_text(
        json.dumps(revised_summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_sha256sums(output_dir)

    print("=" * 96)
    print("Module 04 robustness reporting completed")
    print("=" * 96)
    print("Execution status: PASS")
    print("Substantive robustness conclusion: MIXED")
    print(f"Player threshold: {args.player_threshold}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
