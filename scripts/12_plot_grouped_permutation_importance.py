#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create Supplementary Figure S2 from the grouped permutation
importance summary for the locked Full xPass model.

Inputs
------
- grouped_permutation_importance_summary.csv

Outputs
-------
- Figure_S2_grouped_permutation_importance.png
- Figure_S2_grouped_permutation_importance.pdf
- Figure_S2_grouped_permutation_importance.svg
- Figure_S2_grouped_permutation_importance_caption.txt

The figure is a four-panel forest-style summary showing mean grouped
permutation importance and percentile-based 95% intervals for four metrics:
log loss, Brier score, ROC-AUC, and failure PR-AUC.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PANEL_ORDER = [
    ("increase_log_loss", "Increase in log loss"),
    ("increase_brier_score", "Increase in Brier score"),
    ("decrease_roc_auc", "Decrease in ROC-AUC"),
    ("decrease_failure_pr_auc", "Decrease in failure PR-AUC"),
]

FIXED_GROUP_ORDER = [
    "Pass geometry",
    "Pass characteristics",
    "Tactical context",
    "Pass action flags",
    "Pressure status",
    "Team identity",
    "Match time",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Supplementary Figure S2 from grouped permutation importance results."
    )
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_input(df: pd.DataFrame) -> None:
    required = {
        "feature_group_label",
        "metric",
        "mean_importance",
        "percentile_2_5",
        "percentile_97_5",
        "repetitions",
        "permutation_scope",
    }
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    metrics = set(df["metric"].unique())
    expected_metrics = {m for m, _ in PANEL_ORDER}
    if metrics != expected_metrics:
        raise ValueError(
            f"Unexpected metric set: {sorted(metrics)}; expected {sorted(expected_metrics)}"
        )

    labels = set(df["feature_group_label"].unique())
    if labels != set(FIXED_GROUP_ORDER):
        raise ValueError(
            "Unexpected feature-group labels. "
            f"Found {sorted(labels)}; expected {sorted(FIXED_GROUP_ORDER)}"
        )


def prepare_panel(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    panel = df.loc[df["metric"].eq(metric), [
        "feature_group_label", "mean_importance", "percentile_2_5", "percentile_97_5"
    ]].copy()
    panel["feature_group_label"] = pd.Categorical(
        panel["feature_group_label"], categories=FIXED_GROUP_ORDER, ordered=True
    )
    panel = panel.sort_values("feature_group_label", ascending=False).reset_index(drop=True)
    return panel


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stem = "Figure_S2_grouped_permutation_importance"
    out_png = args.output_dir / f"{stem}.png"
    out_pdf = args.output_dir / f"{stem}.pdf"
    out_svg = args.output_dir / f"{stem}.svg"
    out_caption = args.output_dir / f"{stem}_caption.txt"

    if not args.overwrite:
        existing = [p for p in [out_png, out_pdf, out_svg, out_caption] if p.exists()]
        if existing:
            raise FileExistsError(
                "Outputs already exist. Use --overwrite to replace them.\n"
                + "\n".join(str(p) for p in existing)
            )

    df = pd.read_csv(args.input_csv)
    validate_input(df)

    repetitions = int(df["repetitions"].iloc[0])
    scope = str(df["permutation_scope"].iloc[0]).replace("_", " ")

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), sharey=True, constrained_layout=True)
    axes = axes.ravel()
    letters = ["A", "B", "C", "D"]

    # Colors aligned with a clean publication palette
    point_color = "#1F4E79"
    interval_color = "#1F4E79"
    grid_color = "#D9D9D9"
    zero_color = "#B0B0B0"

    for ax, (metric, label), letter in zip(axes, PANEL_ORDER, letters):
        panel = prepare_panel(df, metric)
        y = np.arange(panel.shape[0])
        mean = panel["mean_importance"].to_numpy(float)
        lower = panel["percentile_2_5"].to_numpy(float)
        upper = panel["percentile_97_5"].to_numpy(float)
        xerr = np.vstack([mean - lower, upper - mean])

        ax.errorbar(
            mean,
            y,
            xerr=xerr,
            fmt="o",
            color=point_color,
            ecolor=interval_color,
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            elinewidth=1.15,
            capsize=2.6,
            zorder=3,
        )
        ax.axvline(0, color=zero_color, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.grid(True, axis="x", color=grid_color, linewidth=0.45, alpha=0.75, zorder=0)
        ax.set_title(label)
        ax.text(-0.12, 1.03, letter, transform=ax.transAxes, fontsize=14, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(panel["feature_group_label"].tolist())
        ax.tick_params(axis="y", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if metric in ("increase_log_loss", "increase_brier_score"):
            ax.set_xlabel(label)
        else:
            ax.set_xlabel(label)

        xmin = min(lower.min(), 0)
        xmax = max(upper.max(), 0)
        pad = (xmax - xmin) * 0.08 if xmax > xmin else 0.01
        ax.set_xlim(xmin - pad, xmax + pad)

    # Shared y-label only on left column would be ideal, but repeated panel labels are acceptable.
    fig.suptitle(
        "Supplementary Figure S2. Grouped permutation importance of the Full xPass model",
        y=1.02,
        fontsize=12,
    )

    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    caption = (
        "Supplementary Figure S2. Grouped permutation importance of the Full xPass model on the "
        "locked independent test set. Each point shows the mean change in model performance after "
        "permuting one pre-specified feature group, and horizontal lines indicate percentile-based "
        "95% intervals across repeated permutations. Panel A shows the increase in log loss, panel "
        "B the increase in Brier score, panel C the decrease in ROC-AUC, and panel D the decrease "
        "in failure PR-AUC. Positive values indicate that model performance worsened after the "
        f"feature group was permuted. All permutations were performed {scope} and repeated {repetitions} times. "
        "These grouped permutation importance estimates are descriptive measures of model reliance and "
        "should not be interpreted as causal effects."
    )
    out_caption.write_text(caption, encoding="utf-8")

    print("=" * 88)
    print("Supplementary Figure S2 created successfully")
    print("=" * 88)
    print(f"Input summary: {args.input_csv}")
    print(f"Output PNG:   {out_png}")
    print(f"Output PDF:   {out_pdf}")
    print(f"Output SVG:   {out_svg}")
    print(f"Caption TXT:  {out_caption}")
    print(f"Permutation scope: {scope}")
    print(f"Repetitions: {repetitions}")


if __name__ == "__main__":
    main()
