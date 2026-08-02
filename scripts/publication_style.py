# -*- coding: utf-8 -*-
"""
publication_style.py

Shared plotting style for:
    05_full_sample_risk_reward_story_final.py
    06_three_team_illustrative_application_final.py

The two modules use the same dimensions, typography, palette, line weights,
error bars, and export settings. Edit this file only when a journal requires
different dimensions or typography.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


@dataclass(frozen=True)
class PublicationStyle:
    # Double-column figures.
    double_column_width_in: float = 7.20
    figure2_height_in: float = 3.30
    figure3_height_in: float = 5.25

    # Compact supplementary dumbbell plot.
    supplementary_width_in: float = 5.05
    supplementary_height_in: float = 2.85

    dpi: int = 600

    # Typography.
    font_family: str = "Arial"
    panel_label_size: float = 10.0
    panel_title_size: float = 8.3
    axis_label_size: float = 7.8
    tick_label_size: float = 6.8
    legend_size: float = 6.7
    annotation_size: float = 6.2
    heatmap_value_size: float = 6.1
    group_label_size: float = 6.8

    # Geometry.
    axis_line_width: float = 0.72
    reference_line_width: float = 0.90
    main_line_width: float = 1.75
    secondary_line_width: float = 1.10
    connector_line_width: float = 1.05
    error_line_width: float = 1.15
    marker_size: float = 4.25
    cap_size: float = 2.4

    # Restrained journal palette.
    deep_blue: str = "#315F8C"
    muted_orange: str = "#C8753D"
    muted_teal: str = "#4F8A84"
    dark_slate: str = "#404B56"
    medium_gray: str = "#81888E"
    warm_gray: str = "#B5AFA5"
    light_gray: str = "#BDC3C8"
    grid_gray: str = "#DDE1E4"
    near_white: str = "#F7F8F9"
    muted_red: str = "#A95F50"

    # Team colors.
    leverkusen: str = "#315F8C"
    athletic: str = "#C8753D"
    monaco: str = "#4F8A84"

    # Pass-height colors.
    ground: str = "#5D7FA3"
    low: str = "#B6B0A6"
    high: str = "#C8753D"


STYLE = PublicationStyle()


def configure_matplotlib(style: PublicationStyle = STYLE) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                style.font_family,
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "font.size": style.tick_label_size,
            "axes.labelsize": style.axis_label_size,
            "axes.titlesize": style.panel_title_size,
            "xtick.labelsize": style.tick_label_size,
            "ytick.labelsize": style.tick_label_size,
            "legend.fontsize": style.legend_size,
            "axes.linewidth": style.axis_line_width,
            "xtick.major.width": style.axis_line_width,
            "ytick.major.width": style.axis_line_width,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def clean_axis(
    ax: plt.Axes,
    *,
    grid_axis: str | None = None,
    style: PublicationStyle = STYLE,
) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(style.axis_line_width)
    ax.spines["bottom"].set_linewidth(style.axis_line_width)
    ax.tick_params(direction="out")

    if grid_axis is not None:
        ax.grid(
            True,
            axis=grid_axis,
            color=style.grid_gray,
            linewidth=0.42,
            alpha=0.62,
            zorder=0,
        )


def add_panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.13,
    y: float = 1.07,
    style: PublicationStyle = STYLE,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=style.panel_label_size,
        fontweight="bold",
        va="top",
        ha="left",
    )


def annotate_horizontal_estimate(
    ax: plt.Axes,
    x: float,
    y: float,
    *,
    text: str | None = None,
    pad_fraction: float = 0.025,
    style: PublicationStyle = STYLE,
) -> None:
    """Place a compact estimate label without covering its marker."""
    x_min, x_max = ax.get_xlim()
    pad = max((x_max - x_min) * pad_fraction, 0.015)
    place_right = x <= x_max - 0.17 * (x_max - x_min)

    ax.text(
        x + pad if place_right else x - pad,
        y,
        text if text is not None else f"{x:.2f}",
        ha="left" if place_right else "right",
        va="center",
        fontsize=style.annotation_size,
        color=style.dark_slate,
        clip_on=False,
    )


def save_figure_all_formats(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    style: PublicationStyle = STYLE,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_dir / f"{stem}.png",
        dpi=style.dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_dir / f"{stem}.svg",
        bbox_inches="tight",
        facecolor="white",
    )


def team_color_map(style: PublicationStyle = STYLE) -> dict[str, str]:
    return {
        "Bayer Leverkusen 2023/24": style.leverkusen,
        "Athletic Club 2015/16": style.athletic,
        "AS Monaco 2015/16": style.monaco,
    }


def height_color_map(style: PublicationStyle = STYLE) -> dict[str, str]:
    return {
        "Ground Pass": style.ground,
        "Low Pass": style.low,
        "High Pass": style.high,
    }


def profile_diverging_cmap(
    style: PublicationStyle = STYLE,
) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "profile_diverging_final",
        [
            style.deep_blue,
            style.near_white,
            style.muted_red,
        ],
        N=256,
    )
