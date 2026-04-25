# -*- coding: utf-8 -*-
"""
Sensitivity analysis runner for the football passing risk-reward pipeline.

This script runs three groups of checks:
1. xPass structural sensitivity, by changing feature-engineering and optimizer settings.
2. Risk-reward structural sensitivity, by changing continuation-value and passer-marginal settings.
3. Formula-level sensitivity, by recomputing gain, cost, and expected reward from a baseline
   risk-reward output under alternative weights.

The script is designed for a public reproducibility repository. It calls the main pipeline scripts
through command-line arguments and does not patch source code or rely on local computer paths.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ----------------------------- Scenario definitions -----------------------------
XPASS_SCENARIOS: List[Dict[str, Any]] = [
    {"name": "baseline", "args": []},
    {"name": "no_dir_flags", "args": ["--disable_dir_flags"]},
    {"name": "no_region_bins", "args": ["--disable_region_bins"]},
    {"name": "no_interactions", "args": ["--disable_interactions"]},
    {"name": "invscaling", "args": ["--lr_strategy", "invscaling", "--power_t", "0.5"]},
    {"name": "logreg_saga", "args": ["--use_logreg_saga"]},
]

RISK_STRUCTURAL_SCENARIOS: List[Dict[str, Any]] = [
    {"name": "baseline", "args": []},
    {"name": "cv_short_h2_g090", "args": ["--h_steps", "2", "--gamma", "0.90", "--window", "5"]},
    {"name": "cv_long_h5_g098", "args": ["--h_steps", "5", "--gamma", "0.98", "--window", "5"]},
    {"name": "window3", "args": ["--window", "3"]},
    {"name": "window7", "args": ["--window", "7"]},
    {"name": "prior20", "args": ["--passer_prior_k", "20.0"]},
    {"name": "prior100", "args": ["--passer_prior_k", "100.0"]},
]

RISK_FORMULA_SCENARIOS: List[Dict[str, Any]] = [
    {"name": "baseline", "lambda_cv": 0.30, "passer_weight": 0.05, "cost_w_turn": 0.70, "cost_w_start": 0.15, "cost_w_end": 0.15},
    {"name": "lambda_low", "lambda_cv": 0.15, "passer_weight": 0.05, "cost_w_turn": 0.70, "cost_w_start": 0.15, "cost_w_end": 0.15},
    {"name": "lambda_high", "lambda_cv": 0.45, "passer_weight": 0.05, "cost_w_turn": 0.70, "cost_w_start": 0.15, "cost_w_end": 0.15},
    {"name": "passer_low", "lambda_cv": 0.30, "passer_weight": 0.02, "cost_w_turn": 0.70, "cost_w_start": 0.15, "cost_w_end": 0.15},
    {"name": "passer_high", "lambda_cv": 0.30, "passer_weight": 0.10, "cost_w_turn": 0.70, "cost_w_start": 0.15, "cost_w_end": 0.15},
    {"name": "cost_60_20_20", "lambda_cv": 0.30, "passer_weight": 0.05, "cost_w_turn": 0.60, "cost_w_start": 0.20, "cost_w_end": 0.20},
    {"name": "cost_80_10_10", "lambda_cv": 0.30, "passer_weight": 0.05, "cost_w_turn": 0.80, "cost_w_start": 0.10, "cost_w_end": 0.10},
]


# ----------------------------- Command line interface -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sensitivity analyses for xPass and risk-reward pass-value metrics."
    )
    parser.add_argument(
        "--xpass_input_csv",
        default=None,
        help="Input pass-event CSV for xPass structural sensitivity. Required unless --skip_xpass is used.",
    )
    parser.add_argument(
        "--risk_input_csv",
        default=None,
        help="Input CSV with xPass probabilities for risk-reward structural sensitivity. Required unless --skip_risk_structural is used.",
    )
    parser.add_argument(
        "--baseline_risk_csv",
        default=None,
        help="Baseline risk-reward output CSV for formula-level sensitivity. If omitted, the runner uses the structural baseline output when available.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for all sensitivity-analysis outputs.",
    )
    parser.add_argument(
        "--scripts_dir",
        default=None,
        help="Directory containing 02_train_xpass.py and 04_compute_risk_reward.py. Defaults to the current script directory.",
    )
    parser.add_argument("--python_exe", default=sys.executable, help="Python executable used to run subprocesses.")
    parser.add_argument("--skip_xpass", action="store_true", help="Skip xPass structural sensitivity.")
    parser.add_argument("--skip_risk_structural", action="store_true", help="Skip risk-reward structural sensitivity.")
    parser.add_argument("--skip_risk_formula", action="store_true", help="Skip formula-level sensitivity.")
    parser.add_argument(
        "--export_formula_event_files",
        action="store_true",
        help="Export event-level formula-sensitivity files. By default, only the summary table is saved.",
    )
    parser.add_argument(
        "--xpass_extra_args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Optional extra arguments passed to every xPass run. Place this option last if used.",
    )
    return parser.parse_args()


# ----------------------------- Utility functions -----------------------------
def run_command(command: List[str], log_dir: Path, log_prefix: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    (log_dir / f"{log_prefix}_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (log_dir / f"{log_prefix}_stderr.log").write_text(completed.stderr, encoding="utf-8")
    (log_dir / f"{log_prefix}_command.txt").write_text(" ".join(command), encoding="utf-8")

    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {completed.returncode}: {' '.join(command)}\n"
            f"Last stderr lines:\n{completed.stderr[-4000:]}"
        )


def safe_spearman(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if mask.sum() < 3:
        return float("nan")
    return float(aa[mask].rank().corr(bb[mask].rank(), method="pearson"))


def estimate_zero_cross_risk(
    df: pd.DataFrame,
    risk_col: str = "risk",
    value_col: str = "expected_reward",
    n_bins: int = 20,
) -> float:
    risk = pd.to_numeric(df[risk_col], errors="coerce")
    value = pd.to_numeric(df[value_col], errors="coerce")
    mask = risk.notna() & value.notna()
    if mask.sum() < 100:
        return float("nan")

    tmp = pd.DataFrame({"risk": risk[mask], "value": value[mask]})
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    tmp["risk_bin"] = pd.cut(tmp["risk"], bins=bins, labels=False, include_lowest=True)
    grouped = tmp.groupby("risk_bin", observed=True)["value"].mean().reset_index()
    if grouped.empty:
        return float("nan")

    centers = bins[:-1] + np.diff(bins) / 2.0
    grouped["center"] = grouped["risk_bin"].map(lambda i: centers[int(i)] if pd.notna(i) else np.nan)
    values = grouped["value"].to_numpy(dtype=float)
    xs = grouped["center"].to_numpy(dtype=float)

    for idx in range(1, len(values)):
        left_value = values[idx - 1]
        right_value = values[idx]
        if np.isfinite(left_value) and np.isfinite(right_value) and left_value > 0 and right_value <= 0:
            left_x = xs[idx - 1]
            right_x = xs[idx]
            if right_value == left_value:
                return float(right_x)
            return float(left_x + (0.0 - left_value) * (right_x - left_x) / (right_value - left_value))
    return float("nan")


def summarize_risk_reward(df: pd.DataFrame, baseline_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    expected = pd.to_numeric(df["expected_reward"], errors="coerce")
    gain = pd.to_numeric(df["GainOnSuccess"], errors="coerce")
    cost = pd.to_numeric(df["CostOnFailure"], errors="coerce")

    summary: Dict[str, Any] = {
        "n": int(len(df)),
        "mean_expected_reward": float(expected.mean()),
        "median_expected_reward": float(expected.median()),
        "positive_reward_rate": float((expected > 0).mean()),
        "mean_gain": float(gain.mean()),
        "mean_cost": float(cost.mean()),
        "zero_cross_risk": estimate_zero_cross_risk(df),
    }

    if baseline_df is not None and len(baseline_df) == len(df):
        summary["spearman_expected_reward_vs_baseline"] = safe_spearman(
            df["expected_reward"], baseline_df["expected_reward"]
        )
    else:
        summary["spearman_expected_reward_vs_baseline"] = float("nan")
    return summary


def require_file(path: Path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{message}: {path}")


# ----------------------------- Sensitivity routines -----------------------------
def run_xpass_sensitivity(
    *,
    input_csv: Path,
    output_dir: Path,
    scripts_dir: Path,
    python_exe: str,
    extra_args: Optional[List[str]] = None,
) -> pd.DataFrame:
    script = scripts_dir / "02_train_xpass.py"
    require_file(script, "xPass training script not found")
    require_file(input_csv, "xPass input CSV not found")

    xpass_dir = output_dir / "xpass_structural"
    xpass_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for scenario in XPASS_SCENARIOS:
        name = scenario["name"]
        scenario_dir = xpass_dir / name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        command = [
            python_exe,
            str(script),
            "--input_csv",
            str(input_csv),
            "--output_dir",
            str(scenario_dir),
        ] + list(scenario["args"])
        if extra_args:
            command.extend(extra_args)

        run_command(command, scenario_dir, f"xpass_{name}")
        metrics_path = scenario_dir / "xpass_test_metrics.csv"
        require_file(metrics_path, "xPass metrics file not generated")
        metrics = pd.read_csv(metrics_path)
        row = metrics.iloc[0].to_dict()
        row["scenario"] = name
        row["scenario_args"] = " ".join(scenario["args"])
        rows.append(row)

    summary = pd.DataFrame(rows)
    ordered_cols = ["scenario", "scenario_args"] + [c for c in summary.columns if c not in {"scenario", "scenario_args"}]
    summary = summary[ordered_cols]
    summary_path = xpass_dir / "xpass_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary


def run_risk_structural_sensitivity(
    *,
    input_csv: Path,
    output_dir: Path,
    scripts_dir: Path,
    python_exe: str,
) -> pd.DataFrame:
    script = scripts_dir / "04_compute_risk_reward.py"
    require_file(script, "Risk-reward script not found")
    require_file(input_csv, "Risk-reward input CSV not found")

    risk_dir = output_dir / "risk_structural"
    risk_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    baseline_df: Optional[pd.DataFrame] = None

    for scenario in RISK_STRUCTURAL_SCENARIOS:
        name = scenario["name"]
        scenario_dir = risk_dir / name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        command = [
            python_exe,
            str(script),
            "--input_csv",
            str(input_csv),
            "--output_dir",
            str(scenario_dir),
        ] + list(scenario["args"])

        run_command(command, scenario_dir, f"risk_{name}")
        rr_path = scenario_dir / "risk_reward_per_pass_v4.csv"
        require_file(rr_path, "Risk-reward output file not generated")
        rr = pd.read_csv(rr_path, low_memory=False)
        if name == "baseline":
            baseline_df = rr.copy()
        row = summarize_risk_reward(rr, baseline_df=baseline_df if name != "baseline" else None)
        row["scenario"] = name
        row["scenario_args"] = " ".join(scenario["args"])
        rows.append(row)

    summary = pd.DataFrame(rows)
    ordered_cols = ["scenario", "scenario_args"] + [c for c in summary.columns if c not in {"scenario", "scenario_args"}]
    summary = summary[ordered_cols]
    summary_path = risk_dir / "risk_structural_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary


def run_risk_formula_sensitivity(
    *,
    baseline_csv: Path,
    output_dir: Path,
    export_event_files: bool = False,
) -> pd.DataFrame:
    require_file(baseline_csv, "Baseline risk-reward CSV not found")
    formula_dir = output_dir / "risk_formula"
    formula_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(baseline_csv, low_memory=False)
    required_cols = [
        "xpass_prob",
        "EPV_start",
        "EPV_end",
        "receiver_CV_cond",
        "passer_marginal",
        "EPV_opp_turnover",
        "EPV_opp_start",
        "EPV_opp_end",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Baseline file is missing required columns: {missing}")

    xpass = pd.to_numeric(df["xpass_prob"], errors="coerce").clip(0.001, 0.999)
    direct_gain = pd.to_numeric(df["EPV_end"], errors="coerce") - pd.to_numeric(df["EPV_start"], errors="coerce")
    cv = pd.to_numeric(df["receiver_CV_cond"], errors="coerce")
    passer = pd.to_numeric(df["passer_marginal"], errors="coerce")
    opp_turn = pd.to_numeric(df["EPV_opp_turnover"], errors="coerce")
    opp_start = pd.to_numeric(df["EPV_opp_start"], errors="coerce")
    opp_end = pd.to_numeric(df["EPV_opp_end"], errors="coerce")

    rows: List[Dict[str, Any]] = []
    baseline_df: Optional[pd.DataFrame] = None

    for scenario in RISK_FORMULA_SCENARIOS:
        gain = direct_gain + scenario["lambda_cv"] * cv + scenario["passer_weight"] * passer
        cost = (
            scenario["cost_w_turn"] * opp_turn
            + scenario["cost_w_start"] * opp_start
            + scenario["cost_w_end"] * opp_end
        )
        expected_reward = xpass * gain - (1.0 - xpass) * cost
        risk = (1.0 - xpass).clip(lower=1e-9)

        current = df.copy()
        current["GainOnSuccess"] = gain
        current["CostOnFailure"] = cost
        current["expected_reward"] = expected_reward
        current["risk"] = risk
        current["net_risk_reward_ratio"] = expected_reward / risk

        if scenario["name"] == "baseline":
            baseline_df = current.copy()

        summary = summarize_risk_reward(
            current,
            baseline_df=baseline_df if scenario["name"] != "baseline" else None,
        )
        summary.update({
            "scenario": scenario["name"],
            "lambda_cv": scenario["lambda_cv"],
            "passer_weight": scenario["passer_weight"],
            "cost_w_turn": scenario["cost_w_turn"],
            "cost_w_start": scenario["cost_w_start"],
            "cost_w_end": scenario["cost_w_end"],
        })
        rows.append(summary)

        if export_event_files:
            event_path = formula_dir / f"risk_formula_{scenario['name']}.csv"
            current.to_csv(event_path, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(rows)
    ordered_cols = [
        "scenario",
        "lambda_cv",
        "passer_weight",
        "cost_w_turn",
        "cost_w_start",
        "cost_w_end",
    ] + [
        col
        for col in summary_df.columns
        if col not in {"scenario", "lambda_cv", "passer_weight", "cost_w_turn", "cost_w_start", "cost_w_end"}
    ]
    summary_df = summary_df[ordered_cols]
    summary_path = formula_dir / "risk_formula_sensitivity_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_df


# ----------------------------- Main -----------------------------
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    scripts_dir = Path(args.scripts_dir).expanduser().resolve() if args.scripts_dir else Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_from_structural: Optional[Path] = None

    if not args.skip_xpass:
        if args.xpass_input_csv is None:
            raise ValueError("--xpass_input_csv is required unless --skip_xpass is used.")
        xpass_summary = run_xpass_sensitivity(
            input_csv=Path(args.xpass_input_csv).expanduser().resolve(),
            output_dir=output_dir,
            scripts_dir=scripts_dir,
            python_exe=args.python_exe,
            extra_args=args.xpass_extra_args,
        )
        print("\n[xPass structural sensitivity]")
        print(xpass_summary)

    if not args.skip_risk_structural:
        if args.risk_input_csv is None:
            raise ValueError("--risk_input_csv is required unless --skip_risk_structural is used.")
        risk_summary = run_risk_structural_sensitivity(
            input_csv=Path(args.risk_input_csv).expanduser().resolve(),
            output_dir=output_dir,
            scripts_dir=scripts_dir,
            python_exe=args.python_exe,
        )
        baseline_from_structural = output_dir / "risk_structural" / "baseline" / "passes_with_appended_new_cols_v4.csv"
        print("\n[risk-reward structural sensitivity]")
        print(risk_summary)

    if not args.skip_risk_formula:
        if args.baseline_risk_csv is not None:
            baseline_csv = Path(args.baseline_risk_csv).expanduser().resolve()
        elif baseline_from_structural is not None and baseline_from_structural.exists():
            baseline_csv = baseline_from_structural
        else:
            raise ValueError(
                "Formula-level sensitivity requires --baseline_risk_csv or an available structural baseline output."
            )
        formula_summary = run_risk_formula_sensitivity(
            baseline_csv=baseline_csv,
            output_dir=output_dir,
            export_event_files=args.export_formula_event_files,
        )
        print("\n[risk-reward formula sensitivity]")
        print(formula_summary)

    print(f"\nSensitivity analysis completed. Outputs were saved to: {output_dir}")


if __name__ == "__main__":
    main()
