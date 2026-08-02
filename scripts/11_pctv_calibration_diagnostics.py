#!/usr/bin/env python
"""
Generate PCTV calibration diagnostics from existing out-of-fold pass-value files.

Only threat_start_oof is evaluated against the observed PCTV target because it is
the prediction for the actually observed pass-associated state. threat_end_oof
and opponent_threat_end_oof are counterfactual state evaluations and therefore
do not have directly observed calibration labels.

Outputs:
- pctv_calibration_summary.csv
- pctv_reliability_bins.csv
- pctv_h3_reliability.png
- pctv_h3_reliability.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1", type=Path, required=True)
    parser.add_argument("--h3", type=Path, required=True)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--ece_bins",
        type=int,
        default=20,
        help="Number of fixed-width bins used for ECE.",
    )
    parser.add_argument(
        "--reliability_bins",
        type=int,
        default=20,
        help="Number of equal-frequency bins used for the reliability curve.",
    )
    return parser.parse_args()


def calibration_intercept_slope(
    y: np.ndarray,
    p: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> tuple[float, float]:
    """
    Fit logit(P(Y=1)) = intercept + slope * logit(p)
    by Newton-Raphson without regularization.
    """
    eps = 1e-7
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(y, dtype=np.float64)
    x = np.log(p / (1.0 - p))

    beta = np.array([0.0, 1.0], dtype=np.float64)

    for _ in range(max_iter):
        eta = beta[0] + beta[1] * x
        fitted = expit(eta)
        residual = y - fitted
        weight = np.clip(fitted * (1.0 - fitted), 1e-12, None)

        score = np.array(
            [
                residual.sum(),
                np.dot(residual, x),
            ],
            dtype=np.float64,
        )
        information = np.array(
            [
                [weight.sum(), np.dot(weight, x)],
                [np.dot(weight, x), np.dot(weight, x * x)],
            ],
            dtype=np.float64,
        )

        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(information, score, rcond=None)[0]

        beta += step
        if np.max(np.abs(step)) < tol:
            break

    return float(beta[0]), float(beta[1])


def fixed_width_ece(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int = 20,
) -> tuple[float, float]:
    """Return ECE and maximum absolute calibration gap."""
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)

    bin_index = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = len(y)
    ece = 0.0
    max_gap = 0.0

    for bin_id in range(n_bins):
        mask = bin_index == bin_id
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(float(p[mask].mean()) - float(y[mask].mean()))
        ece += (count / total) * gap
        max_gap = max(max_gap, gap)

    return float(ece), float(max_gap)


def equal_frequency_bins(
    y: np.ndarray,
    p: np.ndarray,
    horizon: int,
    n_bins: int = 20,
) -> pd.DataFrame:
    """Create equal-frequency reliability bins for plotting and reporting."""
    frame = pd.DataFrame(
        {
            "observed": np.asarray(y, dtype=np.int8),
            "predicted": np.asarray(p, dtype=np.float64),
        }
    )

    frame["bin"] = pd.qcut(
        frame["predicted"],
        q=n_bins,
        labels=False,
        duplicates="drop",
    )

    grouped = (
        frame.groupby("bin", observed=True)
        .agg(
            passes=("observed", "size"),
            predicted_min=("predicted", "min"),
            predicted_max=("predicted", "max"),
            mean_predicted=("predicted", "mean"),
            observed_rate=("observed", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "horizon", horizon)
    grouped["absolute_gap"] = (
        grouped["mean_predicted"] - grouped["observed_rate"]
    ).abs()
    return grouped


def load_horizon(path: Path, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label = f"shot_creation_within_{horizon}_passes"
    required = ["match_id", label, "threat_start_oof"]

    header = pd.read_csv(
        path,
        compression="infer",
        nrows=0,
        encoding="utf-8-sig",
    ).columns.tolist()
    missing = sorted(set(required) - set(header))
    if missing:
        raise RuntimeError(
            f"{path} is missing required columns: {missing}"
        )

    frame = pd.read_csv(
        path,
        compression="infer",
        usecols=required,
        encoding="utf-8-sig",
        low_memory=False,
    )

    y = pd.to_numeric(frame[label], errors="coerce").to_numpy(np.float64)
    p = pd.to_numeric(
        frame["threat_start_oof"],
        errors="coerce",
    ).to_numpy(np.float64)
    match_id = frame["match_id"].astype("string").to_numpy()

    valid = np.isfinite(y) & np.isfinite(p)
    if not valid.all():
        raise RuntimeError(
            f"{path} contains {(~valid).sum():,} rows with missing/nonfinite labels or probabilities."
        )
    if not np.isin(y, [0.0, 1.0]).all():
        raise RuntimeError(f"{path} contains nonbinary target values.")
    if ((p < 0.0) | (p > 1.0)).any():
        raise RuntimeError(f"{path} contains probabilities outside [0, 1].")

    return y.astype(np.int8), p, match_id


def make_h3_plot(
    bins: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    max_value = float(
        max(
            bins["mean_predicted"].max(),
            bins["observed_rate"].max(),
            0.10,
        )
    )
    axis_max = min(1.0, max_value * 1.08)

    ax.plot([0.0, axis_max], [0.0, axis_max], linestyle="--", label="Ideal calibration")
    ax.plot(
        bins["mean_predicted"],
        bins["observed_rate"],
        marker="o",
        label="H = 3 PCTV",
    )

    ax.set_xlim(0.0, axis_max)
    ax.set_ylim(0.0, axis_max)
    ax.set_xlabel("Mean predicted shot-creation probability")
    ax.set_ylabel("Observed shot-creation rate")
    ax.set_title("PCTV reliability for the primary H = 3 target")
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(output_dir / "pctv_h3_reliability.png", dpi=300)
    fig.savefig(output_dir / "pctv_h3_reliability.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = {1: args.h1, 3: args.h3, 5: args.h5}
    summary_rows: list[dict[str, float | int]] = []
    reliability_parts: list[pd.DataFrame] = []

    for horizon, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)

        print(f"[load] H={horizon}: {path}")
        y, p, match_id = load_horizon(path, horizon)

        intercept, slope = calibration_intercept_slope(y, p)
        ece, maximum_gap = fixed_width_ece(
            y,
            p,
            n_bins=args.ece_bins,
        )
        bins = equal_frequency_bins(
            y,
            p,
            horizon=horizon,
            n_bins=args.reliability_bins,
        )
        reliability_parts.append(bins)

        summary_rows.append(
            {
                "horizon": horizon,
                "passes": int(len(y)),
                "matches": int(pd.Series(match_id).nunique()),
                "observed_positive_rate": float(y.mean()),
                "mean_predicted_probability": float(p.mean()),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                f"ece_{args.ece_bins}_fixed_width_bins": ece,
                f"maximum_gap_{args.ece_bins}_fixed_width_bins": maximum_gap,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("horizon")
    reliability = pd.concat(reliability_parts, ignore_index=True)

    summary.to_csv(
        args.output_dir / "pctv_calibration_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reliability.to_csv(
        args.output_dir / "pctv_reliability_bins.csv",
        index=False,
        encoding="utf-8-sig",
    )

    h3_bins = reliability.loc[reliability["horizon"] == 3].copy()
    make_h3_plot(h3_bins, args.output_dir)

    print("\nPCTV calibration summary")
    print(summary.to_string(index=False))
    print(f"\nOutputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
