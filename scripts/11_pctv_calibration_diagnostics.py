#!/usr/bin/env python
"""Generate pooled out-of-fold PCTV calibration diagnostics.

Only ``threat_start_oof`` is evaluated against the observed PCTV target.
The end-state and opponent-view estimates are counterfactual state
evaluations and do not have directly observed calibration labels.
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
import numpy as np
import pandas as pd
from scipy.special import expit



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
        description="Evaluate pooled out-of-fold PCTV calibration for H=1, H=3, and H=5."
    )
    parser.add_argument("--h1", type=Path, required=True)
    parser.add_argument("--h3", type=Path, required=True)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--ece_bins", type=int, default=20)
    parser.add_argument("--reliability_bins", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    expected = [
        "pctv_calibration_summary.csv",
        "pctv_reliability_bins.csv",
        "pctv_h3_reliability.png",
        "pctv_h3_reliability.pdf",
        "pctv_h3_reliability.svg",
        "pctv_calibration_audit.json",
    ]
    existing = [path / name for name in expected if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Outputs already exist. Use --overwrite to replace them:\n"
            + "\n".join(str(item) for item in existing)
        )


def calibration_intercept_slope(
    y: np.ndarray,
    p: np.ndarray,
    max_iter: int = 100,
    tolerance: float = 1e-10,
) -> tuple[float, float]:
    """Fit ``logit(P(Y=1)) = intercept + slope * logit(p)``."""
    epsilon = 1e-7
    probability = np.clip(np.asarray(p, dtype=np.float64), epsilon, 1.0 - epsilon)
    outcome = np.asarray(y, dtype=np.float64)
    predictor = np.log(probability / (1.0 - probability))
    beta = np.array([0.0, 1.0], dtype=np.float64)

    for _ in range(max_iter):
        linear_predictor = beta[0] + beta[1] * predictor
        fitted = expit(linear_predictor)
        residual = outcome - fitted
        weight = np.clip(fitted * (1.0 - fitted), 1e-12, None)
        score = np.array(
            [residual.sum(), np.dot(residual, predictor)],
            dtype=np.float64,
        )
        information = np.array(
            [
                [weight.sum(), np.dot(weight, predictor)],
                [np.dot(weight, predictor), np.dot(weight, predictor * predictor)],
            ],
            dtype=np.float64,
        )
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(information, score, rcond=None)[0]
        beta += step
        if np.max(np.abs(step)) < tolerance:
            break

    return float(beta[0]), float(beta[1])


def fixed_width_ece(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int,
) -> tuple[float, float]:
    outcome = np.asarray(y, dtype=np.float64)
    probability = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    bin_index = np.minimum((probability * n_bins).astype(int), n_bins - 1)
    total = len(outcome)
    ece = 0.0
    maximum_gap = 0.0

    for bin_id in range(n_bins):
        mask = bin_index == bin_id
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(float(probability[mask].mean()) - float(outcome[mask].mean()))
        ece += (count / total) * gap
        maximum_gap = max(maximum_gap, gap)

    return float(ece), float(maximum_gap)


def equal_frequency_bins(
    y: np.ndarray,
    p: np.ndarray,
    horizon: int,
    n_bins: int,
) -> pd.DataFrame:
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


def load_horizon(
    path: Path,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = f"shot_creation_within_{horizon}_passes"
    required = ["match_id", target, "threat_start_oof"]
    columns = pd.read_csv(
        path,
        compression="infer",
        nrows=0,
        encoding="utf-8-sig",
    ).columns.tolist()
    missing = sorted(set(required) - set(columns))
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")

    frame = pd.read_csv(
        path,
        compression="infer",
        usecols=required,
        encoding="utf-8-sig",
        low_memory=False,
    )
    outcome = pd.to_numeric(frame[target], errors="coerce").to_numpy(np.float64)
    probability = pd.to_numeric(
        frame["threat_start_oof"],
        errors="coerce",
    ).to_numpy(np.float64)
    match_id = frame["match_id"].astype("string").to_numpy()

    valid = np.isfinite(outcome) & np.isfinite(probability)
    if not valid.all():
        raise RuntimeError(
            f"{path} contains {(~valid).sum():,} rows with missing or nonfinite values."
        )
    if not np.isin(outcome, [0.0, 1.0]).all():
        raise RuntimeError(f"{path} contains nonbinary target values.")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise RuntimeError(f"{path} contains probabilities outside [0, 1].")

    return outcome.astype(np.int8), probability, match_id


def make_h3_plot(bins: pd.DataFrame, output_dir: Path) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    max_value = float(
        max(
            bins["mean_predicted"].max(),
            bins["observed_rate"].max(),
            0.10,
        )
    )
    axis_max = min(1.0, max_value * 1.08)
    axis.plot(
        [0.0, axis_max],
        [0.0, axis_max],
        linestyle=(0, (4, 3)),
        color="#A8A8A8",
        label="Ideal calibration",
    )
    axis.plot(
        bins["mean_predicted"],
        bins["observed_rate"],
        marker="o",
        color="#315F8C",
        label="H=3 PCTV",
    )
    axis.set_xlim(0.0, axis_max)
    axis.set_ylim(0.0, axis_max)
    axis.set_xlabel("Mean predicted shot-creation probability")
    axis.set_ylabel("Observed shot-creation rate")
    axis.legend(frameon=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_dir / "pctv_h3_reliability.png", dpi=600)
    figure.savefig(output_dir / "pctv_h3_reliability.pdf")
    figure.savefig(output_dir / "pctv_h3_reliability.svg")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    paths = {1: args.h1.resolve(), 3: args.h3.resolve(), 5: args.h5.resolve()}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, args.overwrite)

    summary_rows: list[dict[str, float | int]] = []
    reliability_parts: list[pd.DataFrame] = []

    for horizon, path in paths.items():
        print(f"[load] H={horizon}: {path}")
        outcome, probability, match_id = load_horizon(path, horizon)
        intercept, slope = calibration_intercept_slope(outcome, probability)
        ece, maximum_gap = fixed_width_ece(
            outcome,
            probability,
            n_bins=args.ece_bins,
        )
        bins = equal_frequency_bins(
            outcome,
            probability,
            horizon=horizon,
            n_bins=args.reliability_bins,
        )
        reliability_parts.append(bins)
        summary_rows.append(
            {
                "horizon": horizon,
                "passes": int(len(outcome)),
                "matches": int(pd.Series(match_id).nunique()),
                "observed_positive_rate": float(outcome.mean()),
                "mean_predicted_probability": float(probability.mean()),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                f"ece_{args.ece_bins}_fixed_width_bins": ece,
                f"maximum_gap_{args.ece_bins}_fixed_width_bins": maximum_gap,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("horizon")
    reliability = pd.concat(reliability_parts, ignore_index=True)
    summary.to_csv(
        output_dir / "pctv_calibration_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reliability.to_csv(
        output_dir / "pctv_reliability_bins.csv",
        index=False,
        encoding="utf-8-sig",
    )
    make_h3_plot(
        reliability.loc[reliability["horizon"].eq(3)].copy(),
        output_dir,
    )

    audit = {
        "status": "PASS",
        "inputs": {
            f"h{horizon}": {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for horizon, path in paths.items()
        },
        "ece_bins": args.ece_bins,
        "reliability_bins": args.reliability_bins,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (output_dir / "pctv_calibration_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    print("\nPCTV calibration summary")
    print(summary.to_string(index=False))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
