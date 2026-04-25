# -*- coding: utf-8 -*-
"""
Generate out-of-fold xPass probabilities for pass events.

This script produces leakage-controlled xPass probabilities by holding out
entire matches in each fold. Each fold fits preprocessing parameters only on
its training matches, trains an SGD logistic model, calibrates the model on an
inner calibration split, and predicts the held-out matches.

Expected input: a filtered StatsBomb-style pass-event CSV file or a raw event
CSV file. If raw events are supplied, the script applies the same open-play pass
filtering rules used in the final xPass training script.

Main outputs:
- passes_with_xpass_oof.csv
- xpass_oof_predictions.csv
- xpass_oof_metrics.csv
"""

import argparse
import ast
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# ----------------------------- CLI -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate match-level out-of-fold xPass probabilities."
    )
    parser.add_argument("--input_csv", required=True, help="Input pass-event CSV file.")
    parser.add_argument("--output_dir", required=True, help="Directory for OOF predictions and metrics.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of match-level OOF folds.")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=96000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--alpha", type=float, default=3e-6)
    parser.add_argument("--l1_ratio", type=float, default=0.25)
    parser.add_argument("--penalty", default="elasticnet")
    parser.add_argument("--lr_strategy", choices=["adaptive", "invscaling"], default="adaptive")
    parser.add_argument("--power_t", type=float, default=0.5)
    parser.add_argument("--calibration_size", type=float, default=0.20)
    parser.add_argument("--disable_dir_flags", action="store_true")
    parser.add_argument("--disable_region_bins", action="store_true")
    parser.add_argument("--disable_interactions", action="store_true")
    return parser.parse_args()


# ----------------------------- Utility functions -----------------------------
def parse_xy(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    xs, ys = [], []
    for v in series:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            xs.append(float(v[0]))
            ys.append(float(v[1]))
        elif isinstance(v, str) and v.strip().startswith("["):
            try:
                a = ast.literal_eval(v)
                xs.append(float(a[0]))
                ys.append(float(a[1]))
            except Exception:
                xs.append(np.nan)
                ys.append(np.nan)
        else:
            xs.append(np.nan)
            ys.append(np.nan)
    return pd.Series(xs, index=series.index), pd.Series(ys, index=series.index)


def to_bool01(series: pd.Series) -> pd.Series:
    if series is None:
        return None
    s = pd.Series(series).copy()
    if s.dtype == bool:
        return s.astype(int)
    s = s.astype(str).str.strip().str.lower()
    return s.isin(["true", "1", "yes", "y", "t"]).astype(int)


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def safe_logloss(y_true, p):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return log_loss(y_true, p)


def safe_make_bins(values: pd.Series, q_list, min_bins: int = 3, eps: float = 1e-6):
    v = pd.to_numeric(values, errors="coerce")
    v = v[np.isfinite(v)]
    if v.empty:
        return np.array([0.0, 1.0], dtype=float)
    qs = np.unique(np.quantile(v, q_list))
    if qs.size < min_bins:
        vmin, vmax = float(v.min()), float(v.max())
        if vmin == vmax:
            vmin -= eps
            vmax += eps
        qs = np.linspace(vmin, vmax, num=max(min_bins, 2))
    qs = qs.astype(float)
    qs[0] -= eps
    qs[-1] += eps
    for i in range(1, len(qs)):
        if qs[i] <= qs[i - 1]:
            qs[i] = qs[i - 1] + eps
    return qs


def safe_cut(s: pd.Series, bins: np.ndarray, right: bool = True):
    out = pd.cut(s, bins=bins, labels=False, right=right, include_lowest=True)
    return out.astype("Int64").astype("string")


def build_group_key(df: pd.DataFrame) -> pd.Series:
    if "match_id" in df.columns:
        s = df["match_id"].astype("string").str.strip()
        if s.nunique(dropna=True) >= 2:
            return s.fillna("UNK")
    candidate_groups = [
        ["home_team", "away_team", "match_date", "season", "competition"],
        ["home_team", "away_team", "kickoff", "season", "competition"],
        ["home_team", "away_team", "date", "season", "competition"],
        ["home_team", "away_team", "kick_off"],
    ]
    for cols in candidate_groups:
        have = [c for c in cols if c in df.columns]
        if len(have) >= 2:
            key = df[have].astype("string").fillna("UNK").agg("|".join, axis=1)
            if key.nunique(dropna=True) >= 2:
                return key
    target_groups = int(np.clip(len(df) // 200_000, 10, 200))
    bucket_size = max(1, len(df) // target_groups)
    return (pd.Series(np.arange(len(df)), index=df.index) // bucket_size).astype("Int64").astype("string")


def filter_open_play_passes(df: pd.DataFrame) -> pd.DataFrame:
    if "type" in df.columns:
        df = df[_norm(df["type"]) == "pass"].copy()

    ban_play_patterns = {
        "from throw-in", "from free kick", "from corner", "from goal kick",
        "from kick off", "from keeper", "from penalty", "throw-in", "throw in",
    }
    ban_pass_types = {
        "throw-in", "throw in", "free kick", "corner", "goal kick", "kick off",
        "dropped ball", "drop ball", "recovery", "recovered ball",
    }
    if "play_pattern" in df.columns:
        df = df[~_norm(df["play_pattern"]).isin(ban_play_patterns)].copy()
    if "pass_type" in df.columns:
        df = df[~_norm(df["pass_type"]).isin(ban_pass_types)].copy()
    return df.reset_index(drop=True)


def add_base_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = df.copy()
    if "pass_outcome" in df.columns:
        y = df["pass_outcome"].isna().astype(int)
    elif "outcome" in df.columns:
        y = pd.to_numeric(df["outcome"], errors="coerce").fillna(0).astype(int)
    else:
        raise ValueError("No pass_outcome/outcome column found for pass-completion labels.")

    if not all(c in df.columns for c in ["pass_length", "pass_angle"]):
        if all(c in df.columns for c in ["start_x", "start_y", "end_x", "end_y"]):
            sx, sy, ex, ey = df["start_x"], df["start_y"], df["end_x"], df["end_y"]
        else:
            sx, sy = parse_xy(df.get("location", pd.Series([None] * len(df), index=df.index)))
            ex, ey = parse_xy(df.get("pass_end_location", pd.Series([None] * len(df), index=df.index)))
        df["pass_length"] = np.hypot(ex - sx, ey - sy)
        df["pass_angle"] = np.arctan2(ey - sy, ex - sx)

    if "start_x" in df.columns and "start_y" in df.columns:
        df["x_ratio"] = pd.to_numeric(df["start_x"], errors="coerce") / 120.0
        df["y_ratio"] = pd.to_numeric(df["start_y"], errors="coerce") / 80.0
    else:
        sx, sy = parse_xy(df.get("location", pd.Series([None] * len(df), index=df.index)))
        df["x_ratio"] = sx / 120.0
        df["y_ratio"] = sy / 80.0

    for c in ["minute", "second"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["dx"] = np.cos(df["pass_angle"]) * df["pass_length"]
    df["dy"] = np.sin(df["pass_angle"]) * df["pass_length"]
    df["abs_angle"] = np.abs(df["pass_angle"])
    df["len_cos"] = df["pass_length"] * np.cos(df["pass_angle"])
    df["len_sin"] = df["pass_length"] * np.sin(df["pass_angle"])
    df["time_s"] = (df["minute"].fillna(0) * 60 + df["second"].fillna(0)).astype(float)
    df["is_first_half"] = (df["minute"].fillna(0) < 45).astype(int)
    T = 45 * 60.0
    df["t_sin"] = np.sin(2 * np.pi * df["time_s"] / T)
    df["t_cos"] = np.cos(2 * np.pi * df["time_s"] / T)

    flag_candidates = [
        "under_pressure", "pass_cross", "pass_cut_back", "pass_through_ball", "pass_switch",
        "pass_aerial_won", "pass_miscommunication", "pass_outswinging", "pass_inswinging",
        "is_first_half",
    ]
    for c in [c for c in flag_candidates if c in df.columns]:
        df[c] = to_bool01(df[c])

    df = df.replace([np.inf, -np.inf], np.nan)
    group_key = build_group_key(df)
    mask_group = group_key.notna()
    df = df.loc[mask_group].copy()
    y = y.loc[df.index]
    group_key = group_key.loc[df.index]

    num_base = [c for c in [
        "pass_length", "pass_angle", "minute", "second", "x_ratio", "y_ratio",
        "dx", "dy", "abs_angle", "len_cos", "len_sin", "time_s", "t_sin", "t_cos",
    ] if c in df.columns]
    flag_base = [c for c in flag_candidates if c in df.columns]
    need_num = num_base + flag_base
    mask_any = pd.DataFrame({c: df[c].notna() for c in need_num}).any(axis=1) if need_num else pd.Series(True, index=df.index)
    df = df.loc[mask_any].copy().reset_index(drop=True)
    y = y.loc[mask_any].reset_index(drop=True)
    group_key = group_key.loc[mask_any].reset_index(drop=True)
    return df, y, group_key


def add_fold_features(
    train_df: pd.DataFrame,
    target_dfs: List[pd.DataFrame],
    use_dir_flags: bool,
    use_region_bins: bool,
    use_interactions: bool,
):
    train_df = train_df.copy()
    target_dfs = [d.copy() for d in target_dfs]
    all_dfs = [train_df] + target_dfs

    if "pass_length" in train_df.columns:
        qs_len = safe_make_bins(train_df["pass_length"], np.linspace(0, 1, 11), min_bins=3)
        for d in all_dfs:
            d["len_bin"] = safe_cut(d["pass_length"], qs_len, right=True)

    if "abs_angle" in train_df.columns:
        angle_bins = np.array([0, 0.35, 0.7, np.pi], dtype=float)
        angle_bins[0] -= 1e-6
        angle_bins[-1] += 1e-6
        for d in all_dfs:
            d["abs_angle_bin"] = safe_cut(d["abs_angle"], angle_bins, right=False)

    if use_region_bins and all(c in train_df.columns for c in ["x_ratio", "y_ratio"]):
        qx = safe_make_bins(train_df["x_ratio"], [0.0, 0.33, 0.66, 1.0], min_bins=4)
        qy = safe_make_bins(train_df["y_ratio"], [0.0, 0.5, 1.0], min_bins=3)
        for d in all_dfs:
            d["x_bin"] = safe_cut(d["x_ratio"], qx, right=True)
            d["y_bin"] = safe_cut(d["y_ratio"], qy, right=True)

    for d in all_dfs:
        if "len_bin" in d.columns and "abs_angle_bin" in d.columns:
            d["lenXangle"] = d["len_bin"].astype("string") + "|" + d["abs_angle_bin"].astype("string")

    num_cols = [c for c in [
        "pass_length", "pass_angle", "minute", "second", "x_ratio", "y_ratio",
        "dx", "dy", "abs_angle", "len_cos", "len_sin", "time_s", "t_sin", "t_cos",
    ] if c in train_df.columns]
    flag_cols = [c for c in [
        "under_pressure", "pass_cross", "pass_cut_back", "pass_through_ball", "pass_switch",
        "pass_aerial_won", "pass_miscommunication", "pass_outswinging", "pass_inswinging",
        "is_first_half",
    ] if c in train_df.columns]
    cat_cols = [c for c in [
        "pass_height", "pass_type", "pass_technique", "pass_body_part", "play_pattern",
        "team", "position", "possession_team", "len_bin", "abs_angle_bin", "lenXangle",
        "x_bin", "y_bin",
    ] if c in train_df.columns]

    if use_dir_flags and all(c in train_df.columns for c in ["len_cos", "abs_angle"]):
        for d in all_dfs:
            d["dir_forward"] = (d["len_cos"] > 0).astype(int)
            d["dir_vertical"] = (d["abs_angle"] < np.deg2rad(15)).astype(int)
        flag_cols += ["dir_forward", "dir_vertical"]

    if use_interactions and "under_pressure" in flag_cols:
        for d in all_dfs:
            d["len_underP"] = d["pass_length"] * d["under_pressure"].fillna(0)
            d["cos_underP"] = d["len_cos"] * d["under_pressure"].fillna(0)
        num_cols += ["len_underP", "cos_underP"]

    for c in cat_cols:
        for d in all_dfs:
            d[c] = d[c].astype("string").fillna("UNK")

    return all_dfs[0], all_dfs[1:], num_cols, flag_cols, cat_cols


def make_preprocessor(num_cols, flag_cols, cat_cols):
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))]), num_cols),
            ("flag", "passthrough", flag_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def train_sgd_classifier(Xtr, ytr, args):
    classes = np.array([0, 1], dtype=int)
    class_w = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
    w0, w1 = float(class_w[0]), float(class_w[1])
    clf = SGDClassifier(
        loss="log_loss",
        penalty=args.penalty,
        l1_ratio=args.l1_ratio,
        alpha=args.alpha,
        learning_rate=args.lr_strategy,
        eta0=args.lr,
        power_t=args.power_t,
        average=True,
        random_state=args.random_state,
        max_iter=1,
        warm_start=True,
    )
    n_train = Xtr.shape[0]
    indices = np.arange(n_train)
    for epoch in range(1, args.epochs + 1):
        rng = np.random.default_rng(args.random_state + epoch)
        rng.shuffle(indices)
        first_batch = True
        for start in range(0, n_train, args.batch_size):
            end = min(start + args.batch_size, n_train)
            ids = indices[start:end]
            yb = ytr[ids]
            sw = np.where(yb == 0, w0, w1).astype(float)
            if first_batch:
                clf.partial_fit(Xtr[ids], yb, classes=classes, sample_weight=sw)
                first_batch = False
            else:
                clf.partial_fit(Xtr[ids], yb, sample_weight=sw)
    return clf


def split_train_calibration(train_idx: np.ndarray, y: pd.Series, groups: pd.Series, args):
    train_groups = groups.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    local = np.arange(len(train_idx))
    if train_groups.nunique(dropna=True) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.calibration_size, random_state=args.random_state + 17)
        core_local, cal_local = next(splitter.split(local.reshape(-1, 1), y_train, groups=train_groups))
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=args.calibration_size, random_state=args.random_state + 17)
        core_local, cal_local = next(splitter.split(local.reshape(-1, 1), y_train))
    return train_idx[core_local], train_idx[cal_local]


def main():
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    read_dtypes = {"match_id": "string", "minute": "float64", "second": "float64"}
    df_raw = pd.read_csv(input_csv, low_memory=False, dtype=read_dtypes)
    print(f"[load] rows={len(df_raw)}, cols={df_raw.shape[1]}")

    df_filtered = filter_open_play_passes(df_raw)
    df, y, groups = add_base_features(df_filtered)
    print(f"[sample] rows after filtering and feature checks={len(df)}, groups={groups.nunique()}")

    n_groups = groups.nunique(dropna=True)
    n_splits = int(min(max(2, args.n_splits), n_groups))
    gkf = GroupKFold(n_splits=n_splits)

    pred_uncal = np.full(len(df), np.nan, dtype=float)
    pred_calib = np.full(len(df), np.nan, dtype=float)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, y, groups=groups), start=1):
        print(f"[fold {fold}/{n_splits}] train={len(train_idx)}, test={len(test_idx)}")
        core_idx, cal_idx = split_train_calibration(train_idx, y, groups, args)

        train_core = df.iloc[core_idx].copy()
        cal_df = df.iloc[cal_idx].copy()
        test_df = df.iloc[test_idx].copy()

        train_core, transformed, num_cols, flag_cols, cat_cols = add_fold_features(
            train_core,
            [cal_df, test_df],
            use_dir_flags=not args.disable_dir_flags,
            use_region_bins=not args.disable_region_bins,
            use_interactions=not args.disable_interactions,
        )
        cal_df, test_df = transformed
        feat_cols = num_cols + flag_cols + cat_cols

        prep = make_preprocessor(num_cols, flag_cols, cat_cols)
        prep.fit(train_core[feat_cols])
        X_core = prep.transform(train_core[feat_cols])
        X_cal = prep.transform(cal_df[feat_cols])
        X_test = prep.transform(test_df[feat_cols])

        y_core = y.iloc[core_idx].to_numpy(dtype=int)
        y_cal = y.iloc[cal_idx].to_numpy(dtype=int)
        y_test = y.iloc[test_idx].to_numpy(dtype=int)

        clf = train_sgd_classifier(X_core, y_core, args)
        p_uncal = clf.predict_proba(X_test)[:, 1]

        if len(np.unique(y_cal)) > 1:
            calibrator = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
            calibrator.fit(X_cal, y_cal)
            p_cal = calibrator.predict_proba(X_test)[:, 1]
        else:
            print(f"[fold {fold}] calibration split has one class; using uncalibrated probabilities.")
            p_cal = p_uncal

        pred_uncal[test_idx] = p_uncal
        pred_calib[test_idx] = p_cal

        fold_auc = roc_auc_score(y_test, p_cal) if len(np.unique(y_test)) > 1 else np.nan
        fold_ll = safe_logloss(y_test, p_cal) if len(np.unique(y_test)) > 1 else np.nan
        fold_brier = brier_score_loss(y_test, p_cal)
        fold_acc = accuracy_score(y_test, (p_cal >= 0.5).astype(int))
        fold_rows.append({
            "fold": fold,
            "n_train_core": len(core_idx),
            "n_calibration": len(cal_idx),
            "n_test": len(test_idx),
            "AUC_oof_calib": fold_auc,
            "LogLoss_oof_calib": fold_ll,
            "Brier_oof_calib": fold_brier,
            "ACC_oof_calib@0.5": fold_acc,
        })

    if np.isnan(pred_calib).any():
        raise RuntimeError("Some rows did not receive OOF predictions.")

    df_out = df.copy()
    df_out["y_true"] = y.to_numpy(dtype=int)
    df_out["xpass_pred_uncal_oof"] = np.clip(pred_uncal, 0.001, 0.999)
    df_out["xpass_pred_calib_oof"] = np.clip(pred_calib, 0.001, 0.999)
    # Alternative column names for compatibility with downstream scripts.
    df_out["xpass_oof_uncal"] = df_out["xpass_pred_uncal_oof"]
    df_out["xpass_oof_calib"] = df_out["xpass_pred_calib_oof"]

    full_path = output_dir / "passes_with_xpass_oof.csv"
    pred_path = output_dir / "xpass_oof_predictions.csv"
    metrics_path = output_dir / "xpass_oof_metrics.csv"
    fold_path = output_dir / "xpass_oof_fold_metrics.csv"

    df_out.to_csv(full_path, index=False, encoding="utf-8-sig")

    pred_cols = [c for c in ["match_id", "team", "player", "minute", "second", "position", "pass_type", "pass_height"] if c in df_out.columns]
    pred_export = df_out[pred_cols].copy()
    pred_export["y_true"] = y.to_numpy(dtype=int)
    pred_export["xpass_pred_uncal_oof"] = df_out["xpass_pred_uncal_oof"]
    pred_export["xpass_pred_calib_oof"] = df_out["xpass_pred_calib_oof"]
    pred_export.to_csv(pred_path, index=False, encoding="utf-8-sig")

    metrics = {
        "n_events": len(df_out),
        "n_groups": int(n_groups),
        "n_splits": int(n_splits),
        "AUC_oof_uncal": roc_auc_score(y, pred_uncal) if len(np.unique(y)) > 1 else np.nan,
        "AUC_oof_calib": roc_auc_score(y, pred_calib) if len(np.unique(y)) > 1 else np.nan,
        "LogLoss_oof_calib": safe_logloss(y, pred_calib),
        "Brier_oof_calib": brier_score_loss(y, pred_calib),
        "ACC_oof_calib@0.5": accuracy_score(y, (pred_calib >= 0.5).astype(int)),
    }
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False, encoding="utf-8-sig")

    print("[done] OOF xPass prediction generation completed.")
    print(f"[output] {full_path}")
    print(f"[output] {pred_path}")
    print(f"[output] {metrics_path}")
    print(f"[output] {fold_path}")


if __name__ == "__main__":
    main()
