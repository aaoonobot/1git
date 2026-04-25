# -*- coding: utf-8 -*-
"""
xPass model training and probability calibration.

This script trains an event-level pass-completion model, evaluates
calibration and discrimination performance, and exports calibrated and
uncalibrated xPass probabilities for downstream risk-reward analysis.

Expected input: a StatsBomb-style pass-event CSV file.
Main outputs: model files, validation curves, test-set metrics, and a
filtered pass-event table with xPass probability columns appended.
"""

import os
import argparse
import ast
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight

# ===================== Configuration =====================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an xPass model and export calibrated pass-completion probabilities."
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to the pass-event CSV file used for xPass training."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for model files, predictions, metrics, and plots."
    )
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--val_size", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=96000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_epochs_before_es", type=int, default=10)
    parser.add_argument("--penalty", default="elasticnet")
    parser.add_argument("--l1_ratio", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=3e-6)
    parser.add_argument("--finetune_epochs", type=int, default=1)
    parser.add_argument("--finetune_lr", type=float, default=5e-5)
    parser.add_argument("--finetune_l1_ratio", type=float, default=0.10)
    parser.add_argument("--use_logreg_saga", action="store_true")
    parser.add_argument("--lr_strategy", choices=["adaptive", "invscaling"], default="adaptive")
    parser.add_argument("--power_t", type=float, default=0.5)
    parser.add_argument("--disable_dir_flags", action="store_true")
    parser.add_argument("--disable_region_bins", action="store_true")
    parser.add_argument("--disable_interactions", action="store_true")
    return parser.parse_args()

args = parse_args()
CSV_PATH = Path(args.input_csv).expanduser()
RUN_DIR = Path(args.output_dir).expanduser()
RUN_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = RUN_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_UNCAL_PATH   = RUN_DIR / "xpass_pipeline_auc.pkl"
MODEL_CALIB_PATH   = RUN_DIR / "xpass_pipeline_calibrated.pkl"
BEST_SNAPSHOT_PATH = RUN_DIR / "xpass_best_snapshot.pkl"
TEST_METRICS_CSV   = RUN_DIR / "xpass_test_metrics.csv"
TEST_PRED_CSV      = RUN_DIR / "xpass_test_predictions.csv"
VAL_CURVE_CSV      = RUN_DIR / "xpass_val_curve.csv"
VAL_HOLD_THR_TXT   = RUN_DIR / "xpass_best_threshold.txt"
FILTERED_CSV_PATH  = RUN_DIR / "passes_filtered.csv"

RANDOM_STATE = args.random_state
TEST_SIZE = args.test_size
VAL_SIZE  = args.val_size

EPOCHS     = args.epochs
BATCH_SIZE = args.batch_size
LR         = args.lr
PATIENCE   = args.patience
MIN_EPOCHS_BEFORE_ES = args.min_epochs_before_es
PENALTY = args.penalty
L1_RATIO = args.l1_ratio
ALPHA = args.alpha

FINETUNE_EPOCHS = args.finetune_epochs
FINETUNE_LR     = args.finetune_lr
FINETUNE_L1R    = args.finetune_l1_ratio

USE_LOGREG_SAGA = args.use_logreg_saga
LR_STRATEGY     = args.lr_strategy
POWER_T         = args.power_t

USE_DIR_FLAGS    = not args.disable_dir_flags
USE_REGION_BINS  = not args.disable_region_bins
USE_INTERACTIONS = not args.disable_interactions

# ===================== Utility functions =====================
def parse_xy(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    xs, ys = [], []
    for v in series:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            xs.append(float(v[0])); ys.append(float(v[1]))
        elif isinstance(v, str) and v.strip().startswith('['):
            try:
                a = ast.literal_eval(v)
                xs.append(float(a[0])); ys.append(float(a[1]))
            except Exception:
                xs.append(np.nan); ys.append(np.nan)
        else:
            xs.append(np.nan); ys.append(np.nan)
    return pd.Series(xs), pd.Series(ys)

def safe_logloss(y_true, p):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return log_loss(y_true, p)

def per_match_metrics(match_ids: np.ndarray, y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    dfm = pd.DataFrame({"match_id": match_ids, "y": y, "p": p})
    out = []
    for mid, g in dfm.groupby("match_id"):
        auc = roc_auc_score(g["y"], g["p"]) if g["y"].nunique() > 1 else np.nan
        acc = ((g["p"] >= 0.5).astype(int) == g["y"]).mean()
        out.append({"match_id": mid, "AUC": auc, "ACC": acc, "n": len(g)})
    return pd.DataFrame(out)

def to_bool01(series: pd.Series) -> pd.Series:
    if series is None:
        return None
    s = series.copy()
    if s.dtype == bool:
        return s.astype(int)
    s = s.astype(str).str.strip().str.lower()
    return s.isin(["true","1","yes","y","t"]).astype(int)

def _find_datetime_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["match_date", "date", "kickoff", "kick_off"]
    for c in candidates:
        if c in df.columns:
            try:
                _ = pd.to_datetime(df[c], errors="coerce")
                if _.notna().sum() >= max(10, 0.0001*len(df)):
                    return c
            except Exception:
                pass
    return None

def time_aware_group_split(df: pd.DataFrame,
                           groups: pd.Series,
                           y: pd.Series,
                           val_size: float,
                           random_state: int = 42):
    date_col = _find_datetime_col(df)
    if date_col is not None:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        grp_time = dt.groupby(groups).min()
        valid_groups = grp_time.dropna().index
        if len(valid_groups) >= 5:
            grp_order = grp_time.sort_values()
            n_val = max(1, int(np.ceil(len(grp_order) * val_size)))
            val_groups = set(grp_order.index[-n_val:])
            val_mask = groups.isin(val_groups)
            train_idx = np.where(~val_mask.values)[0]
            val_idx   = np.where(val_mask.values)[0]
            if len(train_idx) > 0 and len(val_idx) > 0:
                print(f"[split] time-aware group split: date_col={date_col}, groups={len(grp_order)}, validation_groups={n_val}")
                return train_idx, val_idx
    print("[split] falling back to GroupShuffleSplit.")
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    tr_idx, va_idx = next(gss.split(df, y, groups=groups))
    return tr_idx, va_idx

def smooth_series(xs, w=3):
    s = pd.Series(xs, dtype=float)
    return s.rolling(w, min_periods=1, center=True).mean().tolist()

def ema_last(seq, alpha=0.3):
    m = None
    for v in seq:
        m = v if m is None else (1-alpha)*m + alpha*v
    return m if m is not None else None

# ===================== Load data =====================
read_dtypes = {"match_id": "string", "minute": "float64", "second": "float64"}
df = pd.read_csv(CSV_PATH, low_memory=False, dtype=read_dtypes)
print(f"[load] rows={len(df)}, cols={df.shape[1]}")

# ===== Select pass events and remove set-piece or restart contexts =====
def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

# Keep pass events when an event-type column is available.
if "type" in df.columns:
    before = len(df)
    df = df[_norm(df["type"]) == "pass"].copy()
    print(f"[filter] type == Pass: {before} -> {len(df)}")

# Remove set-piece and restart situations when the corresponding fields exist.
ban_play_patterns = {
    "from throw-in","from free kick","from corner","from goal kick",
    "from kick off","from keeper","from penalty","throw-in","throw in"
}
ban_pass_types = {
    "throw-in","throw in","free kick","corner","goal kick","kick off",
    "dropped ball","drop ball","recovery","recovered ball"
}

if "play_pattern" in df.columns:
    before = len(df)
    df = df[~_norm(df["play_pattern"]).isin(ban_play_patterns)].copy()
    print(f"[filter] play_pattern exclusions: {before} -> {len(df)}")

if "pass_type" in df.columns:
    before = len(df)
    df = df[~_norm(df["pass_type"]).isin(ban_pass_types)].copy()
    print(f"[filter] pass_type exclusions: {before} -> {len(df)}")

# Save the filtered event table used by downstream steps.
df.to_csv(FILTERED_CSV_PATH, index=False, encoding="utf-8-sig")
print(f"[output] filtered CSV: {FILTERED_CSV_PATH}")
df = df.reset_index(drop=True)

# ===================== xPass target label =====================
if "pass_outcome" in df.columns:
    y_full = df["pass_outcome"].isna().astype(int)   # StatsBomb convention: missing pass_outcome indicates completion.
elif "outcome" in df.columns:
    y_full = pd.to_numeric(df["outcome"], errors="coerce").fillna(0).astype(int)
else:
    raise ValueError("Missing pass_outcome or outcome column for the xPass target label.")

# ===================== Match grouping key =====================
def build_group_key(df: pd.DataFrame) -> pd.Series:
    if "match_id" in df.columns:
        s = df["match_id"].astype("string").str.strip()
        nunique = s.nunique(dropna=True)
        nonnull = s.notna().sum()
        print(f"[group_key] match_id non-missing={nonnull}, unique={nunique}")
        if nunique >= max(2, int(0.0001 * len(df))):
            return s
        s_num = s.str.extract(r"(\d+)", expand=False).pipe(pd.to_numeric, errors="coerce")
        if s_num.notna().mean() >= 0.5:
            return s_num.astype("Int64").astype("string")

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
            nunique = key.nunique(dropna=True)
            print(f"[group_key] composite fields {have}, unique={nunique}")
            if nunique >= max(2, int(0.0001 * len(df))):
                return key

    print("[group_key] no match field found; using row-based adaptive buckets.")
    target_groups = int(np.clip(len(df) // 200_000, 10, 200))
    bucket_size = max(1, len(df) // target_groups)
    # Preserve df.index to avoid alignment errors.
    buckets = (pd.Series(np.arange(len(df)), index=df.index) // bucket_size).astype("Int64").astype("string")
    return buckets

group_key = build_group_key(df)
mask_group = group_key.notna()
if (~mask_group).any():
    print(f"[group_key] removed rows with missing group key: {(~mask_group).sum()}")
df = df.loc[mask_group].copy()
y_full = y_full.loc[df.index]
group_key = group_key.loc[df.index]

# ===================== Geometry, location, and time features =====================
if not all(c in df.columns for c in ["pass_length", "pass_angle"]):
    if all(c in df.columns for c in ["start_x","start_y","end_x","end_y"]):
        sx, sy, ex, ey = (df["start_x"], df["start_y"], df["end_x"], df["end_y"])
    else:
        sx, sy = parse_xy(df.get("location", pd.Series([None]*len(df))))
        ex, ey = parse_xy(df.get("pass_end_location", pd.Series([None]*len(df))))
    df.loc[:, "pass_length"] = np.hypot(ex - sx, ey - sy)
    df.loc[:, "pass_angle"]  = np.arctan2(ey - sy, ex - sx)

if "start_x" in df.columns and "start_y" in df.columns:
    df.loc[:, "x_ratio"] = df["start_x"] / 120.0
    df.loc[:, "y_ratio"] = df["start_y"] / 80.0
else:
    df.loc[:, "x_ratio"] = np.nan
    df.loc[:, "y_ratio"] = np.nan

for c in ["minute","second"]:
    if c not in df.columns: df[c] = np.nan
    df.loc[:, c] = pd.to_numeric(df[c], errors="coerce")

if all(c in df.columns for c in ["pass_length","pass_angle"]):
    df.loc[:, "dx"] = np.cos(df["pass_angle"]) * df["pass_length"]
    df.loc[:, "dy"] = np.sin(df["pass_angle"]) * df["pass_length"]
    df.loc[:, "abs_angle"] = np.abs(df["pass_angle"])
    df.loc[:, "len_cos"] = df["pass_length"] * np.cos(df["pass_angle"])
    df.loc[:, "len_sin"] = df["pass_length"] * np.sin(df["pass_angle"])

df.loc[:, "time_s"] = (df["minute"].fillna(0)*60 + df["second"].fillna(0)).astype(float)
df.loc[:, "is_first_half"] = (df["minute"].fillna(0) < 45).astype(int)
T = 45*60.0
df.loc[:, "t_sin"] = np.sin(2*np.pi*df["time_s"]/T)
df.loc[:, "t_cos"] = np.cos(2*np.pi*df["time_s"]/T)

# ===================== Numeric feature cleaning =====================
num_cols_base: List[str] = [c for c in [
    "pass_length","pass_angle","minute","second","x_ratio","y_ratio",
    "dx","dy","abs_angle","len_cos","len_sin","time_s","t_sin","t_cos"
] if c in df.columns]

flag_cols_base: List[str] = [c for c in [
    "under_pressure",
    "pass_cross","pass_cut_back","pass_through_ball","pass_switch",
    "pass_aerial_won","pass_miscommunication","pass_outswinging","pass_inswinging",
    "is_first_half"
] if c in df.columns]
for c in flag_cols_base:
    df.loc[:, c] = to_bool01(df[c])

df = df.replace([np.inf, -np.inf], np.nan)
need_num = (num_cols_base + flag_cols_base)
mask_any_num_present = pd.DataFrame({c: df[c].notna() for c in need_num}).any(axis=1) if need_num else pd.Series(True, index=df.index)
removed = (~mask_any_num_present).sum()
df = df.loc[mask_any_num_present].copy()
y_full = y_full.loc[df.index]
group_key = group_key.loc[df.index]
print(f"[clean] removed rows with all numeric features missing: {removed}; remaining={len(df)}")

# ===================== Group-aware data splitting =====================
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
dev_idx, test_idx = next(gss.split(df, y_full, groups=group_key))
dev_df  = df.iloc[dev_idx].copy()
test_df = df.iloc[test_idx].copy()
y_dev, y_test = y_full.iloc[dev_idx], y_full.iloc[test_idx]
g_dev, g_test = group_key.iloc[dev_idx], group_key.iloc[test_idx]

train_idx, val_idx = time_aware_group_split(dev_df, g_dev, y_dev, val_size=VAL_SIZE, random_state=RANDOM_STATE)
train_df = dev_df.iloc[train_idx].copy()
val_df   = dev_df.iloc[val_idx].copy()
y_train, y_val   = y_dev.iloc[train_idx], y_dev.iloc[val_idx]
g_train, g_val   = g_dev.iloc[train_idx], g_dev.iloc[val_idx]

# ===================== Binning based on training-set statistics =====================
# Pass length bins
if "pass_length" in train_df.columns:
    qs_len = train_df["pass_length"].quantile(np.linspace(0, 1, 11)).to_numpy()
    qs_len[0]  -= 1e-6; qs_len[-1] += 1e-6
    def _cut_len(s):
        return pd.cut(s, bins=qs_len, labels=False, right=True, include_lowest=True).astype("Int64").astype("string")
    train_df.loc[:, "len_bin"] = _cut_len(train_df["pass_length"])
    val_df.loc[:,   "len_bin"] = _cut_len(val_df["pass_length"])
    test_df.loc[:,  "len_bin"] = _cut_len(test_df["pass_length"])

# Angle bins with fixed boundaries
if "abs_angle" in train_df.columns:
    angle_bins = np.array([0, 0.35, 0.7, np.pi], dtype=float)
    angle_bins[0] -= 1e-6; angle_bins[-1] += 1e-6
    def _cut_ang(s):
        return pd.cut(s, bins=angle_bins, labels=False, right=False, include_lowest=True).astype("Int64").astype("string")
    train_df.loc[:, "abs_angle_bin"] = _cut_ang(train_df["abs_angle"])
    val_df.loc[:,   "abs_angle_bin"] = _cut_ang(val_df["abs_angle"])
    test_df.loc[:,  "abs_angle_bin"] = _cut_ang(test_df["abs_angle"])

# Robust quantile binning with strictly increasing boundaries
def safe_make_bins(values: pd.Series, q_list, min_bins: int = 3, eps: float = 1e-6):
    v = pd.to_numeric(values, errors="coerce")
    v = v[np.isfinite(v)]
    if v.empty:
        return np.array([0.0, 1.0], dtype=float)
    qs = np.unique(np.quantile(v, q_list))
    if qs.size < min_bins:
        vmin, vmax = float(v.min()), float(v.max())
        if vmin == vmax:
            vmin -= eps; vmax += eps
        qs = np.linspace(vmin, vmax, num=max(min_bins, 2))
    qs = qs.astype(float)
    qs[0] -= eps; qs[-1] += eps
    # Ensure strictly increasing boundaries.
    for i in range(1, len(qs)):
        if qs[i] <= qs[i-1]:
            qs[i] = qs[i-1] + eps
    return qs

def safe_cut(s: pd.Series, bins: np.ndarray, right: bool = True):
    out = pd.cut(s, bins=bins, labels=False, right=right, include_lowest=True)
    return out.astype("Int64").astype("string")

# Region bins based on training-set x/y quantiles
if USE_REGION_BINS and all(c in train_df.columns for c in ["x_ratio","y_ratio"]):
    qx = safe_make_bins(train_df["x_ratio"], [0.0, 0.33, 0.66, 1.0], min_bins=4)
    qy = safe_make_bins(train_df["y_ratio"], [0.0, 0.5, 1.0],       min_bins=3)
    for _df in (train_df, val_df, test_df):
        _df.loc[:, "x_bin"] = safe_cut(_df["x_ratio"], qx, right=True)
        _df.loc[:, "y_bin"] = safe_cut(_df["y_ratio"], qy, right=True)


# Crossed length-angle bins
for _df in (train_df, val_df, test_df):
    if "len_bin" in _df.columns and "abs_angle_bin" in _df.columns:
        _df.loc[:, "lenXangle"] = (_df["len_bin"].astype("string") + "|" + _df["abs_angle_bin"].astype("string")).astype("string")

# ===================== Feature set =====================
num_cols  = [c for c in num_cols_base if c in train_df.columns]
flag_cols = [c for c in flag_cols_base if c in train_df.columns]
cat_cols: List[str] = [c for c in [
    "pass_height","pass_type","pass_technique","pass_body_part","play_pattern",
    "team","position","possession_team","len_bin","abs_angle_bin","lenXangle",
    "x_bin","y_bin"
] if c in train_df.columns]

# Directional flags
if USE_DIR_FLAGS and all(c in train_df.columns for c in ["len_cos","abs_angle"]):
    for _df in (train_df, val_df, test_df):
        _df.loc[:, "dir_forward"]  = (_df["len_cos"] > 0).astype(int)
        _df.loc[:, "dir_vertical"] = (_df["abs_angle"] < np.deg2rad(15)).astype(int)
    for c in ["dir_forward","dir_vertical"]:
        if c not in flag_cols and c in train_df.columns:
            flag_cols.append(c)

# Interaction terms
if USE_INTERACTIONS and "under_pressure" in flag_cols:
    for _df in (train_df, val_df, test_df):
        _df.loc[:, "len_underP"] = _df["pass_length"] * _df["under_pressure"].fillna(0)
        _df.loc[:, "cos_underP"] = _df["len_cos"]     * _df["under_pressure"].fillna(0)
    for c in ["len_underP","cos_underP"]:
        if c in train_df.columns and c not in num_cols:
            num_cols.append(c)

# Cast categorical columns to strings.
for c in cat_cols:
    for _df in (train_df, val_df, test_df):
        _df.loc[:, c] = _df[c].astype("string").fillna("UNK")

# ===================== Preprocessing =====================
num_pipeline  = SkPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False)),
])
flag_pipeline = "passthrough"
cat_encoder   = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)

all_feat_cols = num_cols + flag_cols + cat_cols
preprocess = ColumnTransformer(
    transformers=[
        ("num",  num_pipeline,  num_cols),
        ("flag", flag_pipeline, flag_cols),
        ("cat",  cat_encoder,   cat_cols),
    ],
    remainder="drop",
    sparse_threshold=1.0
)
preprocess.fit(train_df[all_feat_cols])

Xtr = preprocess.transform(train_df[all_feat_cols])
Xva = preprocess.transform(val_df[all_feat_cols])
Xte = preprocess.transform(test_df[all_feat_cols])

# Class weights
classes = np.array([0, 1], dtype=int)
class_w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train.values)
w0, w1 = float(class_w[0]), float(class_w[1])
print(f"[class_weight] w0={w0:.3f}, w1={w1:.3f}")

# ===================== Estimator configuration =====================
if USE_LOGREG_SAGA:
    clf = LogisticRegression(
        solver="saga",
        penalty="elasticnet",
        l1_ratio=0.10,
        C=3.0,
        class_weight="balanced",
        max_iter=3000,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
else:
    clf = SGDClassifier(
        loss="log_loss",
        penalty=PENALTY,
        l1_ratio=L1_RATIO,
        alpha=ALPHA,
        learning_rate=LR_STRATEGY,   # "adaptive" or "invscaling"
        eta0=LR,
        power_t=POWER_T,
        average=True,                # ASGD
        random_state=RANDOM_STATE,
        max_iter=1,
        warm_start=True
    )

# ===================== Training loop with early stopping =====================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
(ax_auc, ax_ll, ax_acc) = axes
for ax in axes: ax.grid(True, alpha=0.3)
ax_auc.set_title("Validation AUC");      ax_auc.set_xlabel("Epoch"); ax_auc.set_ylabel("AUC")
ax_ll.set_title("Validation LogLoss");   ax_ll.set_xlabel("Epoch");  ax_ll.set_ylabel("LogLoss")
ax_acc.set_title("Validation ACC@0.5");  ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("ACC")

auc_hist, ll_hist, br_hist, acc_hist = [], [], [], []
best_score, best_epoch, no_improve = np.inf, -1, 0

if USE_LOGREG_SAGA:
    clf.fit(Xtr, y_train.values)
    joblib.dump(clf, BEST_SNAPSHOT_PATH)
    p_val = clf.predict_proba(Xva)[:, 1]
    auc_hist.append(roc_auc_score(y_val, p_val))
    ll_hist.append(safe_logloss(y_val, p_val))
    br_hist.append(brier_score_loss(y_val, p_val))
    acc_hist.append(accuracy_score(y_val, (p_val >= 0.5).astype(int)))
else:
    n_train = Xtr.shape[0]
    indices = np.arange(n_train)
    for epoch in range(1, EPOCHS + 1):
        rng = np.random.default_rng(RANDOM_STATE + epoch)
        rng.shuffle(indices)

        first_batch = True
        for start in range(0, n_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_train)
            batch_ids = indices[start:end]
            Xb = Xtr[batch_ids]
            yb = y_train.iloc[batch_ids].values
            sw = np.where(yb == 0, w0, w1).astype(float)
            if first_batch:
                clf.partial_fit(Xb, yb, classes=classes, sample_weight=sw)
                first_batch = False
            else:
                clf.partial_fit(Xb, yb, sample_weight=sw)

        p_val = clf.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(y_val, p_val) if len(np.unique(y_val)) > 1 else np.nan
        ll  = safe_logloss(y_val, p_val)
        br  = brier_score_loss(y_val, p_val)
        acc = accuracy_score(y_val, (p_val >= 0.5).astype(int))

        auc_hist.append(auc); ll_hist.append(ll); br_hist.append(br); acc_hist.append(acc)

        ll_s = ema_last(ll_hist, alpha=0.3)
        br_s = ema_last(br_hist, alpha=0.3)
        score = 0.5*ll_s + 0.5*br_s

        ax_auc.cla(); ax_ll.cla(); ax_acc.cla()
        for ax in axes: ax.grid(True, alpha=0.3)
        ax_auc.set_title("Validation AUC");      ax_auc.set_xlabel("Epoch"); ax_auc.set_ylabel("AUC")
        ax_ll.set_title("Validation LogLoss");   ax_ll.set_xlabel("Epoch");  ax_ll.set_ylabel("LogLoss")
        ax_acc.set_title("Validation ACC@0.5");  ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("ACC")
        ax_auc.plot(range(1, len(auc_hist)+1), smooth_series(auc_hist, 3), marker="o")
        ax_ll.plot(range(1, len(ll_hist)+1),   smooth_series(ll_hist, 3),  marker="o")
        ax_acc.plot(range(1, len(acc_hist)+1), smooth_series(acc_hist, 3), marker="o")
        plt.savefig(PLOTS_DIR / f"val_curves_epoch_{epoch:03d}.png", dpi=160)

        print(f"[Epoch {epoch:02d}] Val AUC={auc:.4f} | LogLoss={ll:.4f} | Brier={br:.4f} | ACC@0.5={acc:.4f} | score(EMA)={score:.6f}")

        if score < best_score - 1e-5:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            joblib.dump(clf, BEST_SNAPSHOT_PATH)
        else:
            no_improve += 1
            if epoch >= MIN_EPOCHS_BEFORE_ES and no_improve >= PATIENCE:
                print(f"Early stopping: no improvement for {PATIENCE} epochs; best_epoch={best_epoch}, EMA score={best_score:.6f}")
                break

# ============ Optional fine-tuning for SGD ============
if not USE_LOGREG_SAGA and FINETUNE_EPOCHS > 0:
    clf = joblib.load(BEST_SNAPSHOT_PATH)
    clf.eta0 = FINETUNE_LR
    clf.l1_ratio = FINETUNE_L1R
    print(f"[finetune] start: eta0={clf.eta0}, l1_ratio={clf.l1_ratio}, epochs={FINETUNE_EPOCHS}")
    n_train = Xtr.shape[0]
    indices = np.arange(n_train)
    for ep in range(FINETUNE_EPOCHS):
        rng = np.random.default_rng(RANDOM_STATE + 777 + ep)
        rng.shuffle(indices)
        first_batch = True
        for start in range(0, n_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_train)
            batch_ids = indices[start:end]
            Xb = Xtr[batch_ids]
            yb = y_train.iloc[batch_ids].values
            sw = np.where(yb == 0, w0, w1).astype(float)
            if first_batch:
                clf.partial_fit(Xb, yb, classes=classes, sample_weight=sw)
                first_batch = False
            else:
                clf.partial_fit(Xb, yb, sample_weight=sw)
    joblib.dump(clf, BEST_SNAPSHOT_PATH)

plt.savefig(PLOTS_DIR / "val_curves_final.png", dpi=180)

# ===================== Export uncalibrated and calibrated models =====================
best_clf = joblib.load(BEST_SNAPSHOT_PATH)

pipe_auc = Pipeline(steps=[("prep", preprocess), ("clf", best_clf)])
joblib.dump(pipe_auc, MODEL_UNCAL_PATH)
print(f"[output] uncalibrated model: {MODEL_UNCAL_PATH}")

# Split the validation set into calibration and holdout subsets.
unique_groups_val = pd.Series(g_val).dropna().unique()
if len(unique_groups_val) >= 2:
    cal_gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE+7)
    cal_idx, hold_idx = next(cal_gss.split(val_df, y_val, groups=g_val))
    print(f"[calibration] group-aware split; groups_in_val={len(unique_groups_val)}")
else:
    print("[calibration] validation set has one group; using random split.")
    rs = ShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE+7)
    cal_idx, hold_idx = next(rs.split(val_df, y_val))

Xva_mat = Xva
Xcal = Xva_mat[cal_idx];  ycal = y_val.iloc[cal_idx]
Xhold = Xva_mat[hold_idx]; yhold = y_val.iloc[hold_idx]

calibrator = CalibratedClassifierCV(best_clf, method="sigmoid", cv="prefit")
calibrator.fit(Xcal, ycal)
pipe_cal = Pipeline(steps=[("prep", preprocess), ("clf", calibrator)])
joblib.dump(pipe_cal, MODEL_CALIB_PATH)
print(f"[output] sigmoid-calibrated model: {MODEL_CALIB_PATH}")

# ===================== Threshold selection on validation holdout =====================
all_feat_cols = num_cols + flag_cols + cat_cols
p_hold = pipe_cal.predict_proba(val_df.iloc[hold_idx][all_feat_cols])[:, 1]
thr_grid = np.linspace(0.05, 0.95, 37)
best_thr = max(thr_grid, key=lambda t: accuracy_score(yhold, (p_hold >= t).astype(int)))
with open(VAL_HOLD_THR_TXT, "w", encoding="utf-8") as f:
    f.write(f"{best_thr:.4f}\n")
print(f"[threshold] best_thr={best_thr:.4f} based on validation-holdout accuracy")

# ===================== Test-set evaluation and outputs =====================
p_test_uncal = pipe_auc.predict_proba(test_df[all_feat_cols])[:, 1]
p_test_calib = pipe_cal.predict_proba(test_df[all_feat_cols])[:, 1]

metrics_event = {
    "AUC_test_uncal": roc_auc_score(y_test, p_test_uncal) if len(np.unique(y_test)) > 1 else np.nan,
    "AUC_test_calib": roc_auc_score(y_test, p_test_calib) if len(np.unique(y_test)) > 1 else np.nan,
    "LogLoss_test_calib": safe_logloss(y_test, p_test_calib),
    "Brier_test_calib": brier_score_loss(y_test, p_test_calib),
    "ACC_test@0.5_calib": accuracy_score(y_test, (p_test_calib >= 0.5).astype(int)),
    "ACC_test@best_thr_calib": accuracy_score(y_test, (p_test_calib >= best_thr).astype(int)),
    "best_thr_from_val": float(best_thr),
}
print("\n=== Test set: event-level metrics ===")
for k, v in metrics_event.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

per_match = per_match_metrics(g_test.values, y_test.values, p_test_calib)
metrics_match = {
    "ACC_mean_per_match_calib": per_match["ACC"].mean(),
    "AUC_mean_per_match_calib": per_match["AUC"].mean(skipna=True),
    "AUC_valid_matches": int(per_match["AUC"].notna().sum())
}
print("\n=== Test set: match-level metrics using calibrated probabilities ===")
for k, v in metrics_match.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

pd.DataFrame([{**metrics_event, **metrics_match}]).to_csv(TEST_METRICS_CSV, index=False, encoding="utf-8-sig")

cols_dump = ["match_id","team","player","minute","second","position",
             "pass_type","pass_height","pass_body_part","play_pattern"]
cols_dump = [c for c in cols_dump if c in test_df.columns]
test_dump = test_df[cols_dump].copy()
test_dump.insert(0, "group_key", g_test.values)
test_dump["y_true"] = y_test.values
test_dump["xpass_pred_uncal"] = p_test_uncal
test_dump["xpass_pred_calib"] = p_test_calib
test_dump.to_csv(TEST_PRED_CSV, index=False, encoding="utf-8-sig")

pd.DataFrame({
    "epoch": np.arange(1, len(ll_hist)+1),
    "val_auc": auc_hist,
    "val_logloss": ll_hist,
    "val_brier": br_hist,
    "val_acc@0.5": acc_hist
}).to_csv(VAL_CURVE_CSV, index=False, encoding="utf-8-sig")

print(f"\n[output] test metrics: {TEST_METRICS_CSV}")
print(f"[output] test predictions: {TEST_PRED_CSV}")
print(f"[output] validation curve: {VAL_CURVE_CSV}")
print(f"[output] uncalibrated model: {MODEL_UNCAL_PATH}")
print(f"[output] calibrated model: {MODEL_CALIB_PATH}")
print(f"[output] filtered sample: {FILTERED_CSV_PATH}")


print("[done] xPass model training and evaluation completed.")
