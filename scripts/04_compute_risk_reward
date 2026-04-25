# -*- coding: utf-8 -*-
"""
Risk-reward modeling for football pass events.

This script computes event-level risk, gain on success, failure cost,
passer marginal contribution, and expected net reward for pass events.
It is designed to be run after xPass probabilities have been generated.

Expected input: a StatsBomb-style pass-event CSV file with xPass
probability columns, preferably xpass_pred_calib and xpass_pred_uncal.
Main outputs: event-level risk-reward tables, continuation-value outputs,
passer marginal estimates, and diagnostic figures.
"""

import os
import argparse
import math
import ast
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, GroupKFold

# ======================== Configuration ========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute risk-reward metrics for pass events using precomputed xPass probabilities."
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to the pass-event CSV file containing xPass probability columns."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for risk-reward outputs and diagnostic figures."
    )
    parser.add_argument("--h_steps", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_size", type=float, default=0.20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--lambda_cv", type=float, default=0.30)
    parser.add_argument("--passer_weight", type=float, default=0.05)
    parser.add_argument("--passer_prior_k", type=float, default=50.0)
    return parser.parse_args()

args = parse_args()
CSV_PATH = str(Path(args.input_csv).expanduser())
RUN_DIR = str(Path(args.output_dir).expanduser())
os.makedirs(RUN_DIR, exist_ok=True)

H_STEPS = args.h_steps
GAMMA   = args.gamma
WINDOW  = args.window
EPOCHS  = args.epochs
LR      = args.lr
HIDDEN  = args.hidden
BATCH   = args.batch_size
VAL_SIZE = args.val_size
RANDOM_STATE = args.random_state

LAMBDA_CV      = args.lambda_cv
PASSER_WEIGHT  = args.passer_weight
PASSER_PRIOR_K = args.passer_prior_k
EPS            = 1e-9

XPASS_COL_CANDIDATES = [
    "xpass_pred_calib_oof", "xpass_pred_uncal_oof",
    "xpass_oof_calib", "xpass_oof_uncal",
    "xpass_pred_calib", "xpass_pred_uncal", "xpass", "xPass", "prob"
]

# ======================== Utility functions ========================
def parse_xy(v) -> Tuple[float, float]:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        try:
            return float(v[0]), float(v[1])
        except Exception:
            return np.nan, np.nan
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                a = ast.literal_eval(s)
                return float(a[0]), float(a[1])
            except Exception:
                return np.nan, np.nan
    return np.nan, np.nan


def to_bool01(series) -> pd.Series:
    return (
        pd.Series(series)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
        .astype(int)
    )


def ensure_order(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["match_id", "possession", "minute", "second", "index"] if c in df.columns]
    if not cols:
        df = df.copy()
        df["__order__"] = np.arange(len(df))
        cols = ["__order__"]
    return df.sort_values(cols).reset_index(drop=True)


def zone_value(x_ratio, y_ratio):
    """EPV-like location-value proxy; coordinates are assumed to attack from left to right."""
    x_ratio = np.clip(np.asarray(x_ratio, dtype=float), 0, 1)
    y_ratio = np.clip(np.asarray(y_ratio, dtype=float), 0, 1)
    vx = 1.0 / (1.0 + np.exp(-10.0 * (x_ratio - 0.75)))
    vy = np.exp(-((y_ratio - 0.5) ** 2) / (2 * (0.18 ** 2)))
    return 0.7 * vx + 0.3 * vy


def build_match_key(df: pd.DataFrame) -> pd.Series:
    if "match_id" in df.columns:
        s = df["match_id"].astype("string").fillna("UNK")
        if s.nunique(dropna=True) >= 2:
            return s
    candidate_groups = [
        ["home_team", "away_team", "match_date", "season", "competition"],
        ["home_team", "away_team", "kickoff", "season", "competition"],
        ["home_team", "away_team", "date", "season", "competition"],
    ]
    for cols in candidate_groups:
        have = [c for c in cols if c in df.columns]
        if len(have) >= 2:
            key = df[have].astype("string").fillna("UNK").agg("|".join, axis=1)
            if key.nunique(dropna=True) >= 2:
                return key
    return pd.Series(["match_0"] * len(df), index=df.index, dtype="string")


def build_seq_group_key(df: pd.DataFrame, match_key: pd.Series) -> pd.Series:
    if "possession" in df.columns:
        return match_key.astype("string") + "|pos|" + df["possession"].astype("string").fillna("UNK")
    return match_key.astype("string")


def choose_xpass_prob(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    xpass_col = next((c for c in XPASS_COL_CANDIDATES if c in df.columns), None)
    if xpass_col is not None:
        df["xpass_prob"] = pd.to_numeric(df[xpass_col], errors="coerce").clip(0.001, 0.999)
        print(f"[info] using xPass probability column: {xpass_col}")
        return df, xpass_col

    print("[warning] no xPass column found; using an empirical proxy.")
    len_vals = pd.to_numeric(df["pass_length_calc"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(len_vals) == 0:
        df["xpass_prob"] = 0.5
        return df, "xpass_prob"

    len_bins = np.quantile(len_vals, [0, 0.33, 0.66, 1.0])
    len_bins[0] -= 1e-6
    len_bins[-1] += 1e-6
    ang_bins = np.array([0, 0.35, 0.7, math.pi], dtype=float)
    ang_bins[0] -= 1e-6
    ang_bins[-1] += 1e-6

    df["len_bin_tmp"] = pd.cut(df["pass_length_calc"], bins=len_bins, labels=False, include_lowest=True).astype("Int64")
    df["ang_bin_tmp"] = pd.cut(df["abs_angle"], bins=ang_bins, labels=False, include_lowest=True).astype("Int64")
    df["xbin_tmp"] = pd.cut(df["xs"], bins=np.linspace(0, 1, 13), labels=False, include_lowest=True).astype("Int64")
    df["ybin_tmp"] = pd.cut(df["ys"], bins=np.linspace(0, 1, 9), labels=False, include_lowest=True).astype("Int64")
    df["ctx_key"] = list(zip(df["len_bin_tmp"], df["ang_bin_tmp"], df["xbin_tmp"], df["ybin_tmp"], df["underP"]))

    g = df.groupby("ctx_key")["success"].agg(["sum", "count"]).reset_index()
    a, b = 2.0, 2.0
    g["p_hat"] = (g["sum"] + a) / (g["count"] + a + b)
    pmap = dict(zip(g["ctx_key"], g["p_hat"]))
    df["xpass_prob"] = df["ctx_key"].map(pmap).fillna(g["p_hat"].mean()).clip(0.001, 0.999)
    return df, "xpass_prob"


def compute_cv_obs_success_only(df: pd.DataFrame, seq_group_key: pd.Series, H: int, gamma: float) -> np.ndarray:
    """
    Compute continuation-value labels only for completed passes.
    The label is the discounted sum of EPV increments over the next H events
    within the same match-possession sequence.
    """
    cv = np.full(len(df), np.nan, dtype=float)
    tmp = df.copy()
    tmp["__seq_group__"] = seq_group_key.values

    for _, g in tmp.groupby("__seq_group__", sort=False):
        idx = g.index.to_numpy()
        epv_end = g["EPV_end"].to_numpy(dtype=float)
        succ = g["success"].to_numpy(dtype=int)
        n = len(g)
        for local_i in range(n):
            if succ[local_i] != 1:
                continue
            steps = 0
            gain = 0.0
            epv_prev = epv_end[local_i]
            for local_j in range(local_i + 1, n):
                epv_now = epv_end[local_j]
                if not np.isfinite(epv_now):
                    epv_now = epv_prev
                gain += (gamma ** steps) * (epv_now - epv_prev)
                epv_prev = epv_now
                steps += 1
                if steps >= H:
                    break
            cv[idx[local_i]] = gain
    return cv


def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    window: int,
    seq_group_key: pd.Series,
    tail_mask: Optional[pd.Series] = None,
    pad_for_inference: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows within each match-possession sequence.
    - pad_for_inference=False: retain only complete windows.
    - pad_for_inference=True: left-pad short sequences for inference.
    Returns X_seq, tail-row indices, and valid sequence lengths.
    """
    X_list, row_idx_list, valid_len_list = [], [], []
    tmp = df.copy()
    tmp["__seq_group__"] = seq_group_key.values

    for _, g in tmp.groupby("__seq_group__", sort=False):
        g = g.sort_index()
        idx = g.index.to_numpy()
        Xg = g[feature_cols].to_numpy(dtype=np.float32)
        n = len(g)
        for i in range(n):
            tail_row = idx[i]
            if tail_mask is not None and not bool(tail_mask.loc[tail_row]):
                continue
            start = i - window + 1
            if start < 0:
                if not pad_for_inference:
                    continue
                pad_len = -start
                seq = np.vstack([
                    np.zeros((pad_len, Xg.shape[1]), dtype=np.float32),
                    Xg[0:i + 1]
                ])
                valid_len = i + 1
            else:
                seq = Xg[start:i + 1]
                valid_len = window
            if seq.shape[0] != window:
                continue
            X_list.append(seq)
            row_idx_list.append(tail_row)
            valid_len_list.append(valid_len)

    if not X_list:
        return (
            np.zeros((0, window, len(feature_cols)), dtype=np.float32),
            np.array([], dtype=int),
            np.array([], dtype=int),
        )

    return (
        np.stack(X_list).astype(np.float32),
        np.asarray(row_idx_list, dtype=int),
        np.asarray(valid_len_list, dtype=int),
    )


def estimate_turnover_point(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate an intermediate turnover point along the pass path rather than
    assuming that possession is lost exactly at the mirrored endpoint.
    Higher xPass moves the proxy closer to the target location, whereas long,
    oblique, or pressured passes move the proxy closer to the earlier path.
    """
    xpass = np.clip(df["xpass_prob"].to_numpy(dtype=float), 0.001, 0.999)
    length_norm = np.clip(df["pass_length_calc"].to_numpy(dtype=float) / 40.0, 0.0, 1.0)
    angle_norm = np.clip(df["abs_angle"].to_numpy(dtype=float) / math.pi, 0.0, 1.0)
    underp = np.clip(df["underP"].to_numpy(dtype=float), 0.0, 1.0)

    frac = 0.20 + 0.55 * xpass + 0.08 * (1.0 - underp) - 0.10 * length_norm - 0.05 * angle_norm
    frac = np.clip(frac, 0.12, 0.95)

    tx = df["xs"].to_numpy(dtype=float) + frac * (df["xe"].to_numpy(dtype=float) - df["xs"].to_numpy(dtype=float))
    ty = df["ys"].to_numpy(dtype=float) + frac * (df["ye"].to_numpy(dtype=float) - df["ys"].to_numpy(dtype=float))
    opp_turn = zone_value(1.0 - tx, 1.0 - ty)
    return tx, ty, opp_turn


def compute_passer_marginal_oof(
    df: pd.DataFrame,
    match_key: pd.Series,
    player_col: str = "player",
    prior_k: float = 50.0,
    n_splits: int = 5,
) -> pd.Series:
    """
    Estimate passer marginal contribution from residuals, defined as
    success - xpass_prob, using match-grouped out-of-fold aggregation and
    sample-size shrinkage.
    """
    if player_col not in df.columns:
        return pd.Series(np.zeros(len(df), dtype=float), index=df.index)

    out = pd.Series(np.nan, index=df.index, dtype=float)
    residual = df["success"].astype(float) - df["xpass_prob"].astype(float)
    groups = match_key.astype("string")
    unique_groups = pd.unique(groups)
    if len(unique_groups) < 2:
        tab = pd.DataFrame({player_col: df[player_col], "resid": residual}).groupby(player_col)["resid"].agg(["mean", "count"])
        tab["shrink"] = tab["mean"] * (tab["count"] / (tab["count"] + prior_k))
        return df[player_col].map(tab["shrink"]).fillna(0.0).astype(float)

    n_splits = int(min(max(2, n_splits), len(unique_groups)))
    gkf = GroupKFold(n_splits=n_splits)

    dummy_X = np.zeros((len(df), 1), dtype=float)
    for tr_idx, va_idx in gkf.split(dummy_X, groups=groups):
        tr = df.iloc[tr_idx].copy()
        tab = pd.DataFrame({player_col: tr[player_col].values, "resid": residual.iloc[tr_idx].values}).groupby(player_col)["resid"].agg(["mean", "count"])
        tab["shrink"] = tab["mean"] * (tab["count"] / (tab["count"] + prior_k))
        mapped = df.iloc[va_idx][player_col].map(tab["shrink"])
        out.iloc[va_idx] = mapped.astype(float)

    return out.fillna(0.0).astype(float)


# ======================== Load data and construct basic features ========================
print(f"[info] loading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH, low_memory=False)

# Preserve the original column order.
original_cols = df.columns.tolist()

df = ensure_order(df)
# Coordinates
sx, sy = zip(*df["location"].map(parse_xy))
ex, ey = zip(*df["pass_end_location"].map(parse_xy))
df["sx"], df["sy"], df["ex"], df["ey"] = sx, sy, ex, ey

df["xs"] = pd.to_numeric(df["sx"], errors="coerce") / 120.0
df["ys"] = pd.to_numeric(df["sy"], errors="coerce") / 80.0
df["xe"] = pd.to_numeric(df["ex"], errors="coerce") / 120.0
df["ye"] = pd.to_numeric(df["ey"], errors="coerce") / 80.0

# Labels and basic features
df["success"] = df["pass_outcome"].isna().astype(int)
df["pass_length_calc"] = np.hypot(df["ex"] - df["sx"], df["ey"] - df["sy"])
if "pass_angle" in df.columns:
    df["abs_angle"] = np.abs(pd.to_numeric(df["pass_angle"], errors="coerce"))
else:
    df["abs_angle"] = np.abs(np.arctan2(df["ey"] - df["sy"], df["ex"] - df["sx"]))

df["underP"] = to_bool01(df.get("under_pressure", 0))
df["len_cos"] = (df["xe"] - df["xs"]).fillna(0.0)
df["minute"] = pd.to_numeric(df.get("minute", 0), errors="coerce").fillna(0.0)
df["second"] = pd.to_numeric(df.get("second", 0), errors="coerce").fillna(0.0)
df["is_first_half"] = (df["minute"] < 45).astype(int)

# xPass probabilities
match_key = build_match_key(df)
seq_group_key = build_seq_group_key(df, match_key)
df, _ = choose_xpass_prob(df)

# EPV-like spatial value proxy
df["EPV_start"] = zone_value(df["xs"], df["ys"])
df["EPV_end"] = zone_value(df["xe"], df["ye"])
df["EPV_opp_start"] = zone_value(1.0 - df["xs"], 1.0 - df["ys"])
df["EPV_opp_end"] = zone_value(1.0 - df["xe"], 1.0 - df["ye"])

# ======================== Success-conditional continuation-value labels ========================
print("[info] computing success-conditional continuation-value labels...")
df["cv_obs"] = compute_cv_obs_success_only(df, seq_group_key=seq_group_key, H=H_STEPS, gamma=GAMMA)
cv_train_mean = float(pd.Series(df["cv_obs"]).replace([np.inf, -np.inf], np.nan).dropna().mean())
if not np.isfinite(cv_train_mean):
    cv_train_mean = 0.0
print(f"[info] cv_obs mean among completed passes: {cv_train_mean:.6f}")

# ======================== Match-group split and within-possession windows ========================
feature_cols = [
    "xs", "ys", "xe", "ye", "underP", "pass_length_calc", "abs_angle", "xpass_prob",
    "len_cos", "minute", "second", "is_first_half"
]
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Completed passes with observed continuation-value labels.
success_tail_mask = (df["success"] == 1) & df["cv_obs"].notna()
if success_tail_mask.sum() < 100:
    raise RuntimeError("Too few completed passes with continuation-value labels for stable training.")

# Split by match before constructing training and validation sequences.
success_rows = df.index[success_tail_mask].to_numpy()
match_for_success = match_key.loc[success_rows].astype("string")
gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)
tr_pos, va_pos = next(gss.split(success_rows.reshape(-1, 1), groups=match_for_success))
train_matches = set(match_for_success.iloc[tr_pos].tolist())
val_matches = set(match_for_success.iloc[va_pos].tolist())

train_tail_mask = success_tail_mask & match_key.isin(train_matches)
val_tail_mask   = success_tail_mask & match_key.isin(val_matches)

Xtr, tr_row_idx, _ = build_sequences(df, feature_cols, WINDOW, seq_group_key, tail_mask=train_tail_mask, pad_for_inference=False)
Xva, va_row_idx, _ = build_sequences(df, feature_cols, WINDOW, seq_group_key, tail_mask=val_tail_mask,   pad_for_inference=False)
ytr = df.loc[tr_row_idx, "cv_obs"].to_numpy(dtype=np.float32)
yva = df.loc[va_row_idx, "cv_obs"].to_numpy(dtype=np.float32)

print(f"[info] train sequences={len(Xtr)}, validation sequences={len(Xva)}")
if len(Xtr) < 100 or len(Xva) < 30:
    raise RuntimeError("Too few sequences after grouped splitting. Increase the number of matches or reduce WINDOW / VAL_SIZE.")

# ======================== GRU training ========================
print("[info] training GRU for conditional receiver continuation value...")
try:
    import torch
    import torch.nn as nn

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    class TinyGRU(nn.Module):
        def __init__(self, d_in: int, d_hid: int = 32):
            super().__init__()
            self.gru = nn.GRU(d_in, d_hid, batch_first=True)
            self.fc = nn.Linear(d_hid, 1)

        def forward(self, x):
            _, h = self.gru(x)
            return self.fc(h[-1]).squeeze(-1)

    def make_loader(X, y, batch_size):
        ds = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = TinyGRU(d_in=len(feature_cols), d_hid=HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.SmoothL1Loss()

    tr_loader = make_loader(Xtr, ytr, BATCH)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    train_hist, val_hist = [], []
    patience, bad_epochs = 10, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, nb = 0.0, 0
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = lossf(pred, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.item())
            nb += 1

        train_loss = running / max(nb, 1)
        model.eval()
        with torch.no_grad():
            val_pred = model(Xva_t)
            val_loss = float(lossf(val_pred, yva_t).item())

        train_hist.append(train_loss)
        val_hist.append(val_loss)
        print(f"[Epoch {epoch:03d}] train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience and epoch >= 15:
                print(f"[info] early stopping at epoch {epoch}; best_val={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict conditional continuation value for all events; short group starts are left-padded.
    Xall, all_row_idx, all_valid_len = build_sequences(
        df, feature_cols, WINDOW, seq_group_key, tail_mask=None, pad_for_inference=True
    )
    Xall_t = torch.tensor(Xall, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred_all = model(Xall_t).cpu().numpy().astype(float)

    receiver_cv_cond = pd.Series(np.full(len(df), cv_train_mean, dtype=float), index=df.index)
    receiver_cv_cond.loc[all_row_idx] = pred_all
    # Shrink predictions for very short contexts toward the training mean.
    short_mask = pd.Series(False, index=df.index)
    short_mask.loc[all_row_idx] = (all_valid_len < WINDOW)
    receiver_cv_cond.loc[short_mask] = 0.5 * receiver_cv_cond.loc[short_mask] + 0.5 * cv_train_mean

    # Loss curve
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(train_hist) + 1), train_hist, marker="o", label="train")
    plt.plot(range(1, len(val_hist) + 1), val_hist, marker="o", label="val")
    plt.xlabel("Epoch")
    plt.ylabel("SmoothL1Loss")
    plt.title("GRU loss curve (group-split by match)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "gru_loss_curve_v4.png"), dpi=180)
    plt.close()

except Exception as e:
    print(f"[warning] GRU training unavailable; using spatial proxy. reason={e}")
    receiver_cv_cond = (df["EPV_end"] - df["EPV_start"]).fillna(0.0).astype(float)
    receiver_cv_cond = 0.7 * receiver_cv_cond + 0.3 * cv_train_mean

# Optional clipping can be enabled for more conservative continuation-value estimates.
# receiver_cv_cond = receiver_cv_cond.clip(lower=receiver_cv_cond.quantile(0.01), upper=receiver_cv_cond.quantile(0.99))

df["receiver_CV_cond"] = receiver_cv_cond.astype(float)

# ======================== Passer marginal contribution: OOF residual shrinkage ========================
print("[info] computing passer marginal contribution with OOF shrinkage...")
df["passer_marginal"] = compute_passer_marginal_oof(
    df=df,
    match_key=match_key,
    player_col="player" if "player" in df.columns else "__no_player__",
    prior_k=PASSER_PRIOR_K,
    n_splits=5,
)

df["passer_marginal"] = 0.10 * np.tanh(df["passer_marginal"].astype(float))

# ======================== Failure-cost proxy based on an intermediate turnover point ========================
turn_x, turn_y, opp_turn = estimate_turnover_point(df)
df["turnover_x_ratio"] = turn_x
df["turnover_y_ratio"] = turn_y
df["EPV_opp_turnover"] = opp_turn
# Failure cost is dominated by the intermediate turnover proxy while retaining start and end information.

df["CostOnFailure"] = (
    0.70 * df["EPV_opp_turnover"]
    + 0.15 * df["EPV_opp_start"]
    + 0.15 * df["EPV_opp_end"]
).astype(float)

# ======================== Gain on success and expected reward ========================
# receiver_CV_cond is conditional on pass completion and enters GainOnSuccess directly.

df["GainOnSuccess"] = (
    (df["EPV_end"] - df["EPV_start"]) +
    LAMBDA_CV * df["receiver_CV_cond"] +
    PASSER_WEIGHT * df["passer_marginal"]
).astype(float)

df["risk"] = (1.0 - df["xpass_prob"].astype(float)).clip(0.001, 0.999)
df["expected_reward"] = (
    df["xpass_prob"] * df["GainOnSuccess"] -
    (1.0 - df["xpass_prob"]) * df["CostOnFailure"]
).astype(float)
df["net_risk_reward_ratio"] = df["expected_reward"] / df["risk"].clip(lower=EPS)

# ======================== Outputs ========================
# Derived columns to append after the original input columns.
preferred_new_cols = [
    "sx", "sy", "ex", "ey",
    "xs", "ys", "xe", "ye",
    "success",
    "pass_length_calc", "abs_angle",
    "underP", "len_cos", "is_first_half",
    "xpass_prob",
    "EPV_start", "EPV_end", "EPV_opp_start", "EPV_opp_end",
    "cv_obs", "receiver_CV_cond",
    "passer_marginal",
    "turnover_x_ratio", "turnover_y_ratio", "EPV_opp_turnover",
    "GainOnSuccess", "CostOnFailure",
    "risk", "expected_reward", "net_risk_reward_ratio"
]

# Keep only columns generated by this script.
generated_cols = [c for c in preferred_new_cols if c in df.columns and c not in original_cols]

# Append any additional derived columns not listed above.
other_new_cols = [c for c in df.columns if c not in original_cols and c not in generated_cols]

append_cols = generated_cols + other_new_cols

# Final export order: original columns followed by derived columns.
full_cols = original_cols + append_cols

full_path = os.path.join(RUN_DIR, "passes_with_appended_new_cols_v4.csv")
df[full_cols].to_csv(full_path, index=False, encoding="utf-8-sig")

# Export focused output tables for reproducibility checks.
out_cols_cv = [
    c for c in [
        "match_id", "possession", "team", "player", "minute", "second",
        "success", "xpass_prob", "cv_obs", "receiver_CV_cond"
    ] if c in df.columns
]

out_cols_rr = [
    c for c in [
        "match_id", "possession", "team", "player", "minute", "second",
        "pass_type", "pass_height", "pass_body_part", "under_pressure"
    ] if c in df.columns
] + [
    c for c in [
        "xpass_prob", "risk", "EPV_start", "EPV_end", "receiver_CV_cond",
        "passer_marginal", "GainOnSuccess", "CostOnFailure", "expected_reward",
        "net_risk_reward_ratio", "turnover_x_ratio", "turnover_y_ratio"
    ] if c in df.columns
]

cv_path = os.path.join(RUN_DIR, "cv_predictions_v4.csv")
rr_path = os.path.join(RUN_DIR, "risk_reward_per_pass_v4.csv")
pm_path = os.path.join(RUN_DIR, "passer_marginal_oof_v4.csv")

df[out_cols_cv].to_csv(cv_path, index=False, encoding="utf-8-sig")
df[out_cols_rr].to_csv(rr_path, index=False, encoding="utf-8-sig")

passer_dump_cols = [
    c for c in ["match_id", "team", "player", "success", "xpass_prob", "passer_marginal"]
    if c in df.columns
]
df[passer_dump_cols].to_csv(pm_path, index=False, encoding="utf-8-sig")

# Risk-reward frontier figure.
plt.figure(figsize=(8, 6))
plt.scatter(df["risk"], df["expected_reward"], s=10, alpha=0.25)
plt.axhline(0.0, linestyle="--", linewidth=1)
plt.xlabel("Risk = 1 - xPass")
plt.ylabel("Expected Reward")
plt.title("Risk-Reward Frontier of Passes (v4)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "frontier_v4.png"), dpi=300)
plt.close()

print("\n[done] risk-reward modeling completed.")
print(f"[output] {full_path}")
print(f"[output] {cv_path}")
print(f"[output] {rr_path}")
print(f"[output] {pm_path}")
print(f"[output] {os.path.join(RUN_DIR, 'frontier_v4.png')}")
print(f"[output] {os.path.join(RUN_DIR, 'gru_loss_curve_v4.png')}")
