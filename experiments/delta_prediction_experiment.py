"""
delta_prediction_experiment.py
===============================
Goal: Predict month-to-month CHANGE in CalFresh applications (delta)
      instead of raw counts, using only CalFresh Google Trends + lagged
      application variables.

Experiment design
-----------------
Target:
  delta_apps_t = SNAP_Applications_t - SNAP_Applications_{t-1}
  (percent change also computed as pct_delta_t but not used as primary target)

Features used (no FoodBank, no demographics for this round):
  - CalFresh Trends monthly mean and max, lagged 1–3 months
  - Prior-month and 2-month-back application counts
  - Prior-month delta (first-difference autoregression)
  - Month-of-year (sine/cosine encoding + raw month)

Models:
  1. naive      — predict delta = 0 for every county-month
  2. lr_lag     — linear regression, lag-only features
  3. xgb_lag    — XGBoost, lag-only features
  4. lr_trends  — linear regression, lag + CalFresh Trends
  5. xgb_trends — XGBoost, lag + CalFresh Trends  (primary)

Validation: walk-forward (time-ordered; never trains on future data)

Run from project root:
  python experiments/delta_prediction_experiment.py
"""

import os
import sys
import glob
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Allow imports from the pipeline package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config

logging.basicConfig(level=logging.WARNING)  # suppress pipeline debug noise

# ── Output directory ───────────────────────────────────────────────────────────
OUT_DIR = os.path.join(config.OUTPUTS_ROOT, "experiments", "delta_predictions")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Settings ───────────────────────────────────────────────────────────────────
WALK_FORWARD_MIN_MONTHS = 12   # minimum training months before first test window
SPIKE_PERCENTILE        = 90   # top 10% delta values are labelled "spikes"
RANDOM_STATE            = 42

# XGBoost parameters — kept simple for experimentation; can be tuned later
XGBOOST_PARAMS = dict(
    n_estimators    = 300,
    max_depth       = 5,
    learning_rate   = 0.05,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    reg_lambda      = 2,
    random_state    = RANDOM_STATE,
    n_jobs          = -1,
    verbosity       = 0,
)

# ── Feature sets ───────────────────────────────────────────────────────────────
TARGET = "delta_apps"

# Experiment A: lag-only (no Trends) — control condition
LAG_ONLY_FEATURES = [
    "apps_lag1",    # prior-month SNAP applications
    "apps_lag2",    # two-months-back applications
    "delta_lag1",   # prior-month delta (autoregressive)
    "delta_lag2",   # two-months-back delta
    "month_sin",    # cyclical month encoding (sin)
    "month_cos",    # cyclical month encoding (cos)
]

# Experiment B: lag + CalFresh Trends — treatment condition
LAG_TRENDS_FEATURES = LAG_ONLY_FEATURES + [
    "calfresh_mean_lag1",  # monthly-mean CalFresh Trends, 1-month lag
    "calfresh_mean_lag2",  # monthly-mean CalFresh Trends, 2-month lag
    "calfresh_mean_lag3",  # monthly-mean CalFresh Trends, 3-month lag
    "calfresh_max_lag1",   # monthly-max CalFresh Trends, 1-month lag
    "calfresh_max_lag2",   # monthly-max CalFresh Trends, 2-month lag
]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_base_data() -> pd.DataFrame:
    """
    Load training_data.csv — the merged SNAP + trends + demographics table
    produced by stage 2 of the main pipeline.

    This file already contains monthly_average_CalFresh (monthly mean of weekly
    Trends values from the PRIOR month, due to the 1-month temporal shift applied
    in stage1_load_raw.load_snap_applications).
    """
    path = config.TRAINING_DATA_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"training_data.csv not found at {path}\n"
            "Run the main pipeline first: python run_pipeline.py --stages 2"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"  training_data.csv: {len(df):,} rows, "
          f"{df['county'].nunique()} counties, "
          f"{df['date'].min().date()} – {df['date'].max().date()}")
    return df


def load_calfresh_weekly_trends() -> pd.DataFrame:
    """
    Load all weekly CalFresh Google Trends CSVs from the nested folder structure:
      src/data/trends/CalFresh2017-2025/{DMA}/{DMA}{year}.csv

    Each CSV has a quoted header row ("Time","CalFresh") followed by weekly rows
    with date (YYYY-MM-DD) and value (0–100 Google Trends index).

    Returns DataFrame with columns: metro_area, date (datetime), value (float).
    """
    trends_root = os.path.join(config.TRENDS_DIR, "CalFresh2017-2025")
    if not os.path.exists(trends_root):
        print(f"  WARNING: CalFresh trends folder not found at {trends_root}")
        return pd.DataFrame(columns=["metro_area", "date", "value"])

    # One sub-folder per DMA; one or more CSVs per year inside each DMA folder
    dma_dirs = [
        d for d in os.listdir(trends_root)
        if os.path.isdir(os.path.join(trends_root, d))
    ]

    dfs = []
    for dma in dma_dirs:
        dma_path = os.path.join(trends_root, dma)
        for fpath in glob.glob(os.path.join(dma_path, "*.csv")):
            df = _parse_trends_csv(fpath)
            if not df.empty:
                df["metro_area"] = dma
                dfs.append(df)

    if not dfs:
        print("  WARNING: No weekly CalFresh trend CSVs found.")
        return pd.DataFrame(columns=["metro_area", "date", "value"])

    combined = pd.concat(dfs, ignore_index=True)
    # Remove duplicate weeks for the same DMA (can occur across year-boundary files)
    combined = combined.drop_duplicates(subset=["metro_area", "date"])
    print(f"  CalFresh weekly trends: {combined['metro_area'].nunique()} DMAs, "
          f"{len(combined):,} weekly rows, "
          f"{combined['date'].min().date()} – {combined['date'].max().date()}")
    return combined[["metro_area", "date", "value"]]


def _parse_trends_csv(fpath: str) -> pd.DataFrame:
    """
    Parse one Google Trends export CSV with header row ("Time","CalFresh").
    Returns DataFrame with columns: date (datetime), value (float).
    """
    df = pd.read_csv(fpath, comment=None)
    # Normalise column names — Google exports use quoted "Time" and "CalFresh"
    df.columns = [c.strip().strip('"').lower() for c in df.columns]
    if "time" not in df.columns or df.shape[1] < 2:
        return pd.DataFrame(columns=["date", "value"])
    # Second column is the value regardless of its name
    df = df.iloc[:, :2].copy()
    df.columns = ["date", "value"]
    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["date", "value"])


def compute_monthly_max(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate weekly CalFresh Trends to monthly MAX per DMA.

    The main pipeline already computes the monthly mean (monthly_average_CalFresh).
    Monthly max captures peak-week search interest, which may signal acute
    food-insecurity events better than the average.

    Returns columns: metro_area, month_date (first-of-month datetime), calfresh_monthly_max.
    """
    if weekly_df.empty:
        return pd.DataFrame(columns=["metro_area", "month_date", "calfresh_monthly_max"])

    weekly_df = weekly_df.copy()
    weekly_df["year"]  = weekly_df["date"].dt.year
    weekly_df["month"] = weekly_df["date"].dt.month

    monthly = (
        weekly_df
        .groupby(["metro_area", "year", "month"])["value"]
        .max()
        .reset_index()
        .rename(columns={"value": "calfresh_monthly_max"})
    )
    monthly["month_date"] = pd.to_datetime(
        monthly[["year", "month"]].assign(day=1)
    )
    return monthly[["metro_area", "month_date", "calfresh_monthly_max"]]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Build the delta modeling panel
# ══════════════════════════════════════════════════════════════════════════════

def build_delta_panel(base_df: pd.DataFrame, monthly_max_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the county-month panel used for the delta prediction experiment.

    Temporal note on CalFresh Trends:
      - training_data.csv was built with a 1-month temporal shift:
          trends joined on (metro_area, trend_date) where trend_date = SNAP date - 1 month
      - So 'monthly_average_CalFresh' in training_data.csv is already the PRIOR month's
        average. We rename it 'calfresh_mean_lag1' and then shift it again within county
        groups to get lag2 and lag3.
      - Monthly max is merged the same way (join on the month prior to the SNAP date).

    All lag operations are performed within county groups, sorted by date, to
    prevent cross-county data leakage.
    """
    df = base_df.copy().sort_values(["county", "date"]).reset_index(drop=True)

    # ── Merge monthly max Trends (same temporal alignment as mean: prior month) ──
    if not monthly_max_df.empty:
        # The SNAP date's corresponding trend month is (SNAP date - 1 month)
        df["_trend_month"] = (df["date"] - pd.DateOffset(months=1)).dt.to_period("M").dt.to_timestamp()
        monthly_max_df = monthly_max_df.copy()
        monthly_max_df["_trend_month"] = monthly_max_df["month_date"].dt.to_period("M").dt.to_timestamp()
        df = df.merge(
            monthly_max_df[["metro_area", "_trend_month", "calfresh_monthly_max"]],
            on=["metro_area", "_trend_month"],
            how="left",
        )
        df = df.drop(columns=["_trend_month"])
    else:
        df["calfresh_monthly_max"] = np.nan

    # ── Rename the already-lagged mean trend for clarity ──
    if "monthly_average_CalFresh" in df.columns:
        df = df.rename(columns={"monthly_average_CalFresh": "calfresh_mean_lag1"})
    else:
        df["calfresh_mean_lag1"] = np.nan

    # ── Compute delta targets (within each county) ──
    # Sort once more to be safe; shift(1) gives the immediately prior month's value
    df = df.sort_values(["county", "date"])
    df["apps_lag1"]  = df.groupby("county")["SNAP_Applications"].shift(1)
    df["delta_apps"] = df["SNAP_Applications"] - df["apps_lag1"]      # PRIMARY TARGET
    df["pct_delta"]  = df["delta_apps"] / df["apps_lag1"]             # optional; not primary

    # ── Additional lagged application variables ──
    df["apps_lag2"]  = df.groupby("county")["SNAP_Applications"].shift(2)
    df["delta_lag1"] = df.groupby("county")["delta_apps"].shift(1)
    df["delta_lag2"] = df.groupby("county")["delta_apps"].shift(2)

    # ── Lagged CalFresh Trends (mean) ──
    # calfresh_mean_lag1 is already the prior-month mean (from temporal shift in pipeline)
    # Shifting once more within county groups gives the 2- and 3-month lags
    df["calfresh_mean_lag2"] = df.groupby("county")["calfresh_mean_lag1"].shift(1)
    df["calfresh_mean_lag3"] = df.groupby("county")["calfresh_mean_lag1"].shift(2)

    # ── Lagged CalFresh Trends (max) ──
    df["calfresh_max_lag1"] = df["calfresh_monthly_max"]               # already prior month
    df["calfresh_max_lag2"] = df.groupby("county")["calfresh_monthly_max"].shift(1)

    # ── Seasonality features ──
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

    # Drop the intermediate max column (replaced by named lag columns)
    df = df.drop(columns=["calfresh_monthly_max"], errors="ignore")

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray = None,
) -> dict:
    """
    Compute MAE, RMSE, R², directional accuracy, and spike recall.

    Directional accuracy: fraction of predictions where sign(pred) == sign(actual).
    Rows where actual == 0 are excluded (no direction to compare).

    Spike recall: among actual positive spikes (top SPIKE_PERCENTILE of training
    deltas), what fraction did the model also predict as spikes?
    Spike threshold derived from training data so it's not contaminated by test data.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")

    # Directional accuracy (exclude zero-delta months — no sign to compare)
    nonzero_mask = y_true != 0
    if nonzero_mask.sum() > 0:
        dir_acc = float(np.mean(np.sign(y_pred[nonzero_mask]) == np.sign(y_true[nonzero_mask])))
    else:
        dir_acc = float("nan")

    # Spike recall (positive spikes only — large increases are the policy-relevant case)
    spike_recall = float("nan")
    if y_train is not None and len(y_train) > 0:
        threshold   = float(np.percentile(y_train, SPIKE_PERCENTILE))
        true_spikes = y_true >= threshold
        pred_spikes = y_pred >= threshold
        if true_spikes.sum() > 0:
            spike_recall = float(np.mean(pred_spikes[true_spikes]))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy": dir_acc,
        "spike_recall": spike_recall,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Naive baseline model
# ══════════════════════════════════════════════════════════════════════════════

class NaiveNoChangeModel:
    """Predict delta = 0 for every county-month (no-change baseline)."""
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.zeros(len(X))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Walk-forward validation
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward(
    panel_df: pd.DataFrame,
    feature_cols: list,
    model_name: str,
    model_factory,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-ordered walk-forward validation.

    For each month T (after WALK_FORWARD_MIN_MONTHS months of history):
      - Train on all county-months with date < T
      - Predict all county-months with date == T
      - Record fold metrics and per-county-month predictions

    Returns:
      fold_df  — one row per test month (model, month, n_test, mae, rmse, r2, ...)
      pred_df  — one row per county-month (county, date, actual_delta, predicted_delta, residual)
    """
    required = feature_cols + [TARGET]
    clean    = panel_df.dropna(subset=required).copy()
    months   = sorted(clean["date"].unique())

    fold_rows = []
    pred_rows = []

    for test_month in months:
        train_mask = clean["date"] < test_month
        test_mask  = clean["date"] == test_month

        # Skip if not enough training history
        n_train_months = clean.loc[train_mask, "date"].nunique()
        if n_train_months < WALK_FORWARD_MIN_MONTHS:
            continue
        if test_mask.sum() == 0:
            continue

        X_train = clean.loc[train_mask, feature_cols]
        y_train = clean.loc[train_mask, TARGET]
        X_test  = clean.loc[test_mask,  feature_cols]
        y_test  = clean.loc[test_mask,  TARGET]

        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = np.array(model.predict(X_test), dtype=float)

        metrics = compute_metrics(y_test.values, y_pred, y_train=y_train.values)
        fold_rows.append({
            "model":   model_name,
            "month":   test_month,
            "n_test":  int(test_mask.sum()),
            **metrics,
        })

        # Per-county-month predictions
        rows = clean.loc[test_mask, ["county", "date", "metro_area"]].copy()
        rows["actual_delta"]    = y_test.values
        rows["predicted_delta"] = y_pred
        rows["residual"]        = y_test.values - y_pred
        rows["model"]           = model_name
        pred_rows.append(rows)

    fold_df = pd.DataFrame(fold_rows)
    pred_df = (
        pd.concat(pred_rows, ignore_index=True)
        if pred_rows
        else pd.DataFrame(columns=["county", "date", "metro_area",
                                   "actual_delta", "predicted_delta", "residual", "model"])
    )
    return fold_df, pred_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Run all experiments
# ══════════════════════════════════════════════════════════════════════════════

def run_all_experiments(panel_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run walk-forward for each model × feature-set combination.

    Experiments:
      naive      — predict delta = 0 (irreducible baseline)
      lr_lag     — linear regression, lag-only features
      xgb_lag    — XGBoost,           lag-only features
      lr_trends  — linear regression, lag + CalFresh Trends
      xgb_trends — XGBoost,           lag + CalFresh Trends
    """
    experiments = [
        ("naive",       LAG_ONLY_FEATURES,   lambda: NaiveNoChangeModel()),
        ("lr_lag",      LAG_ONLY_FEATURES,   lambda: LinearRegression()),
        ("xgb_lag",     LAG_ONLY_FEATURES,   lambda: XGBRegressor(**XGBOOST_PARAMS)),
        ("lr_trends",   LAG_TRENDS_FEATURES, lambda: LinearRegression()),
        ("xgb_trends",  LAG_TRENDS_FEATURES, lambda: XGBRegressor(**XGBOOST_PARAMS)),
    ]

    all_folds = []
    all_preds = []

    for name, features, factory in experiments:
        print(f"  {name} ...", end=" ", flush=True)
        fold_df, pred_df = walk_forward(panel_df, features, name, factory)
        all_folds.append(fold_df)
        all_preds.append(pred_df)

        if not fold_df.empty:
            avg = fold_df[["mae", "rmse", "r2", "directional_accuracy"]].mean()
            print(
                f"MAE={avg['mae']:>8,.1f}  "
                f"RMSE={avg['rmse']:>8,.1f}  "
                f"R²={avg['r2']:>6.3f}  "
                f"DirAcc={avg['directional_accuracy']:.3f}"
            )
        else:
            print("(no results)")

    return (
        pd.concat(all_folds, ignore_index=True),
        pd.concat(all_preds, ignore_index=True),
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: XGBoost feature importance (full-data fit)
# ══════════════════════════════════════════════════════════════════════════════

def get_xgb_feature_importance(panel_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Train XGBoost on the full dataset and return feature importances.
    This is an in-sample diagnostic — use walk-forward results for evaluation.
    """
    clean = panel_df.dropna(subset=feature_cols + [TARGET])
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(clean[feature_cols], clean[TARGET])
    return (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_actual_vs_predicted(pred_df: pd.DataFrame, model_name: str) -> None:
    """Scatter plot of actual vs predicted delta for one model."""
    sub = pred_df[pred_df["model"] == model_name].dropna(
        subset=["actual_delta", "predicted_delta"]
    )
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sub["actual_delta"], sub["predicted_delta"], alpha=0.3, s=8, color="steelblue")
    # Perfect-prediction line
    lim = max(abs(sub["actual_delta"].max()), abs(sub["actual_delta"].min())) * 1.1
    ax.axline((0, 0), slope=1, color="red", linewidth=1, label="perfect prediction")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Actual delta applications")
    ax.set_ylabel("Predicted delta applications")
    ax.set_title(f"Actual vs Predicted Delta — {model_name}")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"actual_vs_predicted_{model_name}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_residuals(pred_df: pd.DataFrame, model_name: str) -> None:
    """Residual plot (predicted vs residual) for one model."""
    sub = pred_df[pred_df["model"] == model_name].dropna(
        subset=["predicted_delta", "residual"]
    )
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(sub["predicted_delta"], sub["residual"], alpha=0.3, s=8, color="steelblue")
    ax.axhline(0, color="red", linewidth=1)
    ax.set_xlabel("Predicted delta applications")
    ax.set_ylabel("Residual  (actual − predicted)")
    ax.set_title(f"Residuals — {model_name}")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"residuals_{model_name}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_directional_accuracy(fold_df: pd.DataFrame) -> None:
    """Bar chart of mean directional accuracy by model."""
    summary = (
        fold_df.groupby("model")["directional_accuracy"]
        .mean()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(summary.index, summary.values, color="steelblue", edgecolor="white")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="random (50%)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Directional accuracy (mean across folds)")
    ax.set_title("Does the model predict increase vs decrease correctly?")
    ax.legend()
    # Annotate bars
    for bar, val in zip(bars, summary.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "directional_accuracy_by_model.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 65)
    print("DELTA PREDICTION EXPERIMENT")
    print("=" * 65)

    # ── 1. Load data ────────────────────────────────────────────────────────
    print("\n[1] Loading data ...")
    base_df    = load_base_data()
    weekly_df  = load_calfresh_weekly_trends()
    max_df     = compute_monthly_max(weekly_df)

    # ── 2. Build delta panel ─────────────────────────────────────────────────
    print("\n[2] Building delta modeling panel ...")
    panel_df = build_delta_panel(base_df, max_df)

    # Summary of panel
    n_counties  = panel_df["county"].nunique()
    n_months    = panel_df["date"].nunique()
    dmas        = sorted(panel_df["metro_area"].dropna().unique())
    lag_ok      = panel_df.dropna(subset=LAG_ONLY_FEATURES   + [TARGET])
    trends_ok   = panel_df.dropna(subset=LAG_TRENDS_FEATURES + [TARGET])

    print(f"\n  ── Panel summary ────────────────────────────────────────")
    print(f"  Counties : {n_counties}")
    print(f"  Months   : {n_months}  "
          f"({panel_df['date'].min().date()} – {panel_df['date'].max().date()})")
    print(f"  DMAs     : {len(dmas)}")
    for d in dmas:
        print(f"             {d}")
    print(f"  Target   : {TARGET}")
    print(f"\n  Lag-only features    ({len(LAG_ONLY_FEATURES)}):   {LAG_ONLY_FEATURES}")
    print(f"  Lag+Trends features  ({len(LAG_TRENDS_FEATURES)}): {LAG_TRENDS_FEATURES}")
    print(f"\n  Complete rows — lag-only  : {len(lag_ok):,}")
    print(f"  Complete rows — lag+trends: {len(trends_ok):,}")
    print(f"  ─────────────────────────────────────────────────────────")

    # ── 3. Walk-forward experiments ──────────────────────────────────────────
    print("\n[3] Running walk-forward experiments ...")
    fold_df, pred_df = run_all_experiments(panel_df)

    # ── 4. Summary table ─────────────────────────────────────────────────────
    print("\n[4] Overall summary (mean across folds) ...")
    summary = (
        fold_df
        .groupby("model")[["mae", "rmse", "r2", "directional_accuracy", "spike_recall"]]
        .mean()
        .round(4)
        .sort_values("mae")
    )
    print()
    print(summary.to_string())

    # ── 5. Feature importance ────────────────────────────────────────────────
    print("\n[5] Feature importance — xgb_trends (full-data fit) ...")
    fi = get_xgb_feature_importance(panel_df, LAG_TRENDS_FEATURES)
    print(fi.to_string(index=False))

    # ── 6. Experiment comparison ─────────────────────────────────────────────
    print("\n[6] Lag-only vs Lag+Trends: does CalFresh Trends improve delta prediction?")
    for mtype in ["lr", "xgb"]:
        lag_key    = f"{mtype}_lag"
        trends_key = f"{mtype}_trends"
        if lag_key not in summary.index or trends_key not in summary.index:
            continue
        lag_row    = summary.loc[lag_key]
        trends_row = summary.loc[trends_key]
        mae_delta  = lag_row["mae"] - trends_row["mae"]
        mae_pct    = mae_delta / lag_row["mae"] * 100 if lag_row["mae"] > 0 else float("nan")
        r2_delta   = trends_row["r2"] - lag_row["r2"]
        dir_delta  = trends_row["directional_accuracy"] - lag_row["directional_accuracy"]
        print(f"\n  {mtype.upper()} — adding CalFresh Trends:")
        print(f"    MAE: {lag_row['mae']:,.1f} → {trends_row['mae']:,.1f}  "
              f"({mae_pct:+.1f}%  Δ={mae_delta:+,.1f})")
        print(f"    R² : {lag_row['r2']:.4f} → {trends_row['r2']:.4f}  ({r2_delta:+.4f})")
        print(f"    Dir: {lag_row['directional_accuracy']:.4f} → "
              f"{trends_row['directional_accuracy']:.4f}  ({dir_delta:+.4f})")

    best_model = summary["mae"].idxmin()
    print(f"\n  Best model by MAE: {best_model}")

    # ── 7. Save outputs ──────────────────────────────────────────────────────
    print(f"\n[7] Saving outputs to {OUT_DIR} ...")

    fold_df.to_csv(os.path.join(OUT_DIR, "fold_metrics.csv"), index=False)
    print(f"  Saved: fold_metrics.csv  ({len(fold_df)} rows)")

    summary.to_csv(os.path.join(OUT_DIR, "summary_metrics.csv"))
    print(f"  Saved: summary_metrics.csv")

    pred_df.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)
    print(f"  Saved: predictions.csv  ({len(pred_df)} rows)")

    fi.to_csv(os.path.join(OUT_DIR, "feature_importance_xgb_trends.csv"), index=False)
    print(f"  Saved: feature_importance_xgb_trends.csv")

    dir_summary = (
        fold_df.groupby("model")[["directional_accuracy", "spike_recall"]]
        .agg(["mean", "std"])
        .round(4)
    )
    dir_summary.to_csv(os.path.join(OUT_DIR, "directional_accuracy_summary.csv"))
    print(f"  Saved: directional_accuracy_summary.csv")

    # ── 8. Plots ─────────────────────────────────────────────────────────────
    print("\n[8] Generating plots ...")
    plot_actual_vs_predicted(pred_df, "xgb_trends")
    plot_actual_vs_predicted(pred_df, "xgb_lag")
    plot_residuals(pred_df, "xgb_trends")
    plot_directional_accuracy(fold_df)

    print("\n" + "=" * 65)
    print("Done.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
