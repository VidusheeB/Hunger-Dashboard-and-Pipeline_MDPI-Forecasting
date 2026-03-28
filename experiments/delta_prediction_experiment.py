"""
delta_prediction_experiment.py
===============================
Goal: Predict month-to-month PERCENT CHANGE in CalFresh applications
      using the full feature set (demographics + Google Trends for
      CalFresh and FoodBank) vs a no-trends baseline.

This directly tests the original research question: does Google Trends
add predictive value beyond what demographics and lagged applications
already capture — but for DELTA rather than level predictions?

Experiment design
-----------------
Target:
  pct_delta_t = (SNAP_Applications_t - SNAP_Applications_{t-1})
                / SNAP_Applications_{t-1}
  Winsorised at ±100% to remove extreme small-county outliers.

Feature sets compared:
  A. baseline  — demographics + lagged SNAP + seasonality (no Trends)
  B. trends    — baseline + CalFresh Trends + FoodBank Trends

Both use the same pre-engineered features from features.csv (all 14 DMAs,
all 58 counties — full coverage).

Models:
  1. naive           — predict 0% change (no-change baseline)
  2. xgb_baseline    — XGBoost, demographic + lag features only
  3. xgb_trends      — XGBoost, full feature set with Trends
  (linear regression included for reference)

Tuned XGBoost hyperparameters from config.XGBOOST_PARAMS are used throughout.

Validation: walk-forward (time-ordered; never trains on future data)

Run from project root:
  python experiments/delta_prediction_experiment.py
"""

import os
import sys
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
SPIKE_PERCENTILE        = 90   # top 10% pct_delta values = "spikes"
PCT_DELTA_WINSOR        = 1.0  # winsorise at ±100% (removes extreme small-county outliers)

# Use the tuned XGBoost hyperparameters from the main pipeline config
XGBOOST_PARAMS = {**config.XGBOOST_PARAMS, "verbosity": 0}

# ── Target ─────────────────────────────────────────────────────────────────────
TARGET = "pct_delta"

# ── Feature sets ───────────────────────────────────────────────────────────────
# features.csv already has these pre-engineered for all 14 DMAs / 58 counties.
# Temporal alignment: calfresh_lag1 / foodbank_lag1 = Trends from 2 months prior
# to the SNAP date (1-month temporal shift in pipeline + 1-month lag in engineering).

# Experiment A: no Trends — controls for what autocorrelation + demographics explain
BASELINE_FEATURES = [
    "rate_lag1",        # SNAP application rate, 1-month lag (strongest predictor)
    "rate_lag2",        # SNAP application rate, 2-month lag
    "rate_lag3",        # SNAP application rate, 3-month lag
    "rate_roll3_mean",  # 3-month rolling mean of rate (smoothed level)
    "rate_roll3_std",   # 3-month rolling std (local volatility)
    "log_population",   # log₁₀(population) — size of county
    "log_income",       # log₁₀(median income) — wealth signal
    "income_quintile",  # 1–5 income rank within CA
    "month_sin",        # cyclical month encoding
    "month_cos",        # cyclical month encoding
    "quarter",          # coarser seasonality
]

# Experiment B: full feature set — adds both CalFresh + FoodBank Trends signals
TRENDS_FEATURES = BASELINE_FEATURES + [
    "calfresh_lag1",      # CalFresh Trends, 1-month lag
    "calfresh_lag2",      # CalFresh Trends, 2-month lag
    "calfresh_roll3",     # 3-month rolling mean of CalFresh Trends
    "calfresh_momentum",  # month-over-month change in CalFresh search interest
    "foodbank_lag1",      # FoodBank Trends, 1-month lag
    "foodbank_lag2",      # FoodBank Trends, 2-month lag
    "foodbank_roll3",     # 3-month rolling mean of FoodBank Trends
    "foodbank_momentum",  # month-over-month change in FoodBank search interest
]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Data loading and target construction
# ══════════════════════════════════════════════════════════════════════════════

def load_panel() -> pd.DataFrame:
    """
    Load features.csv (pre-engineered by the main pipeline) and add the
    pct_delta target variable.

    features.csv has 1,855 rows covering 58 counties × 14 DMAs with all
    engineered lag/rolling/momentum/demographic features. It already has
    SNAP_Applications for computing the delta target.

    pct_delta = (apps_t - apps_{t-1}) / apps_{t-1}
    Computed within county groups (sorted by date) to prevent leakage.
    Winsorised at ±PCT_DELTA_WINSOR to remove extreme outliers from
    months with very low baseline counts.
    """
    path = config.FEATURES_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"features.csv not found at {path}\n"
            "Run the main pipeline first: python run_pipeline.py"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"  features.csv: {len(df):,} rows, "
          f"{df['county'].nunique()} counties, "
          f"{df['metro_area'].nunique()} DMAs, "
          f"{df['date'].min().date()} – {df['date'].max().date()}")

    # Compute pct_delta within each county (sorted by date)
    df = df.sort_values(["county", "date"])
    apps_lag1    = df.groupby("county")["SNAP_Applications"].shift(1)
    df["pct_delta"] = (df["SNAP_Applications"] - apps_lag1) / apps_lag1

    # Winsorise at ±100% — rare extreme values in small counties
    before = df["pct_delta"].notna().sum()
    df["pct_delta"] = df["pct_delta"].clip(lower=-PCT_DELTA_WINSOR, upper=PCT_DELTA_WINSOR)
    print(f"  pct_delta: {before:,} non-null values, "
          f"winsorised at ±{PCT_DELTA_WINSOR*100:.0f}%")
    print(f"  pct_delta range (p5–p95): "
          f"{df['pct_delta'].quantile(0.05)*100:.1f}% – "
          f"{df['pct_delta'].quantile(0.95)*100:.1f}%")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray = None,
) -> dict:
    """
    Compute MAE, RMSE, R², directional accuracy, and spike recall.

    All inputs are in pct_delta units (e.g. 0.25 = +25% change).

    Directional accuracy: fraction where sign(pred) == sign(actual).
    Rows where actual == 0 are excluded (no direction to compare).

    Spike recall: among actual positive spikes (top SPIKE_PERCENTILE of
    training pct_deltas), what fraction did the model also call spikes?
    Threshold from training data only — no test contamination.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")

    # Directional accuracy
    nonzero = y_true != 0
    if nonzero.sum() > 0:
        dir_acc = float(np.mean(np.sign(y_pred[nonzero]) == np.sign(y_true[nonzero])))
    else:
        dir_acc = float("nan")

    # Spike recall
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
# STEP 3: Naive baseline model
# ══════════════════════════════════════════════════════════════════════════════

class NaiveNoChangeModel:
    """Predict 0% change for every county-month (no-change baseline)."""
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.zeros(len(X))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Walk-forward validation
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward(
    panel_df: pd.DataFrame,
    feature_cols: list,
    model_name: str,
    model_factory,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-ordered walk-forward validation.

    For each month T (after WALK_FORWARD_MIN_MONTHS of training history):
      - Train on all county-months with date < T
      - Predict all county-months with date == T
      - Record fold metrics and per-county-month predictions

    Returns:
      fold_df  — one row per test month
      pred_df  — one row per county-month (actual, predicted, residual)
    """
    required = feature_cols + [TARGET]
    clean    = panel_df.dropna(subset=required).copy()
    months   = sorted(clean["date"].unique())

    fold_rows = []
    pred_rows = []

    for test_month in months:
        train_mask = clean["date"] < test_month
        test_mask  = clean["date"] == test_month

        n_train_months = clean.loc[train_mask, "date"].nunique()
        if n_train_months < WALK_FORWARD_MIN_MONTHS or test_mask.sum() == 0:
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
            "model":  model_name,
            "month":  test_month,
            "n_test": int(test_mask.sum()),
            **metrics,
        })

        rows = clean.loc[test_mask, ["county", "date", "metro_area"]].copy()
        rows["actual_pct_delta"]    = y_test.values
        rows["predicted_pct_delta"] = y_pred
        rows["residual"]            = y_test.values - y_pred
        rows["model"]               = model_name
        pred_rows.append(rows)

    fold_df = pd.DataFrame(fold_rows)
    pred_df = (
        pd.concat(pred_rows, ignore_index=True)
        if pred_rows
        else pd.DataFrame(columns=["county", "date", "metro_area",
                                   "actual_pct_delta", "predicted_pct_delta",
                                   "residual", "model"])
    )
    return fold_df, pred_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Run all experiments
# ══════════════════════════════════════════════════════════════════════════════

def run_all_experiments(panel_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward for each model × feature-set combination:
      naive          — predict 0% change (floor baseline)
      lr_baseline    — linear regression, no Trends
      xgb_baseline   — XGBoost (tuned), no Trends
      lr_trends      — linear regression, full Trends features
      xgb_trends     — XGBoost (tuned), full Trends features  [primary]
    """
    experiments = [
        ("naive",         BASELINE_FEATURES, lambda: NaiveNoChangeModel()),
        ("lr_baseline",   BASELINE_FEATURES, lambda: LinearRegression()),
        ("xgb_baseline",  BASELINE_FEATURES, lambda: XGBRegressor(**XGBOOST_PARAMS)),
        ("lr_trends",     TRENDS_FEATURES,   lambda: LinearRegression()),
        ("xgb_trends",    TRENDS_FEATURES,   lambda: XGBRegressor(**XGBOOST_PARAMS)),
    ]

    all_folds = []
    all_preds = []

    for name, features, factory in experiments:
        print(f"  {name:<16} ...", end=" ", flush=True)
        fold_df, pred_df = walk_forward(panel_df, features, name, factory)
        all_folds.append(fold_df)
        all_preds.append(pred_df)

        if not fold_df.empty:
            avg = fold_df[["mae", "rmse", "r2", "directional_accuracy"]].mean()
            print(
                f"MAE={avg['mae']*100:>5.1f}pp  "
                f"RMSE={avg['rmse']*100:>5.1f}pp  "
                f"R²={avg['r2']:>6.3f}  "
                f"DirAcc={avg['directional_accuracy']:.3f}"
            )
        else:
            print("(no results — check feature availability)")

    return (
        pd.concat(all_folds, ignore_index=True),
        pd.concat(all_preds, ignore_index=True),
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: XGBoost feature importance (full-data fit)
# ══════════════════════════════════════════════════════════════════════════════

def get_xgb_feature_importance(panel_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Train XGBoost on full data and return feature importances (in-sample diagnostic)."""
    clean = panel_df.dropna(subset=feature_cols + [TARGET])
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(clean[feature_cols], clean[TARGET])
    return (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_actual_vs_predicted(pred_df: pd.DataFrame, model_name: str) -> None:
    """Scatter plot of actual vs predicted pct_delta (as %)."""
    sub = pred_df[pred_df["model"] == model_name].dropna(
        subset=["actual_pct_delta", "predicted_pct_delta"]
    )
    if sub.empty:
        return

    actual = sub["actual_pct_delta"] * 100
    pred   = sub["predicted_pct_delta"] * 100
    lim    = max(abs(actual.max()), abs(actual.min())) * 1.1

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, pred, alpha=0.25, s=8, color="steelblue")
    ax.axline((0, 0), slope=1, color="red", linewidth=1, label="perfect prediction")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Actual % change in applications")
    ax.set_ylabel("Predicted % change in applications")
    ax.set_title(f"Actual vs Predicted Percent Delta — {model_name}")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"actual_vs_predicted_{model_name}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_residuals(pred_df: pd.DataFrame, model_name: str) -> None:
    """Residual plot (predicted vs residual) in percentage-point units."""
    sub = pred_df[pred_df["model"] == model_name].dropna(
        subset=["predicted_pct_delta", "residual"]
    )
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(sub["predicted_pct_delta"] * 100, sub["residual"] * 100,
               alpha=0.25, s=8, color="steelblue")
    ax.axhline(0, color="red", linewidth=1)
    ax.set_xlabel("Predicted % change")
    ax.set_ylabel("Residual (pp)")
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
    ax.set_title("Does the model correctly predict increase vs decrease?")
    ax.legend()
    for bar, val in zip(bars, summary.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "directional_accuracy_by_model.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_feature_importance(fi_df: pd.DataFrame) -> None:
    """Horizontal bar chart of feature importances for xgb_trends."""
    fig, ax = plt.subplots(figsize=(7, max(4, len(fi_df) * 0.35)))
    ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="steelblue")
    ax.set_xlabel("Feature importance (XGBoost gain)")
    ax.set_title("Feature Importance — xgb_trends (pct_delta target)")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "feature_importance_xgb_trends.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 65)
    print("DELTA PREDICTION EXPERIMENT  (target: pct_delta)")
    print("=" * 65)

    # ── 1. Load data ────────────────────────────────────────────────────────
    print("\n[1] Loading data ...")
    panel_df = load_panel()

    # Panel summary
    baseline_ok = panel_df.dropna(subset=BASELINE_FEATURES + [TARGET])
    trends_ok   = panel_df.dropna(subset=TRENDS_FEATURES   + [TARGET])
    dmas        = sorted(panel_df["metro_area"].dropna().unique())

    print(f"\n  ── Panel summary ─────────────────────────────────────────")
    print(f"  Counties : {panel_df['county'].nunique()}")
    print(f"  Months   : {panel_df['date'].nunique()}  "
          f"({panel_df['date'].min().date()} – {panel_df['date'].max().date()})")
    print(f"  DMAs     : {len(dmas)}  ({', '.join(dmas)})")
    print(f"  Target   : {TARGET}  (percent change, winsorised at ±{PCT_DELTA_WINSOR*100:.0f}%)")
    print(f"\n  Baseline features ({len(BASELINE_FEATURES)}): {BASELINE_FEATURES}")
    print(f"  Trends features  ({len(TRENDS_FEATURES)}): {TRENDS_FEATURES}")
    print(f"\n  Complete rows — baseline : {len(baseline_ok):,}")
    print(f"  Complete rows — trends   : {len(trends_ok):,}")
    print(f"  ──────────────────────────────────────────────────────────")

    # ── 2. Walk-forward experiments ──────────────────────────────────────────
    print("\n[2] Running walk-forward experiments  (pp = percentage points) ...")
    fold_df, pred_df = run_all_experiments(panel_df)

    # ── 3. Summary table ─────────────────────────────────────────────────────
    print("\n[3] Overall summary (mean across folds) ...")
    summary = (
        fold_df
        .groupby("model")[["mae", "rmse", "r2", "directional_accuracy", "spike_recall"]]
        .mean()
        .sort_values("mae")
    )
    # Display MAE/RMSE in percentage-point units for readability
    display = summary.copy()
    display["mae_pp"]  = (display["mae"]  * 100).round(2)
    display["rmse_pp"] = (display["rmse"] * 100).round(2)
    display["r2"]      = display["r2"].round(4)
    display["directional_accuracy"] = display["directional_accuracy"].round(4)
    display["spike_recall"]         = display["spike_recall"].round(4)
    print()
    print(display[["mae_pp", "rmse_pp", "r2", "directional_accuracy", "spike_recall"]].to_string())

    # ── 4. Feature importance ────────────────────────────────────────────────
    print("\n[4] Feature importance — xgb_trends (full-data fit, in-sample) ...")
    fi = get_xgb_feature_importance(panel_df, TRENDS_FEATURES)
    print(fi.to_string(index=False))

    # ── 5. Baseline vs Trends comparison ────────────────────────────────────
    print("\n[5] Does adding CalFresh + FoodBank Trends improve percent-delta prediction?")
    for mtype in ["lr", "xgb"]:
        base_key   = f"{mtype}_baseline"
        trends_key = f"{mtype}_trends"
        if base_key not in summary.index or trends_key not in summary.index:
            continue
        base_row   = summary.loc[base_key]
        trends_row = summary.loc[trends_key]
        mae_pct    = (base_row["mae"] - trends_row["mae"]) / base_row["mae"] * 100
        r2_delta   = trends_row["r2"] - base_row["r2"]
        dir_delta  = trends_row["directional_accuracy"] - base_row["directional_accuracy"]
        sign       = "+" if mae_pct >= 0 else ""
        print(f"\n  {mtype.upper()} — adding Trends:")
        print(f"    MAE : {base_row['mae']*100:.2f}pp → {trends_row['mae']*100:.2f}pp  "
              f"({sign}{mae_pct:.1f}% improvement)")
        print(f"    R²  : {base_row['r2']:.4f} → {trends_row['r2']:.4f}  ({r2_delta:+.4f})")
        print(f"    Dir : {base_row['directional_accuracy']:.4f} → "
              f"{trends_row['directional_accuracy']:.4f}  ({dir_delta:+.4f})")

    best = summary["mae"].idxmin()
    print(f"\n  Best model by MAE: {best}")

    # ── 6. Save outputs ──────────────────────────────────────────────────────
    print(f"\n[6] Saving outputs to {OUT_DIR} ...")
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

    # ── 7. Plots ─────────────────────────────────────────────────────────────
    print("\n[7] Generating plots ...")
    plot_actual_vs_predicted(pred_df, "xgb_trends")
    plot_actual_vs_predicted(pred_df, "xgb_baseline")
    plot_residuals(pred_df, "xgb_trends")
    plot_directional_accuracy(fold_df)
    plot_feature_importance(fi)

    print("\n" + "=" * 65)
    print("Done.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
